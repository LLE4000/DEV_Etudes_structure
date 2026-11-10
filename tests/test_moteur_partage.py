# -*- coding: utf-8 -*-
"""Le MOTEUR PARTAGÉ (lot S4, docs/site/REFONTE_ETABLI.md, S4-D5, S4-D6, S4-D10 et C1, C4, C5, C13, C14) : un seul
travailleur Pyodide pour toute la session du document sert TOUS les écrans — et rend exactement ce que rendent des
moteurs neufs.

    /usr/bin/python3.12 tests/test_moteur_partage.py              # P0 à P5 et les essais du client
    /usr/bin/python3.12 tests/test_moteur_partage.py --mutations  # en plus : chaque contrôle rougit sous sa mutation
    /usr/bin/python3.12 tests/test_moteur_partage.py --mutations=m_saut,m_chaine   # ces mutations-là seulement

Tout se joue dans node, avec le Pyodide DÉPOSÉ (site/web/vendor/pyodide-0.29.5) et les VRAIS fichiers du site :
js/moteur.js est importé tel quel par une « page » node dont le ``Worker`` est un worker_thread qui charge le VRAI
js/travailleur.js, avec une cale (self, postMessage, onmessage, fetch des fichiers du site, crypto.subtle de node) et un
crochet de chargement autour de pyodide.mjs (node ne lit pas une adresse file:// comme indexURL ; le crochet garde aussi
l'objet Pyodide pour provoquer une erreur fatale, P5). Rien du site n'est modifié pour l'essai.

P0 · le paquet expédié (moteur/moteur.zip) est À JOUR : chaque fichier y est celui du dépôt, les paquets entiers du site
     y sont complets, le manifeste porte son empreinte et celle du paquet de la note. Sinon KO « reconstruire »
     (/usr/bin/python3.12 site/web/construire.py) et rien d'autre n'est joué : une preuve dans Pyodide sur un paquet
     périmé ne prouverait rien (C13 : le commit du registre paresseux l'avait laissé périmé).
P1 · la RÉFÉRENCE (tests/test_reference_refonte.py, inchangé, ses photos gelées par CPython) jouée dans UN interpréteur
     Pyodide, sur le paquet expédié : l'ordre du registre, puis l'ordre inverse dans le MÊME interpréteur (tous les
     modules déjà chargés) ; 0 échec dans les deux, et CPython, sur les mêmes photos, 0 échec aussi.
P2 · le VRAI travailleur, démarré SANS module : préchauffage puis « nouveau » des 17 établis dans un ordre ENTRELACÉ
     figé (P(k₀), prêt, P(k₁), N(k₀), P(k₂), N(k₁)… N(k₁₆)), puis une seconde passe en ordre inverse ; chaque réponse est
     comparée À L'OCTET à celle de CPython dans un processus NEUF par module (ce que construire.py écrit en aperçu).
     Puis les BIBLIOTHÈQUES (BIBLIO_REQUETES : béton et acier, rubriques variées, un cas Ø25 C40/50, la page et le
     panneau d'une note), dans le MÊME interpréteur, après tous les établis : ordre direct puis inverse, chacune = CPython
     dans un processus neuf, à l'octet (aucune exception). La mutation m_biblio_fuite (un établi qui retouche à son
     import une table partagée de beton/, le coefficient d'adhérence des ancrages) la fait rougir.
     Deux exceptions NOMMÉES, et elles seules (la liste rougit si elle grandit OU rétrécit) :
       - soudures numérote ses hachures par un compteur de processus (acier/soudures/dessin.py) : égale à l'octet au
         premier dessin, égale après normalisation de ``(id="|url(#|href="#)s<n>_`` aux suivants ;
       - garde_corps : Pyodide et CPython diffèrent au DERNIER chiffre de coordonnées SVG des vues « coupe »,
         « platine », « section » (13.81774801564556 contre 13.817748015645558) : ces nombres-là à 1e-9 relatif, tout
         le reste à l'identique.
     Sans la sonde, le travailleur n'émet que etape, pret, reponse (C14 : aucun message de plus).
P3 · la NOTE PDF des modules NOTE_PDF, demandée APRÈS tous les autres modules (numpy chargé par un autre), = CPython
     (processus neuf), à la normalisation NOMMÉE près : /CreationDate, /ModDate, /ID (l'identifiant change à chaque
     note, même dans un seul processus). P3b : un travailleur dont le manifeste ne charge numpy pour personne, dont la
     PREMIÈRE requête est la note de garde_corps : numpy est chargé AVANT la note (manifeste « note.avec_numpy », par la
     clé lue dans la requête), sans rejeu par le guet.
P4 · le préchauffage ISOLÉ (un travailleur neuf par clé, ?sonde=1) : {api_web.<clé> des modules et bibliothèques} ∩
     sys.modules = ce que CPython importe pour ``import api_web.<clé>`` ; numpy chargé ⇔ la clé est dans avec_numpy
     (pour une clé que le relevé du paquet ne couvre pas — une bibliothèque, tant que construire.py ne la relève pas —
     le guet charge numpy si l'import le demande). Le GUET (manifeste « avec_numpy » vide) : la requête de garde_corps
     est rejouée, sa réponse = CPython, guet.demande remis à faux, la requête suivante (poutre) n'est pas rejouée ; un
     préchauffage par le guet importe bien son module. Un préchauffage n'est jamais traité DEVANT une vraie requête
     arrivée derrière lui (il est sauté) ; tous les messages passent par UNE chaîne (une note qui attend son paquet ne
     se fait pas doubler) ; la sonde date la première requête de CHAQUE clé.
     Les parties du lot « moteur » (revue S4), jouées dans un second processus du même pilote : l'ANNONCE (plancher annoncé
     avant « pret » : numpy part pendant le démarrage, en parallèle du paquet, rien n'est importé ; le témoin sans annonce
     le voit après « pret ») ; le FOND (C4 : numpy retardé de RETARD_NUMPY ms — un préchauffage de soudures ne retient
     ni une vraie requête de poutre arrivée derrière lui, ni « pret » s'il est posté pendant le démarrage ; pendant le
     chargement en fond, le guet refuse l'import de numpy ; « sauté » est revu avant l'import) ; le GUET HORS LIGNE
     (numpy refusé : la requête — ou le préchauffage — qui l'a déclenché garde son refus, les suivantes répondent).
P6 · la VERSION du site : l'adresse du travailleur porte celle du document (« v ») ; un manifeste d'une autre version →
     pas de « pret », l'erreur nommée MIS_A_JOUR (périmé) ; de même pour un travailleur RECRÉÉ après une mort, le site
     mis à jour entre-temps ; une note PDF dont l'empreinte ne correspond plus → le manifeste relu, l'erreur nommée s'il
     a changé de version (le message d'empreinte sinon, le témoin).
P5 · un travailleur qui TOMBE (erreur fatale de Pyodide, ou fil arrêté) : les requêtes en attente et en vol rendent
     {ok:false, mort:true}, le travailleur est arrêté, la demande suivante en recrée un (le même Moteur) et répond
     juste ; l'ancienne façon (new Moteur(surEtape, module), moteur.demander(fabrique)) marche avec le vrai travailleur.
CLIENT · js/moteur.js contre un faux travailleur (sans Pyodide) : la file (une requête à la fois, fabrique() à son
     tour, après que la réponse précédente a été appliquée), fermer() (lectures abandonnées EcranFerme, modifications
     gardées, promesse résolue quand tout est fait), le préchauffage (jamais posté tant qu'une requête attend, la plus
     récente remplace l'autre, une clé une fois), la mort et la recréation, le setter onmessage (le harnais
     l'enveloppe), l'adresse du travailleur (sans module ; « sonde=1 » sous la sonde seulement ; « v », la version du
     document, quand il en porte une — un travailleur recréé aussi), l'annonce (postée aussitôt, une fois par clé), le
     démarrage RATÉ du moteur partagé (relancé par moteurPartage(), jamais par une requête ; pas s'il est périmé), les
     messages de la sonde (globalThis.__sondes) et la sonde par ouverture (js/sonde.js, nouvelleOuverture).

--mutations : pour chaque contrôle, le moteur (js/moteur.js, js/travailleur.js, js/sonde.js) modifié D'UN TRAIT dans une
COPIE du site (vendor et moteur liés, jamais recopiés), le pilote concerné rejoué, et le contrôle doit ROUGIR ; P0 sur
un paquet dont un fichier diffère, P1 sur une formule du moteur poutre changée de 0,1 % dans l'interpréteur — qui ne
compte que si le texte a été muté (« @@MUTEE ») : un interpréteur arrêté ou un texte absent rougit sous un nom à part,
jamais comme « rougit » (m_formule_garde le vérifie sur un texte absent) ; la passe des bibliothèques sur un paquet
recopié dont un établi retouche une table partagée (m_biblio_fuite, le manifeste à sa nouvelle empreinte).
"""
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(RACINE, "site", "web")
sys.path.insert(0, WEB)
sys.path.insert(0, RACINE)
import construire as SW  # noqa: E402
from api_web import registre as REG  # noqa: E402
from api_web import routeur as ROUTEUR  # noqa: E402

VENDOR = os.path.join(WEB, SW.PYODIDE)
PY = sys.executable
ENV = dict(os.environ, TZ="Europe/Paris")
ETABLIS = [k for k in SW.ADRESSES if k not in SW.REDIRECTIONS]                # les 17 établis (predalle est servi par dalle)
BIBLIOS = ["biblio_" + k for k in ROUTEUR.BIBLIOTHEQUES]
CLES_PRECHAUFFAGE = list(REG.cles()) + BIBLIOS
# l'ordre ENTRELACÉ figé de P2 (une permutation écrite, ni le registre ni son inverse)
ORDRE_P2 = ["garde_corps", "poutre", "soudures", "age", "dalle", "bois", "assemblages", "taux", "corniere", "sol",
            "catalogue", "enrobage", "plancher", "donnees_beton", "flambement", "armatures", "profil"]
# les exceptions NOMMÉES de P2 : (clé, passe) → ce qui est admis
SOUDURES_ID = re.compile(r'(id=\\?"|url\(#|href=\\?"#)s\d+_')          # (les guillemets sont échappés dans le JSON)
VUES_ULP = {"garde_corps": ("coupe", "platine", "section")}
TOLERANCE_ULP = 1e-9
EXCEPTIONS_P2 = {("soudures", 2): "compteur s<n>_", ("garde_corps", 1): "ulp", ("garde_corps", 2): "ulp"}
# la note PDF : ce qui change à chaque note, même dans un seul processus
NORMALISATIONS_PDF = [(rb"/(CreationDate|ModDate) \(D:[^)]*\)", rb"/\1 (D:)"),
                      (rb"/ID\s*\[\s*<[0-9a-fA-F]+>\s*<[0-9a-fA-F]+>\s*\]", rb"/ID [<><>]")]
INFOS_NOTE = {"nom": "Note de l'essai", "chantier": "Chantier de l'essai", "indice": "A", "date": "10/11/2026"}
# les BIBLIOTHÈQUES de P2 (après les établis, dans le même interpréteur) : des rubriques variées des deux, dont le cas
# Ø25 C40/50 des ancrages (page et panneau d'une note) ; chacune comparée à CPython dans un processus NEUF, à l'octet
# (aucun nombre « 50.0 » : JSON.stringify l'écrirait « 50 », la requête doit être la même des deux côtés)
BIBLIO_REQUETES = [
    {"bibliotheque": "beton", "rubrique": "betons", "params": {}},
    {"bibliotheque": "beton", "rubrique": "betons", "params": {"classe": "C40/50", "alpha_cc": 0.85}},
    {"bibliotheque": "beton", "rubrique": "aciers", "params": {"vue": "treillis"}},
    {"bibliotheque": "beton", "rubrique": "sections", "params": {"d": 25, "n": 3}},
    {"bibliotheque": "beton", "rubrique": "sections", "params": {"mode": "nappe", "d": 12, "espacement": "e15"}},
    {"bibliotheque": "beton", "rubrique": "enrobages", "params": {"exposition": "XC4", "classe": "C40/50", "phi": 25, "vue": "feu"}},
    {"bibliotheque": "beton", "rubrique": "ancrages", "params": {}},
    {"bibliotheque": "beton", "rubrique": "ancrages", "params": {"phi": 25, "classe": "C40/50"}},
    {"bibliotheque": "beton", "rubrique": "ancrages", "params": {"phi": 25, "classe": "C40/50"}, "hote": "note"},
    {"bibliotheque": "beton", "rubrique": "ancrages", "params": {"grandeur": "l0", "phi": 25, "classe": "C40/50", "rho1": 50}},
    {"bibliotheque": "beton", "rubrique": "ancrages", "params": {"phi": 32, "classe": "C25/30", "conditions": "mediocres", "sollicitation": "compression"}},
    {"bibliotheque": "acier", "rubrique": "profils", "params": {}},
    {"bibliotheque": "acier", "rubrique": "profils", "params": {"familles": "HEA", "profil": "HEA 200", "tri": "Iy", "ordre": "decroissant"}},
    {"bibliotheque": "acier", "rubrique": "nuances", "params": {"nuance": "S235", "epaisseur": "80"}},
    {"bibliotheque": "acier", "rubrique": "boulons", "params": {"boulon": "M20", "classe": "10.9"}},
    {"bibliotheque": "acier", "rubrique": "boulons", "params": {"vue": "classes"}},
]
BIBLIO_REQUETES = [{"action": "consulter", **q} for q in BIBLIO_REQUETES]
# la version du site construit (la balise stab-version des pages = celle du manifeste) ; numpy RETARDÉ de cette durée
# dans les essais du fond (C4 : ce qui l'attend se voit)
VERSION_SITE = json.load(open(os.path.join(WEB, "moteur", "manifeste.json"), encoding="utf-8"))["version"]
RETARD_NUMPY = 4000
MIS_A_JOUR = "Le site a été mis à jour : rechargez la page"   # COPIE NOMMÉE de js/travailleur.js (vérifiée ci-dessous)


class Bilan:
    def __init__(self, bavard=True):
        self.ok, self.ko, self.bavard = [], [], bavard

    def chk(self, nom, cond, info=""):
        (self.ok if cond else self.ko).append((nom, info))
        if self.bavard:
            print(("OK   " if cond else "KO   ") + nom + ("" if cond else "\n        | " + str(info)[:3000].replace("\n", "\n        | ")))
        return cond


# ====================================================================== les fichiers node (écrits dans un dossier à part)
CROCHETS = r"""
// le crochet de chargement : pyodide.mjs importé par js/travailleur.js passe par l'enveloppe de l'essai
export async function resolve(spec, ctx, next) {
  if (spec.endsWith("vendor/pyodide-0.29.5/pyodide.mjs") && ctx.parentURL && /\/js\/travailleur\.js(\?|$)/.test(ctx.parentURL))
    return { url: new URL("./pyo.mjs", import.meta.url).href, shortCircuit: true };
  return next(spec, ctx);
}
"""
PYO = r"""
// l'enveloppe de loadPyodide : indexURL file:// → chemin (node), et l'objet Pyodide gardé (erreur fatale provoquée, P5) ;
// numpy RETARDÉ (workerData.numpy.retard, ms : un réseau lent, pour voir ce qui l'attend) ou REFUSÉ (.refus : hors ligne)
import { loadPyodide as vrai } from %(pyodide)s;
import { fileURLToPath } from "node:url";
import { workerData } from "node:worker_threads";
export async function loadPyodide(o = {}) {
  const indexURL = o.indexURL && o.indexURL.startsWith("file:") ? fileURLToPath(o.indexURL) : o.indexURL;
  const py = await vrai({ ...o, indexURL });
  globalThis.__py = py;
  const N = (workerData && workerData.numpy) || null;
  if (N) {
    const orig = py.loadPackage.bind(py);
    py.loadPackage = async (nom, opts) => {
      if (N.refus) throw new TypeError("Failed to fetch numpy (hors ligne : refus de l'essai)");
      if (N.retard) await new Promise((ok) => setTimeout(ok, N.retard));
      return orig(nom, opts);
    };
  }
  return py;
}
"""
CALE = r"""
// le fil du travailleur : la cale d'un Web Worker, puis le VRAI js/travailleur.js
import { parentPort, workerData } from "node:worker_threads";
import { register } from "node:module";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
register(workerData.crochets);
globalThis.self = globalThis;
globalThis.onmessage = null;
globalThis.location = { href: workerData.href };
globalThis.postMessage = (m) => parentPort.postMessage(m);
// fetch : les fichiers du site, rien d'autre (le manifeste peut être servi retouché : workerData.manifeste, puis le
// message de cale « manifeste » — une mise en ligne pendant la session ; « note.<k> » retouche le manifeste de la note)
let retouches = workerData.manifeste || null;
globalThis.fetch = async (u) => {
  const url = new URL(String(u));
  if (url.protocol !== "file:") throw new TypeError("réseau interdit dans l'essai : " + url.href);
  const chemin = fileURLToPath(url.href.replace(/[?#].*$/, ""));
  let b = await readFile(chemin);
  if (retouches && chemin.endsWith("/moteur/manifeste.json")) {
    const j = JSON.parse(b.toString("utf8"));
    for (const [k, v] of Object.entries(retouches)) { if (k.startsWith("note.")) j.note[k.slice(5)] = v; else j[k] = v; }
    b = Buffer.from(JSON.stringify(j));
  }
  return new Response(b);
};
// les messages arrivés avant que le script ait posé onmessage attendent (comme dans un vrai Worker)
const enAttente = [];
let charge = false;
const livrer = (data) => { if (typeof globalThis.onmessage === "function") globalThis.onmessage({ data }); };
parentPort.on("message", (data) => {
  if (data && data.__cale) {
    if (data.__cale === "fatal") { try { globalThis.__py._api.fatal_error(new Error("erreur fatale provoquée par l'essai")); } catch (e) { /* levée par Pyodide */ } }
    else if (data.__cale === "sortir") process.exit(7);
    else if (data.__cale === "manifeste") retouches = data.retouches;
    return;
  }
  if (charge && !enAttente.length) livrer(data); else enAttente.push(data);
});
await import(workerData.script);
charge = true;
// (une tâche par message, comme le navigateur : les micro-tâches passent entre deux messages)
const vider = () => { if (enAttente.length) { livrer(enAttente.shift()); setImmediate(vider); } };
vider();
"""
PAGE = r"""
// la page vue de node : son Worker est un worker_thread qui joue le VRAI js/travailleur.js
import { Worker as FilNode } from "node:worker_threads";
export const travailleurs = [];
const CALE = new URL("./cale.mjs", import.meta.url);
const CROCHETS = new URL("./crochets.mjs", import.meta.url).href;
export class TravailleurNode {
  constructor(url, opts) {
    this.url = String(url); this.opts = opts;
    this._f = null; this.onerror = null; this.termine = false;
    this.types = []; this.postes = []; this.reponses = new Map(); this.ordre = []; this.dernier = null; this.recus = [];
    travailleurs.push(this);
    this.nw = new FilNode(CALE, { workerData: { href: this.url, script: this.url, crochets: CROCHETS, manifeste: globalThis.__manifeste || null,
      numpy: globalThis.__numpy || null }, stderr: true, stdout: true });
    this.nw.stderr.on("data", () => {}); this.nw.stdout.on("data", () => {});
    this.nw.on("message", (d) => {
      this.types.push(d && d.type); this.recus.push(d);
      if (d && d.type === "reponse") { this.reponses.set(d.id, d.json); this.ordre.push(d.id); this.dernier = d.json; }
      if (this._f) this._f({ data: d });
    });
    this.nw.on("error", (e) => { if (!this.termine && this.onerror) this.onerror({ message: String(e && e.message || e) }); });
    this.nw.on("exit", (c) => { const avant = this.termine; this.termine = true; if (!avant && this.onerror) this.onerror({ message: `le travailleur s'est arrêté (code ${c})` }); });
  }
  set onmessage(f) { this._f = f; }
  get onmessage() { return this._f; }
  postMessage(m) { this.postes.push(m && m.type ? m.type : "requete"); if (!this.termine) this.nw.postMessage(m); }
  terminate() { this.termine = true; this.nw.terminate(); }
  cale(quoi, x = {}) { this.nw.postMessage({ __cale: quoi, ...x }); }
}
export function preparer(sonde) {
  globalThis.self = globalThis;
  globalThis.location = { href: "http://essai.invalid/poutre" + (sonde ? "?sonde=1" : "") };
  globalThis.Worker = TravailleurNode;
}
export const tic = (ms = 0) => new Promise((ok) => setTimeout(ok, ms));
export async function finir(resultat, fichier) {
  const { writeFileSync } = await import("node:fs");
  writeFileSync(fichier, JSON.stringify(resultat));
  for (const w of travailleurs) { try { w.terminate(); } catch (e) { /* déjà arrêté */ } }
  process.exit(0);
}
"""

# P1 : la référence dans UN interpréteur Pyodide (sans travailleur : le paquet expédié, déballé, numpy chargé)
PILOTE_P1 = r"""
import { loadPyodide } from %(pyodide)s;
import { readFileSync } from "node:fs";
const A = JSON.parse(process.argv[2]);
const py = await loadPyodide({ indexURL: A.vendor, stdout: (s) => console.log(s), stderr: () => {} });
await py.loadPackage("numpy", { messageCallback: () => {} });
py.unpackArchive(new Uint8Array(readFileSync(A.zip)), "zip", { extractDir: "/moteur" });
// une mutation ne vaut que si elle a MUTÉ (elle pose MUTEE) : sinon rien n'est joué, et le pilote le dit
if (A.mutation) {
  py.runPython(A.mutation);
  if (!py.globals.get("MUTEE")) { console.log("@@NON_MUTEE"); process.exit(0); }
  console.log("@@MUTEE");
}
py.FS.mkdirTree("/moteur/tests");
py.FS.mount(py.FS.filesystems.NODEFS, { root: A.tests }, "/moteur/tests");
for (const [i, args] of A.ordres.entries()) {
  console.log("@@ORDRE " + i);
  py.globals.set("ARGS", py.toPy(args));
  py.runPython(`
import sys, runpy
sys.dont_write_bytecode = True
sys.argv = ["test_reference_refonte.py"] + list(ARGS)
try:
    runpy.run_path("/moteur/tests/test_reference_refonte.py", run_name="__main__")
except SystemExit as e:
    print("@@SORTIE", e.code)
`);
}
console.log("@@FIN");
process.exit(0);
"""

# P2 + P3 : un seul moteur partagé, sans la sonde
PILOTE_PARTAGE = r"""
import { preparer, travailleurs, tic, finir } from "./page.mjs";
const A = JSON.parse(process.argv[2]);
preparer(false);
const M = await import(A.moteur);
const out = { passe1: {}, passe2: {}, notes: {}, ms: {} };
const t0 = performance.now();
const m = M.moteurPartage();
const c = m.client();
const K = A.ordre;
m.prechauffer(K[0]); await tic();
await m.pret;
out.ms.pret = Math.round(performance.now() - t0);
const w = travailleurs[0];
const nouveau = async (k) => { const r = await c.demander(() => ({ module: k, action: "nouveau" })); return { ok: r.ok, json: w.dernier }; };
for (let i = 0; i < K.length; i++) {
  if (i + 1 < K.length) { m.prechauffer(K[i + 1]); await tic(5); }
  out.passe1[K[i]] = (await nouveau(K[i])).json;
}
for (const k of [...K].reverse()) out.passe2[k] = (await nouveau(k)).json;
out.ms.nouveaux = Math.round(performance.now() - t0);
// les BIBLIOTHÈQUES, dans le MÊME interpréteur, après tous les établis : ordre direct, puis inverse
const consulter = async (q) => { await c.demander(() => q); return w.dernier; };
out.biblio1 = []; out.biblio2 = new Array(A.biblio.length).fill(null);
for (const q of A.biblio) out.biblio1.push(await consulter(q));
for (let i = A.biblio.length - 1; i >= 0; i--) out.biblio2[i] = await consulter(A.biblio[i]);
out.ms.biblio = Math.round(performance.now() - t0);
for (const k of A.notes) {
  const n = await c.demander(() => ({ module: k, action: "nouveau" }));
  await c.demander(() => ({ module: k, action: "telecharger", etat: n.etat, quoi: "note_pdf", infos: A.infos }), { note: true });
  out.notes[k] = w.dernier;
}
out.ms.notes = Math.round(performance.now() - t0);
out.url = w.url; out.travailleurs = travailleurs.length; out.types = [...new Set(w.types)].sort();
out.postes = w.postes.filter((t) => t === "prechauffer").length;
await finir(out, A.sortie);
"""

# P3b, P4, la chaîne, le saut, la sonde par clé : des travailleurs NEUFS, sous la sonde
PILOTE_SONDE = r"""
import { preparer, travailleurs, TravailleurNode, tic, finir } from "./page.mjs";
const A = JSON.parse(process.argv[2]);
preparer(true);
const M = await import(A.moteur);
const S = await import(A.sonde);
const out = { isolement: {}, url: null };
// un moteur neuf : son manifeste servi retouché, son numpy retardé ou refusé (pour CE travailleur seul)
const neuf = (manifeste = null, numpy = null) => {
  globalThis.__manifeste = manifeste; globalThis.__numpy = numpy;
  const m = new M.Moteur();
  globalThis.__manifeste = null; globalThis.__numpy = null;
  return m;
};
const travailleurDe = (m) => m.w;
const sondesDe = (w, cle) => w.recus.filter((d) => d.type === "sonde" && d.cle === cle);
const attendreQue = async (f, ms = 60000) => { const t = performance.now(); while (!f()) { if (performance.now() - t > ms) return false; await tic(20); } return true; };
const borne = (p, ms = 120000) => Promise.race([p, new Promise((ok) => setTimeout(() => ok({ __delai: true }), ms))]);
const pretOuErreur = (m) => borne(m.pret.then(() => "pret", (e) => "erreur : " + e.message));
const marques = (s) => (s ? Object.fromEntries(s.marques) : {});
const N = (k) => () => ({ module: k, action: "nouveau" });
const parties = new Set(A.parties);

if (parties.has("isolement")) {
  const file = [...A.cles];
  const un = async (k) => {
    const m = neuf();
    const w = travailleurDe(m);
    out.url = out.url || w.url;
    m.prechauffer(k); await tic(5);
    // (le préchauffage d'une clé avec numpy finit APRÈS numpy, chargé en fond — C4, C5 : on attend sa fin, puis on lit
    //  sys.modules ; « le harnais suit » : il lisait aussitôt, quand numpy se chargeait dans la chaîne)
    await attendreQue(() => w.recus.some((d) => d.type === "prechauffe"), 60000);
    const sm = await m.sondeModules();
    out.isolement[k] = { ...sm, prechauffe: w.recus.filter((d) => d.type === "prechauffe"), postes: w.postes };
    w.terminate();
  };
  const fils = Array.from({ length: A.parallele }, async () => { while (file.length) await un(file.shift()); });
  await Promise.all(fils);
}
if (parties.has("guet_requete")) {
  const m = neuf({ avec_numpy: [] }); const c = m.client(); const w = travailleurDe(m);
  await c.demander(() => ({ module: "garde_corps", action: "nouveau" }));
  const json = w.dernier;
  const sm1 = await m.sondeModules();
  await c.demander(() => ({ module: "poutre", action: "nouveau" }));
  await attendreQue(() => sondesDe(w, "poutre").length);
  out.guet_requete = { json, sm1, sonde_gc: sondesDe(w, "garde_corps").map((s) => s.details), sonde_poutre: sondesDe(w, "poutre").map((s) => s.details), sm2: await m.sondeModules() };
  w.terminate();
}
if (parties.has("guet_prechauffer")) {
  const m = neuf({ avec_numpy: [] }); const w = travailleurDe(m);
  m.prechauffer("garde_corps"); await tic(5);
  await attendreQue(() => w.recus.some((d) => d.type === "prechauffe"), 60000);   // (numpy en fond : on attend la fin)
  out.guet_prechauffer = { sm: await m.sondeModules(), prechauffe: w.recus.filter((d) => d.type === "prechauffe") };
  w.terminate();
}
if (parties.has("note_numpy")) {
  const m = neuf({ avec_numpy: [] }); const c = m.client(); const w = travailleurDe(m);
  await c.demander(() => ({ module: "garde_corps", action: "telecharger", etat: A.etat_gc, quoi: "note_pdf", infos: A.infos }), { note: true });
  const json = w.dernier;
  await attendreQue(() => w.recus.some((d) => d.type === "sonde"), 20000);
  out.note_numpy = { json, sonde: sondesDe(w, "garde_corps").map((s) => ({ details: s.details, marques: s.marques.map((x) => x[0]) })), sm: await m.sondeModules() };
  w.terminate();
}
if (parties.has("saut")) {
  // le travailleur seul (sans la file de la page) : deux préchauffages puis une requête, postés avant « pret »
  const w = new TravailleurNode(new URL("./travailleur.js?sonde=1", A.moteur), { type: "module" });
  w.postMessage({ type: "prechauffer", cle: "dalle" });
  w.postMessage({ type: "prechauffer", cle: "bois" });
  w.postMessage({ id: 1, requete: JSON.stringify({ module: "poutre", action: "nouveau" }), note: false });
  w.postMessage({ type: "sonde_modules" });
  await attendreQue(() => w.recus.some((d) => d.type === "sonde_modules"));
  out.saut = { prechauffe: w.recus.filter((d) => d.type === "prechauffe"), sm: w.recus.find((d) => d.type === "sonde_modules"), reponse: w.reponses.has(1) };
  w.terminate();
}
if (parties.has("chaine")) {
  // une note (qui attend son paquet) puis un calcul, postés d'un coup : les réponses dans l'ordre d'arrivée
  const w = new TravailleurNode(new URL("./travailleur.js?sonde=1", A.moteur), { type: "module" });
  w.postMessage({ id: 1, requete: JSON.stringify({ module: "poutre", action: "telecharger", etat: A.etat_poutre, quoi: "note_pdf", infos: A.infos }), note: true });
  w.postMessage({ id: 2, requete: JSON.stringify({ module: "bois", action: "nouveau" }), note: false });
  await attendreQue(() => w.ordre.length >= 2);
  out.chaine = { ordre: w.ordre, ok: [1, 2].map((i) => { try { return JSON.parse(w.reponses.get(i)).ok; } catch (e) { return false; } }) };
  w.terminate();
}
if (parties.has("par_cle")) {
  const m = neuf(); const c = m.client(); const w = travailleurDe(m);
  const avant = (globalThis.__sondes || []).length;
  await c.demander(() => ({ module: "poutre", action: "nouveau" }));
  await c.demander(() => ({ module: "bois", action: "nouveau" }));
  await c.demander(() => ({ module: "poutre", action: "nouveau" }));
  await attendreQue(() => sondesDe(w, "bois").length);
  await tic(300);
  out.par_cle = { sondes: (globalThis.__sondes || []).slice(avant).map((s) => ({ cle: s.cle, premiere: s.premiere, marques: s.marques.map((x) => x[0]), calculer: s.details.calculer_ms !== undefined, import_erreur: s.details.import_erreur || null })),
    derniere: globalThis.__sonde && globalThis.__sonde.cle, types: [...new Set(w.types)].sort() };
  w.terminate();
}
if (parties.has("annonce")) {
  // l'écran qui se monte ANNONCE sa clé (plancher : numpy) avant « pret » : numpy part pendant le démarrage, en
  // parallèle du paquet ; rien n'est importé
  const m = neuf(); const c = m.client(); const w = travailleurDe(m);
  if (typeof m.annoncer === "function") m.annoncer("plancher");   // (absente : le contrôle rougit, le pilote va au bout)
  await m.pret;
  const sm = await m.sondeModules();
  await c.demander(N("plancher"));
  const json = w.dernier;
  await attendreQue(() => sondesDe(w, "plancher").length);
  out.annonce = { sm, json, t: marques(sondesDe(w, "plancher")[0]), postes: w.postes };
  w.terminate();
  // le témoin : sans annonce, numpy ne part qu'à la première requête, après « pret »
  const m2 = neuf(); const c2 = m2.client(); const w2 = travailleurDe(m2);
  await m2.pret;
  await c2.demander(N("plancher"));
  await attendreQue(() => sondesDe(w2, "plancher").length);
  out.annonce.temoin = marques(sondesDe(w2, "plancher")[0]);
  w2.terminate();
}
if (parties.has("fond")) {
  // C4 : moteur prêt, numpy LENT ; le préchauffage d'une clé avec numpy (soudures), puis une vraie requête (poutre)
  // 20 ms après — elle n'attend pas numpy
  const m = neuf(null, { retard: A.retard }); const c = m.client(); const w = travailleurDe(m);
  await m.pret;
  m.prechauffer("soudures"); await tic(20);
  const t = performance.now();
  const r = await c.demander(N("poutre"));
  const attente = performance.now() - t;
  const json = w.dernier;
  const avantPre = !w.recus.some((d) => d.type === "prechauffe");
  const pendant = await m.sondeModules();
  await attendreQue(() => w.recus.some((d) => d.type === "prechauffe"));
  out.fond = { ok: r.ok, json, attente, avantPre, pendant, pre: w.recus.filter((d) => d.type === "prechauffe"), apres: await m.sondeModules() };
  w.terminate();
}
if (parties.has("fond_demarrage")) {
  // C4 : un préchauffage posté PENDANT le démarrage (numpy lent) ne fait pas attendre « pret » ; il se fait ensuite
  const mesurer = async (prech) => {
    const t = performance.now();
    const m = neuf(null, { retard: A.retard }); const w = travailleurDe(m);
    if (prech) m.prechauffer("soudures");
    await m.pret;
    const d = performance.now() - t;
    let pre = null, sm = null;
    if (prech) { await attendreQue(() => w.recus.some((x) => x.type === "prechauffe")); pre = w.recus.filter((x) => x.type === "prechauffe"); sm = await m.sondeModules(); }
    w.terminate();
    return { d, pre, sm };
  };
  const sans = await mesurer(false), avec = await mesurer(true);
  out.fond_demarrage = { sans: sans.d, avec: avec.d, pre: avec.pre, sm: avec.sm };
}
if (parties.has("guet_horsligne")) {
  // hors ligne : numpy REFUSÉ, avec_numpy vide ; garde_corps importe numpy (le guet) — refusé ; puis poutre, dalle
  const m = neuf({ avec_numpy: [] }, { refus: true }); const c = m.client(); const w = travailleurDe(m);
  const r1 = await c.demander(N("garde_corps"));
  const sm = await m.sondeModules();
  const r2 = await c.demander(N("poutre")); const j2 = w.dernier;
  const r3 = await c.demander(N("dalle")); const j3 = w.dernier;
  out.guet_horsligne = { r1: { ok: r1.ok, erreur: r1.erreur }, sm, r2: r2.ok, j2, r3: r3.ok, j3 };
  w.terminate();
  // la voie du préchauffage : garde_corps préchauffé (le guet ; numpy refusé, en fond), puis poutre
  const mp = neuf({ avec_numpy: [] }, { refus: true }); const cp = mp.client(); const wp = travailleurDe(mp);
  await mp.pret;
  mp.prechauffer("garde_corps");
  await attendreQue(() => wp.recus.some((x) => x.type === "prechauffe"), 30000);
  const smp = await mp.sondeModules();
  const rp = await cp.demander(N("poutre"));
  out.guet_horsligne.prechauffage = { pre: wp.recus.filter((x) => x.type === "prechauffe"), sm: smp, ok: rp.ok, json: wp.dernier };
  wp.terminate();
}
if (parties.has("version")) {
  // la version du document (balise stab-version) contre celle du manifeste que le travailleur lit
  const doc = (v) => { globalThis.document = { querySelector: (s) => (s === 'meta[name="stab-version"]' ? { content: v } : null) }; };
  // (a) une autre version : pas de « pret », l'erreur nommée ; la requête la rend sans partir
  doc("autre-version-de-l-essai");
  const ma = neuf(); const ca = ma.client(); const wa = travailleurDe(ma);
  const pa = await pretOuErreur(ma);
  const ra = await borne(ca.demander(N("poutre")), 30000);
  out.version = { a: { pret: pa, url: wa.url, types: [...new Set(wa.types)], etat: ma.etat, perime: ma.perime, r: ra, termine: wa.termine,
    requetes: wa.postes.filter((p) => p === "requete").length } };
  // (b) le témoin : la version du manifeste — « pret », et la réponse
  doc(A.version);
  const mb = neuf(); const cb = mb.client(); const wb = travailleurDe(mb);
  const pb = await pretOuErreur(mb);
  const rb = await borne(cb.demander(N("poutre")));
  out.version.b = { pret: pb, url: wb.url, ok: rb.ok };
  // (c) ce travailleur TOMBE ; le site a été mis à jour entre-temps : le travailleur RECRÉÉ ne sert pas l'autre version
  wb.cale("fatal"); await tic(100);
  const rm = await borne(cb.demander(N("bois")));
  globalThis.__manifeste = { version: "mise-en-ligne-de-l-essai" };
  const rc = await borne(cb.demander(N("poutre")));
  globalThis.__manifeste = null;
  const wc = travailleurs.at(-1);
  out.version.c = { mort: rm.mort === true, r: rc, etat: mb.etat, perime: mb.perime, recree: wc !== wb, url: wc.url, types: [...new Set(wc.types)] };
  // (d) la note PDF : un paquet dont l'empreinte ne correspond plus ; le manifeste relu, de la même version (le témoin :
  // le message d'empreinte), puis d'une autre version (une mise en ligne depuis le démarrage : l'erreur nommée)
  doc(A.version);
  const faux = { "note.sha256": "0".repeat(64) };
  const md = neuf(faux); const cd = md.client(); const wd = travailleurDe(md);
  await pretOuErreur(md);
  const note = () => cd.demander(() => ({ module: "poutre", action: "telecharger", etat: A.etat_poutre, quoi: "note_pdf", infos: A.infos }), { note: true });
  const n1 = await borne(note());
  wd.cale("manifeste", { retouches: { ...faux, version: "mise-en-ligne-de-l-essai" } }); await tic(50);
  const n2 = await borne(note());
  out.version.d = { n1: { ok: n1.ok, erreur: n1.erreur }, n2: { ok: n2.ok, erreur: n2.erreur } };
  wd.terminate();
  delete globalThis.document;
}
if (parties.has("ouverture")) {
  S.marquer("essai_ouverture");
  const avant = performance.getEntriesByType("mark").filter((x) => x.name.startsWith("stab:")).length;
  const sondes = (globalThis.__sondes || []).length;
  S.nouvelleOuverture(3);
  out.ouverture = { avant, apres: performance.getEntriesByType("mark").filter((x) => x.name.startsWith("stab:")).length, sondes_avant: sondes,
    sondes_apres: (globalThis.__sondes || []).length, ouverture: globalThis.__ouverture };
  S.marquer("essai_ouverture");
  out.ouverture.remarque = performance.getEntriesByType("mark").filter((x) => x.name === "stab:essai_ouverture").length;
}
await finir(out, A.sortie);
"""

# P5 : le vrai travailleur qui tombe ; l'ancienne façon de créer un moteur
PILOTE_MORT = r"""
import { preparer, travailleurs, tic, finir } from "./page.mjs";
const A = JSON.parse(process.argv[2]);
preparer(false);
const M = await import(A.moteur);
const out = {};
const N = (k) => () => ({ module: k, action: "nouveau" });
const borne = (p, ms = 120000) => Promise.race([p, new Promise((ok) => setTimeout(() => ok({ __delai: true }), ms))]);
// l'erreur fatale de Pyodide (provoquée par la cale), puis deux requêtes : l'une en vol, l'autre en file
{
  const m = M.moteurPartage(); const c = m.client();
  const existe0 = M.moteurExiste();
  await m.pret;
  const r0 = await c.demander(N("poutre"));
  const w1 = travailleurs.at(-1);
  w1.cale("fatal"); await tic(100);
  const [ra, rb] = await borne(Promise.all([c.demander(N("bois")), c.demander(N("taux"))]), 60000).then((x) => Array.isArray(x) ? x : [x, x]);
  const mortEtat = m.etat, w1termine = w1.termine, existeMort = M.moteurExiste(), nAvant = travailleurs.length;
  const r2 = await borne(c.demander(N("poutre")));
  out.fatal = { existe0, r0: r0.ok, ra, rb, mortEtat, w1termine, existeMort, nAvant, nApres: travailleurs.length, r2ok: r2.ok,
    r2json: travailleurs.at(-1).dernier, meme: M.moteurPartage() === m, existeApres: M.moteurExiste(), etatApres: m.etat };
}
// le fil du travailleur qui s'arrête pendant une requête
{
  const m = new M.Moteur(); const c = m.client();
  await m.pret;
  const w = travailleurs.at(-1);
  const p = c.demander(N("garde_corps"));
  await tic(50); w.cale("sortir");
  const ra = await borne(p, 60000);
  const n = travailleurs.length;
  const r2 = await borne(c.demander(N("bois")));
  out.chute = { ra, etat: m.etat, nAvant: n, nApres: travailleurs.length, r2ok: r2.ok };
}
// l'ancienne façon : new Moteur(surEtape, module), moteur.demander(fabrique) et un objet
{
  const etapes = [];
  const m = new M.Moteur((t) => etapes.push(t), "garde_corps");
  const w = travailleurs.at(-1);
  await m.pret;
  const r = await m.demander(() => ({ module: "garde_corps", action: "nouveau" }));
  const json = w.dernier;
  const b = await m.demander({ action: "consulter", bibliotheque: "beton" });
  out.compat = { etapes, ok: r.ok, json, premier: w.postes[0], biblio: b.ok, url: w.url };
}
await finir(out, A.sortie);
"""

# CLIENT : js/moteur.js contre un faux travailleur (sans Pyodide)
PILOTE_CLIENT = r"""
const A = JSON.parse(process.argv[2]);
globalThis.self = globalThis;
globalThis.location = { href: "http://essai.invalid/poutre" + (A.sonde ? "?sonde=1" : "") };
const FAUX = [];
class Faux {
  constructor(url, opts) { this.url = String(url); this.opts = opts; this.postes = []; this._f = null; this.ecouteurs = []; this.onerror = null; this.termine = false; FAUX.push(this); }
  set onmessage(f) { this._f = f; }
  get onmessage() { return this._f; }
  addEventListener(t, f) { if (t === "message") this.ecouteurs.push(f); }
  postMessage(m) { this.postes.push(m); }
  terminate() { this.termine = true; }
  emettre(d) { if (this._f) this._f({ data: d }); for (const f of this.ecouteurs) f({ data: d }); }
  requetes() { return this.postes.filter((m) => m.id !== undefined); }
  prechauffages() { return this.postes.filter((m) => m.type === "prechauffer").map((m) => m.cle); }
  repondre(rep, i = -1) { const q = this.requetes().at(i); this.emettre({ type: "reponse", id: q.id, json: typeof rep === "string" ? rep : JSON.stringify(rep) }); }
}
globalThis.Worker = Faux;
const R = [];
const chk = (nom, cond, info = "") => R.push([nom, !!cond, cond ? "" : JSON.stringify(info).slice(0, 600)]);
const tic = () => new Promise((ok) => setTimeout(ok, 0));
const borne = (p, ms = 3000) => Promise.race([p, new Promise((ok) => setTimeout(() => ok({ __delai: true }), ms))]);
const calme = async () => { for (let i = 0; i < 4; i++) await tic(); };
const etatDe = (p) => { const s = { v: "attente" }; p.then((x) => { s.v = "resolue"; s.x = x; }, (e) => { s.v = "rejetee"; s.e = e; }); return s; };
let n = 0;
const charger = async () => import(A.moteur + "?essai=" + (++n));
let M = await charger();
const pret = (m) => { FAUX.at(-1).emettre({ type: "pret" }); return m; };

// C1 · la file : une requête à la fois ; fabrique() à son tour, APRÈS que l'écran a appliqué la réponse précédente
{
  const m = pret(new M.Moteur()); const w = FAUX.at(-1); const c = m.client();
  let etat = 0; const vus = [];
  const p1 = (async () => { const r = await (async () => await (async () => await c.demander(() => { vus.push(etat); return { module: "poutre", action: "a", etat }; }))())(); etat = r.etat; })();
  const p2 = c.demander(() => { vus.push(etat); return { module: "poutre", action: "b", etat }; });
  await calme();
  chk("client · une requête à la fois : la seconde attend la réponse de la première", w.requetes().length === 1 && c.enCours === 2, { n: w.requetes().length, enCours: c.enCours });
  w.repondre({ ok: true, etat: 5 });
  await calme(); await p1;
  chk("client · la fabrique de la suivante est appelée à son tour, après que la réponse précédente a été appliquée (même derrière trois « await »)",
    JSON.stringify(vus) === "[0,5]" && JSON.parse(w.requetes()[1].requete).etat === 5, { vus, requete: w.requetes()[1] && w.requetes()[1].requete });
  w.repondre({ ok: true, etat: 6 });
  const r2 = await p2;
  chk("client · la réponse est rendue telle quelle, l'en-cours revient à zéro", r2.etat === 6 && c.enCours === 0, { r2, enCours: c.enCours });
  // une fabrique qui lève : cette demande seule rejette, la suivante part
  const pe = c.demander(() => { throw new Error("fabrique cassée"); });
  const pf = c.demander(() => ({ module: "poutre", action: "c" }));
  let leve = null; try { await pe; } catch (e) { leve = e.message; }
  await calme();
  chk("client · une fabrique qui lève rejette sa demande seule, la file continue", leve === "fabrique cassée" && w.requetes().length === 3, { leve, n: w.requetes().length });
  w.repondre("{pas du json");
  const rf = await pf;
  chk("client · une réponse illisible devient un refus lisible (la file ne se bloque pas)", rf.ok === false && /illisible/.test(rf.erreur), rf);
}
// C2 · fermer() : lectures abandonnées (EcranFerme), modifications gardées, promesse résolue quand tout est fait
{
  const m = pret(new M.Moteur()); const w = FAUX.at(-1); const c = m.client();
  const appels = [];
  const pa = etatDe(c.demander(() => { appels.push("a"); return { module: "poutre", action: "lecture_a" }; }));
  await calme();
  const pb = etatDe(c.demander(() => { appels.push("b"); return { module: "poutre", action: "lecture_b" }; }));
  const pc = etatDe(c.demander(() => { appels.push("c"); return { module: "poutre", action: "modifier_c" }; }, { modifie: true }));
  const pd = etatDe(c.demander(() => { appels.push("d"); return { module: "poutre", action: "lecture_d" }; }));
  const f = etatDe(c.fermer());
  const pe = etatDe(c.demander(() => { appels.push("e"); return {}; }, { modifie: true }));
  await calme();
  chk("client · fermer() est synchrone : le client est fermé, ses lectures pas encore parties rejettent EcranFerme",
    c.ferme === true && pb.v === "rejetee" && pd.v === "rejetee" && pb.e && pb.e.name === "EcranFerme" && pb.e instanceof M.EcranFerme && pb.e instanceof Error,
    { ferme: c.ferme, pb: pb.v, pd: pd.v, nom: pb.e && pb.e.name });
  chk("client · fermé, toute nouvelle demande (même une modification) rejette EcranFerme", pe.v === "rejetee" && pe.e && pe.e.name === "EcranFerme", pe.v);
  chk("client · la requête en vol et la modification en file restent comptées ; fermer() attend", c.enCours === 2 && f.v === "attente" && pa.v === "attente" && pc.v === "attente",
    { enCours: c.enCours, f: f.v, pa: pa.v, pc: pc.v });
  w.repondre({ ok: true, etat: 1 });
  await calme();
  chk("client · la lecture en vol se résout normalement ; la modification gardée part ensuite", pa.v === "resolue" && JSON.parse(w.requetes().at(-1).requete).action === "modifier_c" && f.v === "attente",
    { pa: pa.v, derniere: w.requetes().at(-1).requete, f: f.v });
  w.repondre({ ok: true, etat: 2 });
  await calme();
  chk("client · la modification gardée se résout ; fermer() se résout quand le client n'a plus rien", pc.v === "resolue" && pc.x.etat === 2 && f.v === "resolue" && c.enCours === 0,
    { pc: pc.v, f: f.v, enCours: c.enCours });
  chk("client · les fabriques des lectures abandonnées ne sont jamais appelées", JSON.stringify(appels) === '["a","c"]', appels);
  const f2 = etatDe(m.client().fermer());
  await calme();
  chk("client · fermer() d'un client sans rien en cours se résout aussitôt", f2.v === "resolue", f2.v);
}
// C3 · le préchauffage : jamais posté tant qu'une requête attend ou court ; la plus récente remplace ; une fois par clé
{
  const m = pret(new M.Moteur()); const w = FAUX.at(-1); const c = m.client();
  m.prechauffer("dalle");
  await calme();
  chk("préchauffage · posté aussitôt quand rien n'attend", JSON.stringify(w.prechauffages()) === '["dalle"]', w.prechauffages());
  const p1 = c.demander(() => ({ module: "poutre", action: "a" }));
  await calme();
  m.prechauffer("bois"); m.prechauffer("sol");
  const p2 = c.demander(() => ({ module: "poutre", action: "b" }));
  await calme();
  const pendant1 = w.prechauffages().length;
  w.repondre({ ok: true }); await calme();
  const pendant2 = w.prechauffages().length;
  const ordre = w.postes.map((x) => x.type || x.id);
  w.repondre({ ok: true }); await calme(); await p1; await p2;
  chk("préchauffage · jamais posté tant qu'une requête attend ou court (une vraie requête passe toujours devant)",
    pendant1 === 1 && pendant2 === 1 && typeof ordre.at(-1) === "number", { pendant1, pendant2, ordre });
  chk("préchauffage · la plus récente remplace celle pas encore partie", JSON.stringify(w.prechauffages()) === '["dalle","sol"]', w.prechauffages());
  m.prechauffer("sol"); m.prechauffer("dalle"); m.prechauffer("Pas une clé"); m.prechauffer("../x");
  await calme();
  chk("préchauffage · une clé déjà annoncée n'est pas reposté ; une clé mal formée est ignorée", JSON.stringify(w.prechauffages()) === '["dalle","sol"]', w.prechauffages());
}
// C4 · l'ancienne façon : new Moteur(surEtape, module), demander(fabrique | objet, {note})
{
  const etapes = [];
  const m = new M.Moteur((t) => etapes.push(t), "garde_corps"); const w = FAUX.at(-1);
  await calme();
  const u = new URL(w.url);
  chk(`compatibilité · le travailleur est créé SANS module : ${A.sonde ? "« ?sonde=1 » seul (sous la sonde)" : "adresse sans paramètre"}, en module`,
    u.pathname.endsWith("/js/travailleur.js") && u.search === (A.sonde ? "?sonde=1" : "") && w.opts && w.opts.type === "module", { url: w.url, opts: w.opts });
  chk("compatibilité · le module annoncé part en préchauffage (numpy en parallèle du démarrage s'il s'en sert)",
    w.postes.length === 1 && w.postes[0].type === "prechauffer" && w.postes[0].cle === "garde_corps", w.postes);
  w.emettre({ type: "etape", texte: "Python démarre…" });
  const tard = []; const fin = m.surEtape((t) => tard.push(t));
  w.emettre({ type: "etape", texte: "Chargement…" }); fin();
  w.emettre({ type: "etape", texte: "Premier calcul…" });
  chk("compatibilité · surEtape : le rappel du constructeur reçoit chaque étape ; un abonné tardif reçoit l'étape courante ; le désabonnement tient",
    JSON.stringify(etapes) === '["Python démarre…","Chargement…","Premier calcul…"]' && JSON.stringify(tard) === '["Python démarre…","Chargement…"]' && m.etape === "Premier calcul…",
    { etapes, tard, etape: m.etape });
  chk("compatibilité · état « demarrage » avant « pret »", m.etat === "demarrage", m.etat);
  w.emettre({ type: "pret" }); await m.pret;
  const p = m.demander({ action: "consulter", bibliotheque: "beton" });
  await calme();
  const q1 = w.requetes().at(-1);
  w.repondre({ ok: true, x: 1 }); await p;
  const p2 = m.demander(() => ({ module: "poutre", action: "telecharger", quoi: "note_pdf" }), { note: true });
  await calme();
  const q2 = w.requetes().at(-1);
  w.repondre({ ok: true }); await p2;
  chk("compatibilité · demander(objet) et demander(fabrique, {note}) : {id, requete: chaîne JSON, note}",
    m.etat === "pret" && JSON.parse(q1.requete).bibliotheque === "beton" && q1.note === false && q2.note === true && typeof q2.requete === "string",
    { etat: m.etat, q1, q2 });
}
// C5 · la mort : message « mort » ou erreur du fil après « pret » → {ok:false, mort:true}, arrêt, recréation à la demande
{
  const m = pret(new M.Moteur()); const w = FAUX.at(-1); const c = m.client(); const nF = FAUX.length;
  const pa = etatDe(c.demander(() => ({ module: "poutre", action: "a" })));
  const pb = etatDe(c.demander(() => ({ module: "poutre", action: "b" }), { modifie: true }));
  await calme();
  w.emettre({ type: "mort", texte: "Pyodide fatally failed" });
  await calme();
  chk("mort · toutes les requêtes en attente et en vol rendent {ok:false, mort:true}", pa.v === "resolue" && pb.v === "resolue" && pa.x.mort === true && pb.x.mort === true
    && pa.x.ok === false && /arrêté/.test(pa.x.erreur) && c.enCours === 0, { pa, pb, enCours: c.enCours });
  chk("mort · état « mort », travailleur arrêté (terminate)", m.etat === "mort" && w.termine === true, { etat: m.etat, termine: w.termine });
  const pc = c.demander(() => ({ module: "poutre", action: "c" }));
  await calme();
  const w2 = FAUX.at(-1);
  chk("mort · la demande suivante du client recrée UN travailleur (le même Moteur) et y part", FAUX.length === nF + 1 && w2 !== w && w2.requetes().length === 1 && m.etat === "demarrage" && m.w === w2,
    { n: FAUX.length - nF, etat: m.etat });
  w2.emettre({ type: "pret" }); w2.repondre({ ok: true, v: 3 });
  chk("mort · le travailleur neuf répond", (await borne(pc)).v === 3, "");
  // erreur du fil après « pret » : la même mort
  const pd = etatDe(c.demander(() => ({ module: "poutre", action: "d" })));
  await calme();
  w2.onerror({ message: "le fil est tombé" });
  await calme();
  chk("mort · une erreur du fil APRÈS « pret » est une mort (requête en vol {ok:false, mort:true})", pd.v === "resolue" && pd.x.mort === true && m.etat === "mort" && w2.termine, { pd, etat: m.etat });
  // erreur du fil AVANT « pret » : un démarrage raté, pas une mort ; rien ne reste suspendu
  const m3 = new M.Moteur(); const w3 = FAUX.at(-1); const c3 = m3.client(); const n3 = FAUX.length;
  const pr = etatDe(m3.pret);
  const pe = etatDe(c3.demander(() => ({ module: "poutre", action: "e" })));
  await calme();
  w3.onerror({ message: "script introuvable" });
  await calme();
  const pf = etatDe(c3.demander(() => ({ module: "poutre", action: "f" })));
  await calme();
  chk("démarrage raté (erreur du fil AVANT « pret ») : « erreur », pret rejetée, requêtes rendues {ok:false} sans « mort », aucun travailleur recréé",
    m3.etat === "erreur" && pr.v === "rejetee" && pe.v === "resolue" && pe.x.ok === false && !pe.x.mort && pf.v === "resolue" && pf.x.ok === false && FAUX.length === n3,
    { etat: m3.etat, pr: pr.v, pe, pf, n: FAUX.length - n3 });
  // « erreur » du travailleur (démarrage Python raté) : il répond encore
  const m4 = new M.Moteur(); const w4 = FAUX.at(-1);
  const pr4 = etatDe(m4.pret);
  w4.emettre({ type: "erreur", texte: "paquet abîmé" }); await calme();
  chk("message « erreur » : état « erreur », pret rejetée avec le texte", m4.etat === "erreur" && pr4.v === "rejetee" && pr4.e.message === "paquet abîmé", { etat: m4.etat, pr: pr4.v });
}
// C9 · la version du document dans l'adresse du travailleur (aussi celle d'un travailleur recréé) ; l'annonce
{
  globalThis.document = { querySelector: (s) => (s === 'meta[name="stab-version"]' ? { content: "v1abc" } : null) };
  const m = new M.Moteur(); const w = FAUX.at(-1); const c = m.client();
  w.emettre({ type: "pret" }); w.emettre({ type: "mort", texte: "fatal" });
  c.demander(() => ({ module: "poutre", action: "a" }));
  await calme();
  delete globalThis.document;
  const u = new URL(w.url), u2 = new URL(FAUX.at(-1).url);
  chk(`version · l'adresse du travailleur porte la version du document (« v »), celle d'un travailleur recréé après une mort aussi ; sans module${A.sonde ? " ; « sonde=1 » en plus" : ""}`,
    u.searchParams.get("v") === "v1abc" && u2.searchParams.get("v") === "v1abc" && FAUX.at(-1) !== w && !u.searchParams.has("module")
    && u.searchParams.get("sonde") === (A.sonde ? "1" : null) && [...u.searchParams.keys()].length === (A.sonde ? 2 : 1), { u: w.url, u2: FAUX.at(-1).url });
}
{
  const m = new M.Moteur(); const w = FAUX.at(-1); const c = m.client();
  const annoncer = (k) => { if (typeof m.annoncer === "function") m.annoncer(k); };   // (absente : le contrôle rougit)
  annoncer("plancher");
  const a1 = w.postes.map((x) => x.type || "requete");
  w.emettre({ type: "pret" });
  c.demander(() => ({ module: "poutre", action: "a" })); await calme();
  annoncer("soudures"); annoncer("plancher"); annoncer("Pas une clé"); annoncer("../x");
  const annonces = w.postes.filter((x) => x.type === "annoncer").map((x) => x.cle);
  m.prechauffer("plancher"); w.repondre({ ok: true }); await calme();
  chk("annonce · postée aussitôt (avant « pret », et derrière une requête en vol), une fois par clé, une clé mal formée ignorée ; elle ne vaut pas préchauffage",
    JSON.stringify(a1) === '["annoncer"]' && JSON.stringify(annonces) === '["plancher","soudures"]' && JSON.stringify(w.prechauffages()) === '["plancher"]',
    { a1, annonces, pre: w.prechauffages() });
}
// C10 · un démarrage RATÉ du moteur partagé : relancé par moteurPartage() (l'écran suivant), jamais par une requête ;
// un démarrage PÉRIMÉ (le site a été mis à jour) ne l'est pas
{
  M = await charger();
  const m = M.moteurPartage(); const w = FAUX.at(-1); const c = m.client(); const n0 = FAUX.length;
  w.emettre({ type: "erreur", texte: "réseau coupé" }); await calme();
  const pe = etatDe(c.demander(() => ({ module: "poutre", action: "e" }))); await calme();
  const nReq = FAUX.length, existe = M.moteurExiste();
  const m2 = M.moteurPartage(); const w2 = FAUX.at(-1);
  const pr = etatDe(m.pret);
  await calme();
  w2.emettre({ type: "pret" }); await calme();
  chk("partagé · démarrage raté (message « erreur ») : une requête ne relance rien ; moteurPartage() (l'écran suivant) arrête l'ancien travailleur, en démarre un neuf (le même objet), et son « pret » se résout",
    nReq === n0 && !existe && m2 === m && w2 !== w && w.termine && FAUX.length === n0 + 1 && pe.v === "resolue" && pe.x.ok === false && pr.v === "resolue" && m.etat === "pret" && M.moteurExiste(),
    { nReq: nReq - n0, n: FAUX.length - n0, termine: w.termine, pe, pr: pr.v, etat: m.etat });
}
{
  M = await charger();
  const m = M.moteurPartage(); const w = FAUX.at(-1); const c = m.client(); const n0 = FAUX.length;
  w.onerror({ message: "script introuvable" }); await calme();
  const r1 = await borne(c.demander(() => ({ module: "poutre", action: "f" })));
  const nReq = FAUX.length, existe = M.moteurExiste();
  const m2 = M.moteurPartage(); const w2 = FAUX.at(-1);
  w2.emettre({ type: "pret" }); await calme();
  chk("partagé · démarrage raté (fil arrêté avant « pret ») : la requête rend {ok:false} sans relance ; moteurPartage() en démarre un neuf (le même objet), prêt ensuite",
    r1.ok === false && nReq === n0 && !existe && m2 === m && w2 !== w && FAUX.length === n0 + 1 && m.etat === "pret" && M.moteurExiste(),
    { r1, n: FAUX.length - n0, etat: m.etat });
}
{
  M = await charger();
  const m = M.moteurPartage(); const w = FAUX.at(-1); const c = m.client(); const n0 = FAUX.length;
  const pr = etatDe(m.pret);
  w.emettre({ type: "erreur", texte: "Le site a été mis à jour : rechargez la page", perime: true }); await calme();
  const r = await borne(c.demander(() => ({ module: "poutre", action: "g" })));
  const m2 = M.moteurPartage();
  chk("partagé · démarrage PÉRIMÉ (le site a été mis à jour) : travailleur arrêté, pret rejetée avec le texte, la requête rend l'erreur nommée sans partir, moteurPartage() ne relance pas",
    m.etat === "erreur" && m.perime && w.termine && pr.v === "rejetee" && /mis à jour/.test(pr.e.message) && r.ok === false && /mis à jour/.test(r.erreur)
    && !w.requetes().length && m2 === m && FAUX.length === n0 && !M.moteurExiste(), { etat: m.etat, perime: m.perime, r, n: FAUX.length - n0 });
}
// C6 · moteurPartage / moteurExiste
{
  M = await charger();
  const avant = M.moteurExiste();
  const m = M.moteurPartage(); const w = FAUX.at(-1);
  chk("partagé · moteurExiste() faux avant la première demande, vrai après ; moteurPartage() rend toujours le même", !avant && M.moteurExiste() && M.moteurPartage() === m && FAUX.at(-1) === w,
    { avant, apres: M.moteurExiste() });
  w.emettre({ type: "pret" }); w.emettre({ type: "mort", texte: "fatal" });
  const mort = M.moteurExiste();
  const m2 = M.moteurPartage();
  chk("partagé · mort : moteurExiste() faux ; moteurPartage() redémarre un travailleur neuf (même objet)", !mort && m2 === m && FAUX.at(-1) !== w && m.etat === "demarrage" && M.moteurExiste(),
    { mort, etat: m.etat });
}
// C7 · le setter onmessage est gardé : le harnais du site l'enveloppe pour retoucher une réponse
{
  class Espion extends Faux {
    set onmessage(f) { this.__f = f; super.onmessage = (ev) => f(ev.data && ev.data.type === "reponse" ? { data: { ...ev.data, json: ev.data.json.replace('"v": 1', '"v": 2') } } : ev); }
    get onmessage() { return this.__f; }
  }
  globalThis.Worker = Espion;
  const m = new M.Moteur(); const w = FAUX.at(-1); w.emettre({ type: "pret" });
  const p = m.demander(() => ({ module: "poutre", action: "a" }));
  await calme(); w.repondre('{"ok": true, "v": 1}');
  const r = await borne(p);
  globalThis.Worker = Faux;
  chk("setter · js/moteur.js pose « onmessage » par le setter (l'enveloppe du harnais retouche la réponse)", r.v === 2, r);
}
// C8 · la sonde : chaque message « sonde » dans globalThis.__sondes (historique), le dernier dans __sonde ; par ouverture
{
  const m = pret(new M.Moteur()); const w = FAUX.at(-1);
  delete globalThis.__sondes; delete globalThis.__sonde;
  w.emettre({ type: "sonde", cle: "poutre", premiere: true, marques: [], details: {} });
  w.emettre({ type: "sonde", cle: "bois", premiere: false, marques: [], details: {} });
  w.emettre({ type: "prechauffe", cle: "dalle", ms: 3 });
  if (A.sonde) {
    chk("sonde · chaque message « sonde » s'ajoute à globalThis.__sondes ; __sonde est le dernier ; « prechauffe » dans __prechauffes",
      (globalThis.__sondes || []).length === 2 && globalThis.__sonde.cle === "bois" && (globalThis.__prechauffes || []).length === 1, { n: (globalThis.__sondes || []).length });
    const S = await import(A.sondeJs);
    S.marquer("essai_a"); S.marquerUneFois("essai_b");
    const avant = performance.getEntriesByType("mark").filter((x) => x.name.startsWith("stab:")).length;
    S.nouvelleOuverture(7);
    const apres = performance.getEntriesByType("mark").filter((x) => x.name.startsWith("stab:")).length;
    S.marquerUneFois("essai_b");
    chk("sonde par ouverture · nouvelleOuverture(n) efface les marques « stab: » de la page (marquerUneFois revaut pour l'ouverture) et garde __sondes",
      avant >= 2 && apres === 0 && performance.getEntriesByType("mark").filter((x) => x.name === "stab:essai_b").length === 1 && globalThis.__sondes.length === 2 && globalThis.__ouverture === 7,
      { avant, apres, sondes: globalThis.__sondes.length, ouverture: globalThis.__ouverture });
  } else {
    chk("sonde · sans « ?sonde=1 », les messages de la sonde sont ignorés (rien dans globalThis)", globalThis.__sondes === undefined && globalThis.__sonde === undefined && globalThis.__prechauffes === undefined,
      { sondes: globalThis.__sondes });
    chk("sonde · sans « ?sonde=1 », aucune marque « stab: » sur la page", !performance.getEntriesByType("mark").some((x) => x.name.startsWith("stab:")), "");
  }
}
(await import("node:fs")).writeFileSync(A.sortie, JSON.stringify(R));
process.exit(0);
"""


def ecrire_node(dossier):
    """Les fichiers node de l'essai, dans ``dossier`` (jamais dans le site)."""
    pyodide = json.dumps("file://" + os.path.join(VENDOR, "pyodide.mjs"))
    for nom, texte in (("crochets.mjs", CROCHETS), ("pyo.mjs", PYO % {"pyodide": pyodide}), ("cale.mjs", CALE), ("page.mjs", PAGE),
                       ("p1.mjs", PILOTE_P1 % {"pyodide": pyodide}), ("partage.mjs", PILOTE_PARTAGE), ("sonde.mjs", PILOTE_SONDE),
                       ("mort.mjs", PILOTE_MORT), ("client.mjs", PILOTE_CLIENT)):
        with open(os.path.join(dossier, nom), "w", encoding="utf-8") as f:
            f.write(texte)


def nom_sortie(dossier, script):
    """Le fichier de sortie d'UN lancement de pilote : unique, même pour deux lancements du même pilote dans la même
    milliseconde (les deux pilotes « client », avec et sans la sonde, partent ensemble : nommés au processus et à la
    milliseconde, ils pouvaient s'écraser — un résultat perdu, l'autre attribué au mauvais pilote)."""
    return os.path.join(dossier, f"{script}.{os.getpid()}.{uuid.uuid4().hex}.json")


def node(dossier, script, args, timeout=900):
    """Lance un pilote node ; rend (sortie JSON lue dans le fichier « sortie » ou None, durée, trace d'erreur)."""
    t0 = time.time()
    sortie = nom_sortie(dossier, script)
    args = dict(args, sortie=sortie)
    try:
        p = subprocess.run(["node", os.path.join(dossier, script), json.dumps(args)], capture_output=True, text=True, timeout=timeout, env=ENV, cwd=dossier)
        err = (p.stderr or "")[-1500:] + (p.stdout or "")[-500:]
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, f"{script} : délai dépassé ({timeout} s)"
    try:
        with open(sortie, encoding="utf-8") as f:
            out = json.load(f)
        os.remove(sortie)
    except (OSError, ValueError):
        out = None
    return out, time.time() - t0, err


# ====================================================================== CPython : des processus NEUFS
_NOUVEAU = "import sys, json; sys.path.insert(0, %r); from api_web import routeur; sys.stdout.write(routeur.traiter(json.dumps({'module': sys.argv[1], 'action': 'nouveau'})))" % RACINE
_NOTE = """import sys, json; sys.path.insert(0, %r); from api_web import routeur
k = sys.argv[1]
n = json.loads(routeur.traiter(json.dumps({"module": k, "action": "nouveau"})))
sys.stdout.write(routeur.traiter(json.dumps({"module": k, "action": "telecharger", "etat": n["etat"], "quoi": "note_pdf", "infos": json.loads(sys.argv[2])})))""" % RACINE
_IMPORT = """import sys, json, importlib; sys.path.insert(0, %r)
from api_web import routeur, registre
cle = sys.argv[1]
if cle.startswith("biblio_"):
    importlib.import_module(routeur.BIBLIOTHEQUES[cle[len("biblio_"):]])
else:
    importlib.import_module("api_web." + cle)
print(json.dumps({"modules": sorted(m for m in sys.modules if m.startswith("api_web.")), "numpy": "numpy" in sys.modules}))""" % RACINE


_CONSULTER = "import sys; sys.path.insert(0, %r); from api_web import routeur; sys.stdout.write(routeur.traiter(sys.argv[1]))" % RACINE


def cpython(code, *args):
    p = subprocess.run([PY, "-c", code, *args], capture_output=True, text=True, cwd=RACINE, env=ENV, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])
    return p.stdout


def references(pool):
    """Les réponses de CPython, un processus neuf par clé : « nouveau » (17 établis), note PDF (NOTE_PDF), import ; et
    un processus neuf par consultation de BIBLIO_REQUETES."""
    f_nouveau = {k: pool.submit(cpython, _NOUVEAU, k) for k in ETABLIS}
    f_note = {k: pool.submit(cpython, _NOTE, k, json.dumps(INFOS_NOTE)) for k in SW.NOTE_PDF}
    f_import = {k: pool.submit(cpython, _IMPORT, k) for k in CLES_PRECHAUFFAGE}
    f_biblio = [pool.submit(cpython, _CONSULTER, json.dumps(q)) for q in BIBLIO_REQUETES]
    return {"nouveau": {k: f.result() for k, f in f_nouveau.items()}, "note": {k: f.result() for k, f in f_note.items()},
            "import": {k: json.loads(f.result()) for k, f in f_import.items()}, "biblio": [f.result() for f in f_biblio]}


# ====================================================================== les comparaisons (et leurs exceptions nommées)
_NOMBRE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _svg_ulp(a, b):
    """Deux SVG égaux au texte près des nombres, ceux-ci à TOLERANCE_ULP relatif ; rend (égaux, nombres différents)."""
    ta, tb = _NOMBRE.split(a), _NOMBRE.split(b)
    na, nb = _NOMBRE.findall(a), _NOMBRE.findall(b)
    if ta != tb or len(na) != len(nb):
        return False, 0
    diff = 0
    for x, y in zip(na, nb):
        if x != y:
            fx, fy = float(x), float(y)
            if abs(fx - fy) > TOLERANCE_ULP * max(abs(fx), abs(fy), 1e-300):
                return False, diff
            diff += 1
    return True, diff


def comparer(cle, cp, py, passe):
    """(égal, exception employée ou None, détail) : l'octet, sinon l'exception NOMMÉE de cette clé."""
    if py is None:
        return False, None, "aucune réponse"
    if cp == py:
        return True, None, ""
    if cle == "soudures":
        if SOUDURES_ID.sub(r"\1sN_", cp) == SOUDURES_ID.sub(r"\1sN_", py):
            return True, "compteur s<n>_", ""
        return False, None, "soudures diffère au-delà du compteur s<n>_"
    if cle in VUES_ULP:
        a, b = json.loads(cp), json.loads(py)
        autres = [k for k in set(a) | set(b) if k != "vues" and a.get(k) != b.get(k)]
        if autres or len(a.get("vues", [])) != len(b.get("vues", [])):
            return False, None, f"diffère hors des vues : {autres}"
        touchees = []
        for va, vb in zip(a["vues"], b["vues"]):
            if va == vb:
                continue
            c = va.get("cle")
            sans = {k: v for k, v in va.items() if k != "svg"} == {k: v for k, v in vb.items() if k != "svg"}
            egal, n = _svg_ulp(va.get("svg", ""), vb.get("svg", ""))
            if c not in VUES_ULP[cle] or not sans or not egal:
                return False, None, f"vue « {c} » : {'hors de la liste nommée' if c not in VUES_ULP[cle] else 'écart au-delà de 1e-9 relatif'}"
            touchees.append(c)
        if tuple(sorted(touchees)) != tuple(sorted(VUES_ULP[cle])):
            return False, None, f"les vues qui diffèrent ({touchees}) ne sont pas exactement les vues nommées {VUES_ULP[cle]} : la liste rétrécit"
        return True, "ulp", ""
    a = json.loads(cp) if cp.startswith("{") else {}
    b = json.loads(py) if py.startswith("{") else {}
    return False, None, "clés qui diffèrent : " + ", ".join(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))[:300]


def biblio_nom(q):
    """« beton/ancrages {phi 25, classe C40/50} (note) » : le nom court d'une consultation."""
    p = ", ".join(f"{k} {v}" for k, v in q.get("params", {}).items())
    return f"{q['bibliotheque']}/{q['rubrique']}" + (f" {{{p}}}" if p else "") + (f" ({q['hote']})" if q.get("hote") else "")


def diff_json(a, b):
    """Les clés de premier niveau qui diffèrent entre deux réponses JSON (texte court)."""
    try:
        x, y = json.loads(a), json.loads(b)
    except (TypeError, ValueError):
        return "réponse illisible"
    return "clés qui diffèrent : " + ", ".join(sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k)))[:300]


def pdf_normalise(rep_json):
    r = json.loads(rep_json)
    if not r.get("ok"):
        return None, r.get("erreur")
    b = base64.b64decode(r["fichier"]["base64"])
    for motif, rempl in NORMALISATIONS_PDF:
        b = re.sub(motif, rempl, b)
    return (r["fichier"]["nom"], r["fichier"]["type"], b), None


# ====================================================================== P0 · le paquet à jour
def p0(B, site):
    zip_ = os.path.join(site, "moteur", "moteur.zip")
    man = json.load(open(os.path.join(site, "moteur", "manifeste.json"), encoding="utf-8"))
    with zipfile.ZipFile(zip_) as z:
        noms = z.namelist()
        differents = [n for n in noms if not os.path.isfile(os.path.join(RACINE, n)) or open(os.path.join(RACINE, n), "rb").read() != z.read(n)]
    manquants = []
    for paquet in SW.PAQUETS_ENTIERS:
        for d, _, fs in os.walk(os.path.join(RACINE, paquet)):
            if "__pycache__" in d:
                continue
            manquants += [os.path.relpath(os.path.join(d, f), RACINE) for f in fs if f.endswith(".py") and os.path.relpath(os.path.join(d, f), RACINE) not in noms]
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    ok = (not differents and not manquants and man["sha256"] == sha(zip_) and man["fichiers"] == noms
          and man["note"]["sha256"] == sha(os.path.join(site, "moteur", "note.zip")))
    return B.chk("P0 · le paquet expédié est à jour (chaque fichier = dépôt, paquets entiers complets, empreintes du manifeste) — sinon : "
                 "reconstruire (/usr/bin/python3.12 site/web/construire.py)", ok,
                 {"differents": differents[:5], "manquants": manquants[:5], "sha": man["sha256"] == sha(zip_)})


# ====================================================================== P1 · la référence dans un interpréteur
_RESULTAT = re.compile(r"RÉSULTAT : (\d+) OK, (\d+) échec")


def p1_lancer(dossier, site, ordres, mutation=""):
    """Joue P1 ; rend (résultats par ordre, durée, fin de stderr). Avec ``mutation`` : rien n'est jugé (résultats vides)
    si le pilote n'a pas dit « @@MUTEE » — le texte à muter manquait, ou l'interpréteur n'a pas tourné."""
    t0 = time.time()
    args = {"vendor": VENDOR, "zip": os.path.join(site, "moteur", "moteur.zip"), "tests": os.path.join(RACINE, "tests"), "ordres": ordres, "mutation": mutation}
    p = subprocess.run(["node", os.path.join(dossier, "p1.mjs"), json.dumps(args)], capture_output=True, text=True, timeout=1800, env=ENV, cwd=dossier)
    if mutation and "@@MUTEE" not in p.stdout.split("@@ORDRE ")[0]:
        return [], time.time() - t0, "rien n'a été muté (texte absent, ou interpréteur arrêté) : " + (p.stdout[-300:] + p.stderr[-300:])
    blocs = p.stdout.split("@@ORDRE ")[1:]
    res = []
    for b in blocs:
        m = _RESULTAT.search(b)
        kos = [l for l in b.splitlines() if l.startswith("KO ")]
        res.append((int(m.group(1)), int(m.group(2)), kos[:5]) if m else (0, -1, [b[-400:] or p.stderr[-400:]]))
    return res, time.time() - t0, p.stderr[-600:]


def p1_cpython():
    t0 = time.time()
    p = subprocess.run([PY, os.path.join(RACINE, "tests", "test_reference_refonte.py")], capture_output=True, text=True, cwd=RACINE, env=ENV, timeout=1800)
    m = _RESULTAT.search(p.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else (0, -1), time.time() - t0


def p1_juger(B, res, cp):
    B.chk(f"P1 · CPython rejoue la référence sur ses photos : {cp[0]} OK, {cp[1]} échec(s) (la référence est à jour)", cp[1] == 0 and cp[0] > 0, cp)
    if len(res) != 2:
        B.chk("P1 · les deux ordres ont tourné dans l'interpréteur Pyodide", False, res)
        return
    (n1, k1, e1), (n2, k2, e2) = res
    B.chk(f"P1 · Pyodide, UN interpréteur, ordre du registre : {n1} OK, {k1} échec(s) — autant de contrôles que CPython", k1 == 0 and n1 == cp[0], e1)
    B.chk(f"P1 · le MÊME interpréteur, ordre inverse ensuite (tous les modules déjà chargés) : {n2} OK, {k2} échec(s)", k2 == 0 and n2 == n1 - 1, e2)


# ====================================================================== P2 · P3 · le moteur partagé
def p2_juger(B, out, refs):
    if not out:
        B.chk("P2 · le pilote du moteur partagé a rendu ses réponses", False, "aucune sortie")
        return
    B.chk("P2 · le travailleur est créé SANS module (adresse sans paramètre) ; un seul pour toute la séquence",
          out["url"].endswith("/js/travailleur.js") and out["travailleurs"] == 1, {"url": out["url"], "n": out["travailleurs"]})
    B.chk(f"P2 · les {len(ORDRE_P2) - 1} préchauffages entrelacés sont partis (contrôle positif)", out["postes"] >= len(ORDRE_P2) - 1, out["postes"])
    employees = {}
    for passe, nom in ((1, "passe1"), (2, "passe2")):
        for k in ORDRE_P2:
            egal, exc, det = comparer(k, refs["nouveau"][k], out[nom].get(k), passe)
            if exc:
                employees[(k, passe)] = exc
            B.chk(f"P2 · {k} · {'1ʳᵉ passe (préchauffé, ordre entrelacé)' if passe == 1 else '2ᵉ passe (ordre inverse, après tous)'} : "
                  f"« nouveau » = CPython processus neuf{' (exception nommée : ' + exc + ')' if exc else ', à l’octet'}", egal, det)
    B.chk("P2 · les exceptions employées sont EXACTEMENT celles nommées (la liste rougit si elle grandit ou rétrécit)",
          employees == EXCEPTIONS_P2, {"employees": sorted(map(str, employees)), "nommees": sorted(map(str, EXCEPTIONS_P2))})
    B.chk("P2 · sans « ?sonde=1 », le travailleur n'émet que etape, pret, reponse", out["types"] == ["etape", "pret", "reponse"], out["types"])
    # les bibliothèques, dans le MÊME interpréteur, après les 34 « nouveau » : ordre direct puis inverse, à l'octet
    for i, q in enumerate(BIBLIO_REQUETES):
        cp = refs["biblio"][i]
        for passe, nom in ((1, "biblio1"), (2, "biblio2")):
            py = (out.get(nom) or [None] * len(BIBLIO_REQUETES))[i]
            ok = py is not None and py == cp and json.loads(cp).get("ok") is True
            B.chk(f"P2 · biblio · {biblio_nom(q)} · {'après les établis, ordre direct' if passe == 1 else 'ordre inverse, ensuite'} : "
                  "« consulter » = CPython processus neuf, à l'octet", ok, "aucune réponse" if py is None else diff_json(cp, py))
    for k in SW.NOTE_PDF:
        a, ea = pdf_normalise(refs["note"][k])
        b, eb = pdf_normalise(out["notes"][k]) if out["notes"].get(k) else (None, "aucune réponse")
        B.chk(f"P3 · {k} · la note PDF, après tous les autres modules, = CPython (normalisés : /CreationDate, /ModDate, /ID)",
              a is not None and a == b, ea or eb or (a and b and (a[0], len(a[2])) != (b[0], len(b[2])) and "nom ou taille diffèrent") or "octets différents")
    print(f"     (P2 : prêt en {out['ms']['pret']} ms, 34 « nouveau » en {out['ms']['nouveaux']} ms ; P3 : notes à {out['ms']['notes']} ms)")


# ====================================================================== P3b · P4 · sous la sonde
def p4_juger(B, out, refs, cles):
    if not out:
        B.chk("P4 · le pilote de la sonde a rendu ses mesures", False, "aucune sortie")
        return
    man = json.load(open(os.path.join(WEB, "moteur", "manifeste.json"), encoding="utf-8"))
    releves = set(SW.ADRESSES)
    cles_mods = {"api_web." + k for k in REG.cles()} | set(ROUTEUR.BIBLIOTHEQUES.values())
    if "isolement" in out and out["isolement"]:
        B.chk("P4 · sous la sonde, le travailleur est créé avec « ?sonde=1 » seul", (out.get("url") or "").endswith("/js/travailleur.js?sonde=1"), out.get("url"))
        for k in cles:
            lu = out["isolement"].get(k) or {}
            vu = sorted(set(lu.get("modules") or []) & cles_mods)
            attendu = sorted(set(refs["import"][k]["modules"]) & cles_mods)
            cible = ("api_web." + k) if not k.startswith("biblio_") else ROUTEUR.BIBLIOTHEQUES[k[len("biblio_"):]]
            pre = lu.get("prechauffe") or [{}]
            B.chk(f"P4 · {k} · préchauffé seul : les modules d'écran importés = ceux de CPython pour « import {cible} » ({', '.join(m[8:] for m in attendu)})",
                  vu == attendu and cible in vu and pre[0].get("saute") is False, {"vu": vu, "attendu": attendu, "prechauffe": pre})
            if k in releves:
                B.chk(f"P4 · {k} · numpy chargé ⇔ la clé est dans avec_numpy ({'oui' if k in man['avec_numpy'] else 'non'})",
                      lu.get("charge") is (k in man["avec_numpy"]) and lu.get("guet") is False, {"charge": lu.get("charge"), "guet": lu.get("guet")})
            else:
                attendu_np = k in man["avec_numpy"] or refs["import"][k]["numpy"]
                B.chk(f"P4 · {k} · clé hors du relevé : numpy chargé ⇔ {'avec_numpy' if k in man['avec_numpy'] else 'son import le demande (le guet)'} ({'oui' if attendu_np else 'non'})",
                      lu.get("charge") is attendu_np and lu.get("guet") is False, {"charge": lu.get("charge"), "guet": lu.get("guet")})
    if "guet_requete" in out:
        g = out["guet_requete"]
        egal, exc, det = comparer("garde_corps", refs["nouveau"]["garde_corps"], g["json"], 1)
        B.chk("P4 · guet · avec_numpy vide : « nouveau » de garde_corps = CPython (le guet charge numpy et REJOUE la requête)",
              egal and any(d.get("rejouee") for d in g["sonde_gc"]), {"det": det, "sonde": g["sonde_gc"]})
        B.chk("P4 · guet · après le rejeu, guet.demande est remis à faux ; la requête suivante (poutre) n'est pas rejouée",
              g["sm1"].get("guet") is False and g["sm1"].get("charge") is True and g["sonde_poutre"] and not any(d.get("rejouee") for d in g["sonde_poutre"]),
              {"sm1": {k: g["sm1"].get(k) for k in ("guet", "charge", "numpy")}, "poutre": g["sonde_poutre"]})
    if "guet_prechauffer" in out:
        g = out["guet_prechauffer"]
        B.chk("P4 · guet · avec_numpy vide : préchauffer garde_corps passe par le guet (numpy chargé, import refait, demande remise à faux)",
              "api_web.garde_corps" in (g["sm"].get("modules") or []) and g["sm"].get("charge") is True and g["sm"].get("guet") is False,
              {k: g["sm"].get(k) for k in ("charge", "guet")})
    if "note_numpy" in out:
        g = out["note_numpy"]
        a, ea = pdf_normalise(refs["note"]["garde_corps"])
        b, eb = pdf_normalise(g["json"]) if g["json"] else (None, "aucune réponse")
        d = (g["sonde"] or [{}])[0].get("details", {})
        B.chk("P3b · première requête = la note de garde_corps, avec_numpy vide : numpy chargé AVANT la note (note.avec_numpy, clé lue dans la requête), sans rejeu ; PDF = CPython",
              a is not None and a == b and len(g["sonde"]) == 1 and not d.get("rejouee") and g["sm"].get("charge") is True,
              {"rejouee": d.get("rejouee"), "charge": g["sm"].get("charge"), "erreur": ea or eb})
    if "saut" in out:
        s = out["saut"]
        sautes = {p["cle"]: p["saute"] for p in s["prechauffe"]}
        mods = set(s["sm"].get("modules") or []) if s.get("sm") else set()
        B.chk("P4 · un préchauffage n'est jamais traité devant une vraie requête arrivée derrière lui (dalle : un plus récent ; bois : la requête) : sautés, rien importé",
              sautes == {"dalle": True, "bois": True} and "api_web.dalle" not in mods and "api_web.bois" not in mods and "api_web.poutre" in mods and s["reponse"],
              {"sautes": sautes, "mods": sorted(m for m in mods if m[8:] in ("dalle", "bois", "poutre"))})
    if "chaine" in out:
        c = out["chaine"]
        B.chk("P4 · une seule chaîne : une note (qui attend son paquet) puis un calcul postés d'un coup répondent dans l'ordre", c["ordre"] == [1, 2] and c["ok"] == [True, True], c)
    if "par_cle" in out:
        s = out["par_cle"]["sondes"]
        cles_ = [x["cle"] for x in s]
        demandees = {"requete1_debut", "requete1_traitee", "reponse1_postee", "module_import_debut", "module_import_fin"}
        B.chk("P4 · la sonde date la première requête de CHAQUE clé (poutre, bois ; pas la 2ᵉ de poutre), les « calculer » de mesure à la toute première seulement",
              cles_ == ["poutre", "bois"] and [x["premiere"] for x in s] == [True, False] and [x["calculer"] for x in s] == [True, False]
              and all(demandees <= set(x["marques"]) for x in s) and not any(x["import_erreur"] for x in s) and out["par_cle"]["derniere"] == "bois", s)
        B.chk("P4 · sous la sonde, les messages en plus : sonde (et prechauffe, sonde_modules à la demande)",
              set(out["par_cle"]["types"]) >= {"etape", "pret", "reponse", "sonde"}, out["par_cle"]["types"])
    cles_registre = {"api_web." + k for k in REG.cles()} | set(ROUTEUR.BIBLIOTHEQUES.values())
    if "annonce" in out:
        a = out["annonce"]
        t, tm = a["t"], a["temoin"]
        importes = sorted(set(a["sm"].get("modules") or []) & cles_registre)
        B.chk("P4 · annonce · plancher annoncé avant « pret » : numpy part PENDANT le démarrage, en parallèle du paquet (numpy_debut < deballage_fin), "
              "« pret » l'attend, et rien n'est importé (aucun api_web.<clé>) — comme l'ancien « ?module= »",
              t.get("numpy_debut") is not None and t.get("deballage_fin") is not None and t["numpy_debut"] < t["deballage_fin"]
              and t.get("numpy_fin") is not None and t.get("pret") is not None and t["numpy_fin"] <= t["pret"] + 1
              and a["sm"].get("charge") is True and not importes and a["postes"][:1] == ["annoncer"],
              {"numpy_debut": t.get("numpy_debut"), "deballage_fin": t.get("deballage_fin"), "numpy_fin": t.get("numpy_fin"), "pret": t.get("pret"),
               "charge": a["sm"].get("charge"), "importes": importes, "postes": a["postes"][:3]})
        egal, _, det = comparer("plancher", refs["nouveau"]["plancher"], a["json"], 1)
        B.chk("P4 · annonce · la première réponse de plancher = CPython (processus neuf), à l'octet", egal, det)
        B.chk("P4 · annonce · le témoin : sans annonce, numpy ne part qu'à la première requête, APRÈS « pret » (le contrôle voit la différence)",
              tm.get("numpy_debut") is not None and tm.get("pret") is not None and tm["numpy_debut"] > tm["pret"],
              {"numpy_debut": tm.get("numpy_debut"), "pret": tm.get("pret")})
    if "fond" in out:
        f = out["fond"]
        egal, _, det = comparer("poutre", refs["nouveau"]["poutre"], f["json"], 1)
        B.chk(f"P4 · fond · un préchauffage de soudures (numpy, retardé de {RETARD_NUMPY} ms) ne retient pas une vraie requête de poutre arrivée 20 ms "
              f"après : réponse en {f['attente']:.0f} ms (< {RETARD_NUMPY // 2}), avant la fin du préchauffage, = CPython",
              f["ok"] and f["attente"] < RETARD_NUMPY / 2 and f["avantPre"] and egal, {"attente": f["attente"], "avantPre": f["avantPre"], "det": det})
        pre = f["pre"] or [{}]
        B.chk("P4 · fond · pendant que numpy se charge en fond (hors de la chaîne), le guet refuse tout import de numpy (en_cours) ; après : chargé, "
              "et soudures SAUTÉ à la reprise (une vraie requête est arrivée derrière lui) — rien d'importé",
              f["pendant"].get("en_cours") is True and f["pendant"].get("charge") is False and len(f["pre"]) == 1 and pre[0].get("saute") is True
              and f["apres"].get("charge") is True and f["apres"].get("en_cours") is False and "api_web.soudures" not in (f["apres"].get("modules") or []),
              {"pendant": {k: f["pendant"].get(k) for k in ("en_cours", "charge")}, "pre": f["pre"], "apres": {k: f["apres"].get(k) for k in ("en_cours", "charge")}})
    if "fond_demarrage" in out:
        f = out["fond_demarrage"]
        pre = f["pre"] or [{}]
        B.chk(f"P4 · fond · un préchauffage de soudures posté PENDANT le démarrage (numpy retardé de {RETARD_NUMPY} ms) ne fait pas attendre « pret » : "
              f"{f['avec']:.0f} ms contre {f['sans']:.0f} ms sans lui (écart < {RETARD_NUMPY // 2})",
              f["avec"] - f["sans"] < RETARD_NUMPY / 2, {"avec": f["avec"], "sans": f["sans"]})
        B.chk("P4 · fond · ce préchauffage se fait ensuite (numpy en fond, puis l'import repasse par la chaîne) : soudures importé, numpy chargé",
              len(f["pre"] or []) == 1 and pre[0].get("saute") is False and "api_web.soudures" in ((f["sm"] or {}).get("modules") or [])
              and (f["sm"] or {}).get("charge") is True, {"pre": f["pre"], "charge": (f["sm"] or {}).get("charge")})
    if "guet_horsligne" in out:
        g = out["guet_horsligne"]
        eg2, _, d2 = comparer("poutre", refs["nouveau"]["poutre"], g["j2"], 1)
        eg3, _, d3 = comparer("dalle", refs["nouveau"]["dalle"], g["j3"], 1)
        B.chk("P4 · guet hors ligne · requête : garde_corps (le guet, numpy refusé) garde son refus ; guet.demande est remis à faux ; "
              "poutre et dalle, ensuite, répondent = CPython",
              g["r1"]["ok"] is False and "numpy" in str(g["r1"].get("erreur")) and g["sm"].get("guet") is False and g["r2"] and eg2 and g["r3"] and eg3,
              {"r1": g["r1"], "guet": g["sm"].get("guet"), "r2": g["r2"], "r3": g["r3"], "d2": d2, "d3": d3})
        p = g["prechauffage"]
        egp, _, dp = comparer("poutre", refs["nouveau"]["poutre"], p["json"], 1)
        B.chk("P4 · guet hors ligne · préchauffage : garde_corps préchauffé (le guet, numpy refusé en fond) n'importe rien et le dit ; guet.demande à "
              "faux ; poutre, ensuite, répond = CPython",
              len(p["pre"]) == 1 and p["pre"][0].get("erreur") and p["sm"].get("guet") is False and p["ok"] and egp,
              {"pre": p["pre"], "guet": p["sm"].get("guet"), "ok": p["ok"], "dp": dp})
    if "version" in out:
        v = out["version"]
        a, b_, c, d = v["a"], v["b"], v["c"], v["d"]
        B.chk("P6 · version · le document d'une AUTRE version que le manifeste : pas de « pret » mais l'erreur nommée « Le site a été mis à jour : "
              "rechargez la page » (périmé), travailleur arrêté ; la requête rend cette erreur sans partir",
              a["pret"] == "erreur : " + MIS_A_JOUR and "pret" not in a["types"] and "erreur" in a["types"] and a["etat"] == "erreur" and a["perime"] is True
              and a["termine"] and a["r"].get("ok") is False and a["r"].get("erreur") == MIS_A_JOUR and a["requetes"] == 0
              and f"v=autre-version-de-l-essai" in a["url"], a)
        B.chk("P6 · version · le témoin : la version du manifeste dans l'adresse (« v ») — « pret », et la réponse",
              b_["pret"] == "pret" and b_["ok"] and f"v={VERSION_SITE}" in b_["url"], b_)
        B.chk("P6 · version · un travailleur RECRÉÉ après une mort, le site mis à jour entre-temps : il ne sert pas l'autre version (erreur nommée, "
              "périmé, aucun « pret »)",
              c["mort"] and c["recree"] and f"v={VERSION_SITE}" in c["url"] and c["r"].get("ok") is False and c["r"].get("erreur") == MIS_A_JOUR
              and c["etat"] == "erreur" and c["perime"] is True and "pret" not in c["types"], c)
        B.chk("P6 · version · note PDF dont l'empreinte ne correspond plus : manifeste relu de la même version, le message d'empreinte (le témoin) ; "
              "d'une autre version (mise en ligne depuis le démarrage), l'erreur nommée",
              d["n1"]["ok"] is False and "ne correspond pas à son manifeste" in str(d["n1"].get("erreur")) and d["n2"]["ok"] is False
              and d["n2"].get("erreur") == MIS_A_JOUR, d)
    if "ouverture" in out:
        o = out["ouverture"]
        B.chk("sonde par ouverture (dans node, la page sondée) : nouvelleOuverture efface les marques « stab: », garde __sondes, et les marques reprennent",
              o["avant"] > 0 and o["apres"] == 0 and o["sondes_apres"] == o["sondes_avant"] and o["sondes_avant"] > 0 and o["ouverture"] == 3 and o["remarque"] == 1, o)


# ====================================================================== P5 · la mort
def p5_juger(B, out, refs):
    if not out:
        B.chk("P5 · le pilote de la mort a rendu ses mesures", False, "aucune sortie")
        return
    f = out["fatal"]
    B.chk("P5 · erreur fatale de Pyodide : la requête en vol et celle en file rendent {ok:false, mort:true}",
          f["r0"] and all(r.get("ok") is False and r.get("mort") is True for r in (f["ra"], f["rb"])), {"ra": f["ra"], "rb": f["rb"]})
    B.chk("P5 · état « mort », travailleur arrêté, moteurExiste() faux", f["mortEtat"] == "mort" and f["w1termine"] and f["existe0"] and not f["existeMort"], f)
    egal, _, det = comparer("poutre", refs["nouveau"]["poutre"], f["r2json"], 1)
    B.chk("P5 · la demande suivante recrée UN travailleur (le même Moteur, moteurPartage() le rend) et répond = CPython",
          f["nApres"] == f["nAvant"] + 1 and f["r2ok"] and egal and f["meme"] and f["existeApres"] and f["etatApres"] == "pret", {"det": det, **{k: f[k] for k in ("nAvant", "nApres", "meme", "etatApres")}})
    c = out["chute"]
    B.chk("P5 · le fil du travailleur qui s'arrête pendant une requête : {ok:false, mort:true}, puis un travailleur neuf répond",
          c["ra"].get("mort") is True and c["ra"].get("ok") is False and c["nApres"] == c["nAvant"] + 1 and c["r2ok"], c)
    k = out["compat"]
    egal, _, det = comparer("garde_corps", refs["nouveau"]["garde_corps"], k["json"], 1)
    B.chk("compatibilité · new Moteur(surEtape, module) puis demander(fabrique) et demander(objet) avec le VRAI travailleur : étapes reçues, préchauffage annoncé, réponses = CPython",
          k["ok"] and egal and k["biblio"] and len(k["etapes"]) >= 3 and k["premier"] == "prechauffer" and k["url"].endswith("/js/travailleur.js"), {"det": det, "etapes": k["etapes"], "premier": k["premier"]})


def client_juger(B, out, sonde):
    if not out:
        B.chk(f"client ({'sonde' if sonde else 'sans sonde'}) · le pilote a rendu ses essais", False, "aucune sortie")
        return
    for nom, ok, info in out:
        B.chk(f"{nom}{' [sonde]' if sonde else ''}", ok, info)


# ====================================================================== un passage complet sur un site (le vrai ou une copie mutée)
def jouer(B, dossier, site, refs, pilotes, cles=None, parallele=2):
    """Lance les pilotes demandés (en parallèle) sur ``site`` et juge ; rend les durées."""
    moteur = "file://" + os.path.join(site, "js", "moteur.js")
    sonde = "file://" + os.path.join(site, "js", "sonde.js")
    etat_gc = json.loads(refs["nouveau"]["garde_corps"])["etat"]
    etat_poutre = json.loads(refs["nouveau"]["poutre"])["etat"]
    cles = cles or CLES_PRECHAUFFAGE
    durees = {}
    taches = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        if "partage" in pilotes:
            taches["partage"] = ex.submit(node, dossier, "partage.mjs", {"moteur": moteur, "ordre": ORDRE_P2, "notes": list(SW.NOTE_PDF), "infos": INFOS_NOTE,
                                                                       "biblio": BIBLIO_REQUETES})
        # (le même pilote sous la sonde, en deux processus : les parties d'avant S4-C, et celles du lot « moteur »)
        for cle_pilote in ("sonde", "neufs"):
            if cle_pilote in pilotes:
                taches[cle_pilote] = ex.submit(node, dossier, "sonde.mjs", {"moteur": moteur, "sonde": sonde, "cles": cles, "parallele": parallele, "infos": INFOS_NOTE,
                                                                          "etat_gc": etat_gc, "etat_poutre": etat_poutre, "parties": pilotes[cle_pilote],
                                                                          "version": VERSION_SITE, "retard": RETARD_NUMPY})
        if "mort" in pilotes:
            taches["mort"] = ex.submit(node, dossier, "mort.mjs", {"moteur": moteur})
        if "client" in pilotes:
            taches["client"] = ex.submit(node, dossier, "client.mjs", {"moteur": moteur, "sondeJs": sonde, "sonde": False})
            taches["client_sonde"] = ex.submit(node, dossier, "client.mjs", {"moteur": moteur, "sondeJs": sonde, "sonde": True})
        res = {k: f.result() for k, f in taches.items()}
    for k, (out, d, err) in res.items():
        durees[k] = d
        if out is None and B.bavard:
            print(f"     ({k} : {err[-600:]})")
    if "partage" in res:
        p2_juger(B, res["partage"][0], refs)
    if "sonde" in res:
        p4_juger(B, res["sonde"][0], refs, cles)
    if "neufs" in res:
        p4_juger(B, res["neufs"][0], refs, cles)
    if "mort" in res:
        p5_juger(B, res["mort"][0], refs)
    if "client" in res:
        client_juger(B, res["client"][0], False)
    if "client_sonde" in res:
        # (la page sondée importe sonde.js depuis le site joué)
        client_juger(B, res["client_sonde"][0], True)
    return durees


# ====================================================================== les MUTATIONS (le moteur modifié d'un trait, dans une copie)
TOUTES_PARTIES = ["isolement", "guet_requete", "guet_prechauffer", "note_numpy", "saut", "chaine", "par_cle", "ouverture"]
# les parties du lot « moteur » (revue S4) : un second processus du même pilote, en parallèle
NOUVELLES_PARTIES = ["annonce", "fond", "fond_demarrage", "guet_horsligne", "version"]
CLES_MUTATION = ["dalle", "garde_corps", "poutre", "biblio_acier"]
# (nom, fichier du site, ancien, nouveau, pilotes, contrôles qui doivent rougir — début de leur nom)
MUTATIONS = [
    ("m_prech_tout", "js/travailleur.js", "        registre.module(cle)\n", "        [registre.module(c) for c in registre.cles()]\n",
     {"sonde": ["isolement"]}, ["P4 · dalle · préchauffé seul", "P4 · poutre · préchauffé seul"]),
    ("m_prech_rien", "js/travailleur.js", "          try { importerCle(cle); } catch (e) { if (estFatale(e)) throw e; /* avalée : la vraie requête dira l'erreur */ }",
     "          void importerCle;", {"sonde": ["isolement"]}, ["P4 · poutre · préchauffé seul"]),
    ("m_numpy_toujours", "js/travailleur.js", "  return !!man && (!Array.isArray(man.avec_numpy)", "  return true || (!Array.isArray(man.avec_numpy)",
     {"sonde": ["isolement"]}, ["P4 · poutre · numpy chargé", "P4 · dalle · numpy chargé"]),
    ("m_guet_non_remis", "js/travailleur.js", "      guet.demande = false;\n      await chargerNumpy();\n      json = traiter(data.requete);",
     "      await chargerNumpy();\n      json = traiter(data.requete);", {"sonde": ["guet_requete"]}, ["P4 · guet · après le rejeu"]),
    ("m_guet_sans_rejeu", "js/travailleur.js", "      await chargerNumpy();\n      json = traiter(data.requete);", "      await chargerNumpy();",
     {"sonde": ["guet_requete"]}, ["P4 · guet · avec_numpy vide : « nouveau »"]),
    # etat-10 : la demande remise à faux APRÈS le chargement (l'ancien ordre) — hors ligne, tout reste refusé
    ("m_guet_remis_apres", "js/travailleur.js", "      guet.demande = false;\n      await chargerNumpy();\n      json = traiter(data.requete);",
     "      await chargerNumpy();\n      guet.demande = false;\n      json = traiter(data.requete);", {"neufs": ["guet_horsligne"]}, ["P4 · guet hors ligne · requête"]),
    ("m_guet_prech_non_remis", "js/travailleur.js", "{ guet.demande = false; attend = true; }", "{ attend = true; }",
     {"neufs": ["guet_horsligne"]}, ["P4 · guet hors ligne · préchauffage"]),
    # etat-09 / C4 : le préchauffage attend numpy DANS la chaîne ; ou il compte comme une intention réelle au démarrage
    ("m_prech_numpy_chaine", "js/travailleur.js", "        if (numpyPour(cle) && !numpyCharge) attend = true;\n        else {",
     "        if (numpyPour(cle) && !numpyCharge) await chargerNumpy();\n        {", {"neufs": ["fond"]}, ["P4 · fond · un préchauffage de soudures (numpy"]),
    ("m_prech_intention", "js/travailleur.js", "    dernierPrechauffage = n;\n    tache = () => prechauffer(cle, n);",
     "    dernierPrechauffage = n;\n    noterIntention(cle);\n    tache = () => prechauffer(cle, n);", {"neufs": ["fond_demarrage"]},
     ["P4 · fond · un préchauffage de soudures posté PENDANT"]),
    ("m_guet_sans_blocage", "js/travailleur.js", "    if (guet) guet.en_cours = true;\n", "", {"neufs": ["fond"]}, ["P4 · fond · pendant que numpy"]),
    # l'annonce : ignorée ; ou elle importe (comme un préchauffage)
    ("m_annonce_ignoree", "js/travailleur.js", "    if (CLE.test(cle)) noterIntention(cle, true);", "    void noterIntention;",
     {"neufs": ["annonce"]}, ["P4 · annonce · plancher annoncé"]),
    ("m_annonce_importe", "js/travailleur.js", "    if (CLE.test(cle)) noterIntention(cle, true);",
     "    if (CLE.test(cle)) { noterIntention(cle, true); dernierPrechauffage = n; tache = () => prechauffer(cle, n); }",
     {"neufs": ["annonce"]}, ["P4 · annonce · plancher annoncé"]),
    # la version du site : non comparée au démarrage ; ni à l'empreinte de la note
    ("m_version_ignoree", "js/travailleur.js", "  if (VERSION && man.version !== VERSION) throw perime();", "  void perime;",
     {"neufs": ["version"]}, ["P6 · version · le document d'une AUTRE", "P6 · version · un travailleur RECRÉÉ"]),
    ("m_note_version", "js/travailleur.js", "  return m && manifeste && m.version !== manifeste.version ? perime() : new Error(message);",
     "  return new Error(message);", {"neufs": ["version"]}, ["P6 · version · note PDF"]),
    ("m_cle_adresse", "js/travailleur.js", "    if (q && typeof q.module === \"string\" && q.module) return q.module;",
     "    if (q && typeof q.module === \"string\" && q.module) return new URL(self.location.href).searchParams.get(\"module\") || \"\";",
     {"sonde": ["note_numpy"]}, ["P3b"]),
    ("m_note_sans_cle", "js/travailleur.js", "m.avec_numpy.includes(cle)) await chargerNumpy();", "m.avec_numpy.includes(\"\")) await chargerNumpy();",
     {"sonde": ["note_numpy"]}, ["P3b"]),
    ("m_fatale_ignoree", "js/travailleur.js", "    if (pyodideTombe()) throw new Error(\"Pyodide fatally failed\");\n  } catch (e) {\n    if (estFatale(e)) { mourir(e); return; }\n    json =",
     "  } catch (e) {\n    if (estFatale(e)) { mourir(e); return; }\n    json =", {"mort": True}, ["P5 · erreur fatale"]),
    ("m_chaine", "js/travailleur.js", "  if (tache) enchainer(tache);", "  if (tache) tache();",
     {"sonde": ["chaine"]}, ["P4 · une seule chaîne"]),
    ("m_saut", "js/travailleur.js", "const estSaute = (n) => n !== dernierPrechauffage || derniereRequete > n;", "const estSaute = (n) => n !== dernierPrechauffage;",
     {"sonde": ["saut"]}, ["P4 · un préchauffage n'est jamais traité devant"]),
    ("m_message_sans_sonde", "js/travailleur.js", "{ if (SONDE) postMessage({ type: \"prechauffe\"", "{ postMessage({ type: \"prechauffe\"",
     {"partage": True}, ["P2 · sans « ?sonde=1 »"]),
    ("m_sonde_une_fois", "js/travailleur.js", "const premiere = SONDE && !vues.has(cle);", "const premiere = SONDE && session;",
     {"sonde": ["par_cle"]}, ["P4 · la sonde date la première requête de CHAQUE clé"]),
    ("m_prech_calcule", "js/travailleur.js", "        registre.module(cle)\n",
     "        registre.module(cle); routeur.traiter('{\"module\": \"%s\", \"action\": \"nouveau\"}' % cle)\n",
     {"partage": True}, ["P2 · soudures · 1ʳᵉ passe", "P2 · les exceptions employées"]),
    ("m_prech_devant", "js/moteur.js", "    this.attente = cle;\n    this._prevoir();", "    this.attente = cle;\n    this.w.postMessage({ type: \"prechauffer\", cle });",
     {"client": True}, ["préchauffage · jamais posté"]),
    ("m_fermer_modifie", "js/moteur.js", "if (r.client === client && !r.modifie) {", "if (r.client === client) {",
     {"client": True}, ["client · la requête en vol et la modification", "client · la modification gardée"]),
    ("m_fermer_lectures", "js/moteur.js", "if (r.client === client && !r.modifie) {", "if (false) {",
     {"client": True}, ["client · fermer() est synchrone"]),
    ("m_mort_attentes", "js/moteur.js", "    for (const r of tout) this._finir(r, { ...rep });", "    void tout;",
     {"client": True, "mort": True}, ["mort · toutes les requêtes", "P5 · erreur fatale"]),
    ("m_sans_relance", "js/moteur.js", "    if (this.etat === \"mort\") this._demarrer();", "    void 0;",
     {"client": True, "mort": True}, ["mort · la demande suivante", "partagé · mort", "P5 · la demande suivante"]),
    ("m_ecouteur", "js/moteur.js", "    w.onmessage = ({ data }) => { if (this.w === w) this._recevoir(data); };",
     "    w.addEventListener(\"message\", ({ data }) => { if (this.w === w) this._recevoir(data); });", {"client": True}, ["setter ·"]),
    ("m_adresse_module", "js/moteur.js", "const url = new URL(\"./travailleur.js\", import.meta.url);", "const url = new URL(\"./travailleur.js?module=poutre\", import.meta.url);",
     {"client": True}, ["compatibilité · le travailleur est créé SANS module"]),
    ("m_fabrique_tot", "js/moteur.js", "this.file.push({ client, fabrique, note, modifie, resoudre, rejeter });",
     "this.file.push({ client, fabrique: typeof fabrique === \"function\" ? fabrique() : fabrique, note, modifie, resoudre, rejeter });",
     {"client": True}, ["client · la fabrique de la suivante"]),
    ("m_tour_microtache", "js/moteur.js", "if (apresReponse) setTimeout(f, 0); else queueMicrotask(f);", "queueMicrotask(f);",
     {"client": True}, ["client · la fabrique de la suivante"]),
    ("m_sondes_historique", "js/moteur.js", "(globalThis.__sondes = globalThis.__sondes || []).push(data);", "globalThis.__sondes = [data];",
     {"client": True}, ["sonde · chaque message"]),
    ("m_ouverture", "js/sonde.js", "  for (const m of performance.getEntriesByType(\"mark\")) if (m.name.startsWith(PREFIXE)) performance.clearMarks(m.name);",
     "  void 0;", {"client": True, "sonde": ["ouverture"]}, ["sonde par ouverture"]),
    # etat-07 / deploiement-01 : un démarrage raté reste définitif ; ou il est relancé à CHAQUE requête ; ou même périmé
    ("m_erreur_definitive", "js/moteur.js", "  else if (partage.etat === \"erreur\" && !partage.perime) partage._redemarrer();\n", "",
     {"client": True}, ["partagé · démarrage raté (message", "partagé · démarrage raté (fil"]),
    ("m_redemarre_requete", "js/moteur.js", "    if (this.etat === \"erreur\" && !this.w) return Promise.resolve(",
     "    if (this.etat === \"erreur\") this._redemarrer();\n    if (false) return Promise.resolve(", {"client": True},
     ["démarrage raté (erreur du fil AVANT", "partagé · démarrage raté (message", "partagé · démarrage raté (fil"]),
    ("m_redemarre_perime", "js/moteur.js", "  else if (partage.etat === \"erreur\" && !partage.perime) partage._redemarrer();",
     "  else if (partage.etat === \"erreur\") partage._redemarrer();", {"client": True}, ["partagé · démarrage PÉRIMÉ"]),
    # la version du document n'est plus passée au travailleur
    ("m_version_absente", "js/moteur.js", "    if (v) url.searchParams.set(\"v\", v);", "    void v;",
     {"client": True, "neufs": ["version"]}, ["version · l'adresse du travailleur", "P6 · version · le document d'une AUTRE"]),
    # l'annonce n'est pas postée
    ("m_annonce_moteur", "js/moteur.js", "    this.w.postMessage({ type: \"annoncer\", cle });", "    void 0;",
     {"client": True, "neufs": ["annonce"]}, ["annonce · postée aussitôt", "P4 · annonce · plancher annoncé"]),
]
# une table PARTAGÉE de beton/ (le coefficient d'adhérence des ancrages) retouchée à l'import d'un établi, dans le
# paquet : le préchauffage et P2 des établis n'en voient rien ; la passe des bibliothèques, jouée APRÈS les établis
# dans le même interpréteur, doit rougir
FUITE_BIBLIO = ("api_web/poutre.py", b"\n# mutation de l'essai : une table partagee de beton/ retouchee a l'import d'un etabli\n"
                b"from beton.bibliotheque import ancrages as _ANCRAGES_ESSAI\n_ANCRAGES_ESSAI.ETA1[\"bonnes\"] = 0.95\n")


def copie_du_site(base, js_de=WEB, paquet_mute=False, retouche=None):
    """Une copie du site pour une mutation : js/ recopié, vendor lié ; moteur lié (ou recopié, paquet muté pour P0).
    ``retouche`` = (fichier du paquet, octets ajoutés) : le paquet recopié, ce fichier allongé, et le manifeste recopié
    porte la NOUVELLE empreinte (le travailleur l'accepte : seul le contenu du moteur change)."""
    site = os.path.join(base, "site")
    os.makedirs(site)
    shutil.copytree(os.path.join(js_de, "js"), os.path.join(site, "js"))
    os.makedirs(os.path.join(site, "vendor"))
    os.symlink(VENDOR, os.path.join(site, "vendor", os.path.basename(VENDOR)))
    if paquet_mute or retouche:
        shutil.copytree(os.path.join(WEB, "moteur"), os.path.join(site, "moteur"))
        chemin = os.path.join(site, "moteur", "moteur.zip")
        nom, ajout = retouche or ("api_web/registre.py", b"\n# mutation\n")
        with zipfile.ZipFile(chemin) as z:
            contenu = [(i, z.read(i.filename)) for i in z.infolist()]
        assert sum(i.filename == nom for i, _ in contenu) == 1, f"{nom} n'est pas dans le paquet"
        with zipfile.ZipFile(chemin, "w", zipfile.ZIP_DEFLATED) as z:
            for i, b in contenu:
                z.writestr(i, b + ajout if i.filename == nom else b)
        if retouche:
            chemin_man = os.path.join(site, "moteur", "manifeste.json")
            man = json.load(open(chemin_man, encoding="utf-8"))
            man["sha256"] = hashlib.sha256(open(chemin, "rb").read()).hexdigest()
            json.dump(man, open(chemin_man, "w", encoding="utf-8"))
    else:
        os.symlink(os.path.join(WEB, "moteur"), os.path.join(site, "moteur"))
    return site


# la mutation de P1 : la formule f_yd du moteur poutre (+0,1 %), dans l'interpréteur. Le code n'échoue pas si le texte
# manque : il ne pose MUTEE qu'après l'avoir remplacé (le pilote le dit : « @@MUTEE »)
CIBLE_FORMULE = ('"fyd": fyk / gamma_s}', '"fyd": fyk / gamma_s * 1.001}')
GARDE_FORMULE = "m_formule : l'interpréteur a tourné et le texte à muter est présent une fois"


def m_formule(dossier, cible=CIBLE_FORMULE):
    """Joue la mutation de P1 et rend son Bilan. Un interpréteur qui n'a pas tourné, ou un texte à muter absent, est un
    KO sous un nom À PART (GARDE_FORMULE, jamais « P1 · … ») : il ne compte JAMAIS comme « rougit »."""
    ancien, nouveau = cible
    code = (f"p = '/moteur/beton/poutre/moteur.py'\ns = open(p, encoding='utf-8').read()\nif s.count({ancien!r}) == 1:\n"
            f"    open(p, 'w', encoding='utf-8').write(s.replace({ancien!r}, {nouveau!r}))\n    MUTEE = True\n")
    res, _, err = p1_lancer(dossier, WEB, [["poutre"]], mutation=code)
    b = Bilan(bavard=False)
    if not res:
        b.chk(GARDE_FORMULE, False, err)
    else:
        b.chk("P1 · la formule f_yd du moteur poutre (+0,1 %) dans l'interpréteur", res[0][1] == 0 and res[0][0] > 0, res)
    return b


def mutations(B, dossier, refs, choisies):
    speciales = ["m_paquet", "m_formule", "m_formule_garde", "m_biblio_fuite"]
    choisies = [m for m in MUTATIONS if not choisies or m[0] in choisies] + [(s,) for s in speciales if not choisies or s in choisies]

    def une(mut):
        base = tempfile.mkdtemp(prefix="mutation_", dir=dossier)
        t0 = time.time()
        b = Bilan(bavard=False)
        if mut[0] == "m_paquet":
            site = copie_du_site(base, paquet_mute=True)
            p0(b, site)
            return mut[0], ["P0"], b, time.time() - t0
        if mut[0] == "m_formule":
            return mut[0], ["P1"], m_formule(dossier), time.time() - t0
        if mut[0] == "m_formule_garde":
            # la GARDE de m_formule : un texte à muter absent ne doit rien rougir sous « P1 » (il rougit sous son nom à part)
            g = m_formule(dossier, cible=("TEXTE_DISPARU de l'essai", "x"))
            ok = not [n for n, _ in g.ko if n.startswith("P1")] and any(n == GARDE_FORMULE for n, _ in g.ko)
            b.chk("m_formule_garde : sur un texte absent, aucun KO « P1 » (la mutation n'est jamais comptée « rougit ») et la garde le dit", ok,
                  [n for n, _ in g.ko])
            # (attendu ICI : que ce contrôle PASSE — jugé à part ci-dessous, pas comme un « rougit »)
            return mut[0], None, b, time.time() - t0
        if mut[0] == "m_biblio_fuite":
            site = copie_du_site(base, retouche=FUITE_BIBLIO)
            jouer(b, dossier, site, refs, {"partage": True})
            return mut[0], ["P2 · biblio"], b, time.time() - t0
        nom, fichier, ancien, nouveau, pilotes, attendus = mut
        site = copie_du_site(base)
        chemin = os.path.join(site, fichier)
        texte = open(chemin, encoding="utf-8").read()
        if texte.count(ancien) != 1:
            b.chk(f"{nom} : le texte à muter est présent une fois", False, texte.count(ancien))
            return nom, attendus, b, time.time() - t0
        open(chemin, "w", encoding="utf-8").write(texte.replace(ancien, nouveau))
        jouer(b, dossier, site, refs, pilotes, cles=CLES_MUTATION, parallele=2)
        return nom, attendus, b, time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for nom, attendus, b, d in ex.map(une, choisies):
            if attendus is None:   # un contrôle de la garde elle-même : il doit passer
                B.chk(f"mutation · {nom} : la garde tient [{d:.0f} s]", not b.ko and bool(b.ok), {"ko": [n[:120] for n, _ in b.ko]})
                continue
            rouges = [n for n, _ in b.ko if any(n.startswith(a) for a in attendus)]
            B.chk(f"mutation · {nom} : rougit ({'; '.join(r[:70] for r in rouges[:3]) or 'rien'}) [{d:.0f} s]", bool(rouges),
                  {"ko": [n[:90] for n, _ in b.ko][:6], "attendus": attendus})


# ====================================================================== l'ensemble
def main(argv):
    t0 = time.time()
    B = Bilan()
    choisies = None
    for a in argv:
        if a == "--mutations":
            choisies = []
        elif a.startswith("--mutations="):
            choisies = [x for x in a.split("=", 1)[1].split(",") if x]
    if not p0(B, WEB):
        print(f"\nDurée : {time.time() - t0:.1f} s\n\nRÉSULTAT : {len(B.ok)} OK, {len(B.ko)} échec(s)")
        return 1
    dossier = tempfile.mkdtemp(prefix="moteur_partage_")
    try:
        ecrire_node(dossier)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            # P1 (le plus long) part le premier, CPython en parallèle
            f_p1 = pool.submit(p1_lancer, dossier, WEB, [[], list(reversed(REG.cles()))])
            f_cp = pool.submit(p1_cpython)
            t_ref = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool_cp:
                refs = references(pool_cp)
            print(f"     (références CPython, processus neufs : {time.time() - t_ref:.1f} s)")
            noms = {nom_sortie(dossier, "client.mjs") for _ in range(50)}
            B.chk("pilotes · cinquante lancements du même pilote, dans la même milliseconde, ont cinquante fichiers de sortie (jamais deux "
                  "pilotes qui s'écrasent)", len(noms) == 50, len(noms))
            B.chk("P6 · version · le texte nommé « Le site a été mis à jour : rechargez la page » : la copie de l'essai est celle de js/travailleur.js",
                  f'const MIS_A_JOUR = "{MIS_A_JOUR}";' in open(os.path.join(WEB, "js", "travailleur.js"), encoding="utf-8").read())
            durees = jouer(B, dossier, WEB, refs, {"partage": True, "sonde": TOUTES_PARTIES, "neufs": NOUVELLES_PARTIES, "mort": True, "client": True})
            print("     (pilotes : " + ", ".join(f"{k} {v:.1f} s" for k, v in durees.items()) + ")")
            if choisies is not None:
                mutations(B, dossier, refs, choisies)
            res, d1, err = f_p1.result()
            cp, dcp = f_cp.result()
            print(f"     (P1 : Pyodide deux ordres {d1:.1f} s ; CPython {dcp:.1f} s){'' if res else ' ' + err}")
            p1_juger(B, res, cp)
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    print(f"\nDurée : {time.time() - t0:.1f} s")
    print(f"\nRÉSULTAT : {len(B.ok)} OK, {len(B.ko)} échec(s)")
    for nom, info in B.ko:
        print("   -", nom[:160], "|", str(info).splitlines()[0][:200] if str(info) else "")
    return 1 if B.ko else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
