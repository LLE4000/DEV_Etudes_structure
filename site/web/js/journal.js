// Le JOURNAL D'ACTIVITÉ du bureau, enregistré DANS CE NAVIGATEUR (docs/site/PROJETS_EQUIPE.md § 3) : qui a fait quoi,
// sur quelle note, dans quel projet. Une ligne : { t (l'heure), q (la signature de l'auteur, ou null : personne ne
// s'était présenté), k (le geste : une clé de VERBES), n ({ id, nom, m } : la note), p (le projet), s (le sous-projet),
// de (l'ancien nom d'un renommage), x (un nom : le membre ajouté) }. Les plus récentes d'abord.
//
// Le journal ne doit JAMAIS empêcher le travail : noter() ne lève pas, et un journal qui ne s'écrit pas (stockage
// plein, refusé) ne dit rien — la note, elle, a sa propre porte (js/notes.js) et ses propres messages.
// Aucun effet au chargement ; pas de document ici (ces fonctions tournent aussi dans node).
import * as M from "./memoire.js";
import { signature, signaturePropre } from "./equipe.js";

export const CLE = "stab.journal.v1";
export const MAX = 400;                          // au-delà, les plus anciennes lignes tombent
export const FUSION_MS = 30 * 60 * 1000;         // modifier une note dix fois en une demi-heure : UNE ligne, à la dernière heure
const SILENCE_MS = 60 * 1000;                    // et pas plus d'une écriture du journal par minute pour une même note

// le geste → ce qu'on lit : « LLE a créé la note … »
export const VERBES = {
  "note.creee": "a créé la note", "note.modifiee": "a modifié la note", "note.renommee": "a renommé la note",
  "note.classee": "a rangé la note", "note.supprimee": "a supprimé la note", "note.exportee": "a exporté la note",
  "note.importee": "a importé la note", "note.pdf": "a produit la note PDF",
  "projet.cree": "a créé le projet", "projet.renomme": "a renommé le projet", "projet.archive": "a archivé le projet",
  "projet.rouvert": "a rouvert le projet", "projet.supprime": "a supprimé le projet",
  "sous.cree": "a créé le sous-projet", "sous.renomme": "a renommé le sous-projet", "sous.supprime": "a supprimé le sous-projet",
  "membre.ajoute": "a ajouté à l'équipe", "membre.arrive": "a rejoint l'équipe", "membre.retire": "a retiré de l'équipe",
  "membre.admin": "a nommé administrateur", "membre.membre": "a retiré le rôle d'administrateur à",
  "module.verifie": "a vérifié le module", "module.retire": "a retiré la vérification du module",
  "gabarit.change": "a changé le gabarit du bureau :", "gabarit.usine": "a rendu aux valeurs d'usine le gabarit :",
};

const texte = (x, max = 120) => (typeof x === "string" ? x : "").slice(0, max);
const vide = () => [];
const valide = (v) => Array.isArray(v);

// une ligne LUE : on n'en garde que des données bien formées (un journal réécrit à la main, un ancien format)
function ligne(e) {
  if (!e || typeof e !== "object" || !Object.hasOwn(VERBES, e.k) || !Number.isFinite(e.t)) return null;
  const n = e.n && typeof e.n === "object" ? { id: texte(e.n.id, 40), nom: texte(e.n.nom), m: texte(e.n.m, 40) } : null;
  return { t: e.t, q: signaturePropre(e.q), k: e.k, n, p: texte(e.p), s: texte(e.s), de: texte(e.de), x: texte(e.x) };
}

// Les lignes du journal, les plus récentes d'abord. Filtres : ``projet`` (son nom, comparé sans casse), ``membre``
// (l'identifiant de l'auteur), ``note`` (l'identifiant de la note) ; ``max`` lignes au plus.
export function lire({ projet = null, membre = null, note = null, max = MAX } = {}) {
  const cleP = projet === null ? null : String(projet).trim().toLocaleLowerCase("fr");
  const out = [];
  for (const brut of M.lire(CLE, vide, valide).valeur) {
    const e = ligne(brut);
    if (!e) continue;
    if (cleP !== null && e.p.trim().toLocaleLowerCase("fr") !== cleP) continue;
    if (membre !== null && (!e.q || e.q.id !== membre)) continue;
    if (note !== null && (!e.n || e.n.id !== note)) continue;
    out.push(e);
    if (out.length >= max) break;
  }
  return out;
}

// la dernière écriture du journal pour une note et un geste (en mémoire vive : ne coûte aucune lecture du stockage)
const derniers = new Map();

// Noter un geste. ``note`` : la note concernée ({ id, nom, module, chantier, dossier }) ; ``projet``, ``sous`` : sinon
// le projet et le sous-projet nommés ; ``de`` : l'ancien nom ; ``membre`` : le nom d'un membre ; ``module`` : le nom d'un module. Rend vrai si une ligne
// a été écrite ou avancée. Ne lève jamais.
export function noter(quoi, { note = null, projet = "", sous = "", de = "", membre = "", module = "", qui = undefined, quand = Date.now() } = {}) {
  try {
    if (!Object.hasOwn(VERBES, quoi)) return false;
    const q = qui === undefined ? signature() : signaturePropre(qui);
    const n = note ? { id: texte(note.id, 40), nom: texte(note.nom), m: texte(note.module, 40) } : null;
    const e = { t: quand, q, k: quoi, n, p: texte(note ? note.chantier : projet), s: texte(note ? note.dossier : sous), de: texte(de), x: texte(membre || module) };
    if (quoi === "note.modifiee" && n) {
      const vu = derniers.get(n.id);
      if (vu && quand - vu < SILENCE_MS) return false;
      derniers.set(n.id, quand);
    }
    return M.changer(CLE, vide, valide, (j) => {
      if (quoi === "note.modifiee" && n) {
        // la même personne a créé ou modifié cette note il y a moins d'une demi-heure : la même ligne, avancée
        const i = j.findIndex((x) => x && x.n && x.n.id === n.id && (x.k === "note.modifiee" || x.k === "note.creee" || x.k === "note.importee")
          && ((x.q && x.q.id) || "") === ((q && q.id) || "") && quand - x.t < FUSION_MS);
        if (i >= 0 && j.slice(0, i).every((x) => !x || !x.n || x.n.id !== n.id)) {
          if (j[i].k !== "note.modifiee") return undefined;            // créée (ou importée) à l'instant : la ligne dit déjà tout
          const [x] = j.splice(i, 1);
          j.unshift({ ...x, ...e });
          return undefined;
        }
      }
      j.unshift(e);
      if (j.length > MAX) j.length = MAX;
      return undefined;
    });
  } catch (err) { return false; }
}

// « à l'instant », « il y a 5 min », « il y a 2 h », « hier, 14:05 », « 12 nov., 09:30 » (``maintenant`` : pour les essais)
export function ilYA(t, maintenant = Date.now()) {
  const d = Math.max(0, maintenant - t);
  if (d < 60 * 1000) return "à l'instant";
  if (d < 60 * 60 * 1000) return `il y a ${Math.floor(d / 60000)} min`;
  const jour = new Date(t), auj = new Date(maintenant);
  const h = jour.toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit" });
  if (jour.toDateString() === auj.toDateString()) return `il y a ${Math.floor(d / 3600000)} h`;
  const hier = new Date(maintenant - 86400000);
  if (jour.toDateString() === hier.toDateString()) return `hier, ${h}`;
  return `${jour.toLocaleDateString("fr-BE", { day: "numeric", month: "short" })}, ${h}`;
}

// le jour d'une ligne, pour grouper : « Aujourd'hui », « Hier », « mardi 10 novembre »
export function jourDe(t, maintenant = Date.now()) {
  const jour = new Date(t), auj = new Date(maintenant), hier = new Date(maintenant - 86400000);
  if (jour.toDateString() === auj.toDateString()) return "Aujourd'hui";
  if (jour.toDateString() === hier.toDateString()) return "Hier";
  const s = jour.toLocaleDateString("fr-BE", { weekday: "long", day: "numeric", month: "long" });
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// Ce qu'une ligne DIT, en textes (l'écran les échappe et les compose) :
//   { qui (le nom, ou « Quelqu'un »), verbe, objet (le nom de la note, du projet, du membre), lieu (« Réalex › Étude »),
//     suite (« (avant : …) ») }
export function phrase(e) {
  const qui = e.q ? e.q.nom : "Quelqu'un";
  const lieu = [e.p, e.s].filter(Boolean).join(" › ");
  if (e.k.startsWith("note.")) return { qui, verbe: VERBES[e.k], objet: (e.n && e.n.nom) || "sans nom", lieu, suite: e.k === "note.renommee" && e.de ? `(avant : ${e.de})` : "" };
  if (e.k.startsWith("projet.")) return { qui, verbe: VERBES[e.k], objet: e.p, lieu: "", suite: e.k === "projet.renomme" && e.de ? `(avant : ${e.de})` : "" };
  if (e.k.startsWith("sous.")) return { qui, verbe: VERBES[e.k], objet: e.s, lieu: e.p, suite: e.k === "sous.renomme" && e.de ? `(avant : ${e.de})` : "" };
  if (e.k === "membre.arrive") return { qui, verbe: VERBES[e.k], objet: "", lieu: "", suite: "" };
  // un module vérifié, un rôle donné ou retiré : le geste, puis son objet (le nom du module, de la personne)
  // (et, lot 6, le gabarit du bureau d'un module : son nom)
  if (e.k.startsWith("module.") || e.k.startsWith("gabarit.") || e.k === "membre.admin" || e.k === "membre.membre") return { qui, verbe: VERBES[e.k], objet: e.x, lieu: "", suite: "" };
  return { qui, verbe: e.k === "membre.retire" ? "a retiré" : "a ajouté", objet: e.x, lieu: "", suite: e.k === "membre.retire" ? "de l'équipe" : "à l'équipe" };
}
