// Les notes du bureau, enregistrées DANS CE NAVIGATEUR (localStorage) en attendant les comptes partagés (Supabase).
// Une note : { id, module, nom, chantier, dossier, indice, cree, modifie, etat (l'état du moteur), resume, vignette }.
const CLE = "stab.notes.v1";
// La copie d'une mémoire ILLISIBLE : son texte brut, tel quel, sous cette clé suivie de l'heure de la copie
// (« stab.notes.v1.illisible.2026-11-07T17:08:03.120Z »). Elle n'est jamais réécrite par le site.
export const PREFIXE_COPIE = `${CLE}.illisible.`;

// Les modules du site viennent du registre (api_web/registre.py), écrits par construire.py dans js/modules.js :
// adresses SANS « .html » (Cloudflare sert poutre.html à l'adresse « poutre »), nom, famille.
import { MODULES } from "./modules.js";
import { telecharger } from "./commun.js";
export { MODULES };

// Pourquoi une écriture a été refusée : le mot court de l'indicateur d'enregistrement (« Non enregistré : … »), la
// phrase longue du message. Une seule source pour l'établi et l'accueil.
export const RAISONS = {
  plein: { court: "stockage plein", long: "le stockage de ce navigateur est plein" },
  illisible: { court: "mémoire illisible", long: "la mémoire des notes est illisible" },
  indisponible: { court: "stockage refusé", long: "ce navigateur refuse d'enregistrer (navigation privée, réglage du site)" },
};

// La mémoire des notes, LUE. Une mémoire ABSENTE (jamais écrite) n'est pas une mémoire ILLISIBLE (abîmée : tronquée,
// réécrite par autre chose, autre chose qu'un objet de notes) : les deux n'ont aucune note lisible, mais la seconde
// porte peut-être les notes de l'utilisateur. La réécrire avec la seule note courante les effaçait toutes (mesuré :
// 30 notes, le JSON tronqué d'un octet, une écriture → 1 note). Elle ne se réécrit donc qu'une fois son texte copié à
// part (changer, plus bas).
//   { etat: "absente" | "lue" | "illisible" | "indisponible", notes, brut (le texte illisible) }
export function memoire() {
  let brut;
  try { brut = localStorage.getItem(CLE); } catch (e) { return { etat: "indisponible", notes: {} }; }
  if (brut === null || !brut.trim()) return { etat: "absente", notes: {} };
  let notes;
  try { notes = JSON.parse(brut); } catch (e) { return { etat: "illisible", notes: {}, brut }; }
  if (!notes || typeof notes !== "object" || Array.isArray(notes)) return { etat: "illisible", notes: {}, brut };
  return { etat: "lue", notes };
}

// Les notes à LIRE (accueil, établi) : une mémoire illisible n'en a aucune de lisible — memoire() dit pourquoi.
export function toutes() { return memoire().notes; }

export function lire(id) { return toutes()[id] || null; }

// Le bilan de la dernière écriture : réussie ou non, la raison d'un refus (une clé de RAISONS), la copie faite d'une
// mémoire illisible (sa clé, « » sans copie).
let dernier = { ok: true, raison: "", copie: "" };
export function bilan() { return { ...dernier }; }

const plein = (e) => !!e && (e.name === "QuotaExceededError" || e.name === "NS_ERROR_DOM_QUOTA_REACHED" || e.code === 22 || e.code === 1014);

function refus(raison, copie = "") {
  dernier = { ok: false, raison, copie };
  return false;
}

// Les copies de mémoires illisibles gardées dans ce navigateur, les plus anciennes d'abord : { cle, quand (ms), taille }.
export function copies() {
  const out = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const cle = localStorage.key(i);
      if (!cle || !cle.startsWith(PREFIXE_COPIE)) continue;
      const quand = Date.parse(cle.slice(PREFIXE_COPIE.length).replace(/-\d+$/, ""));
      out.push({ cle, quand: Number.isFinite(quand) ? quand : null, taille: (localStorage.getItem(cle) || "").length });
    }
  } catch (e) { /* stockage indisponible : aucune copie lisible */ }
  return out.sort((a, b) => (a.cle < b.cle ? -1 : a.cle > b.cle ? 1 : 0));
}

export function texteCopie(cle) {
  try { return String(cle).startsWith(PREFIXE_COPIE) ? localStorage.getItem(cle) : null; } catch (e) { return null; }
}

// Copier UNE FOIS le texte brut d'une mémoire illisible : une copie identique déjà gardée suffit. Rend la clé de la
// copie, relue et conforme, ou « » si elle n'a pas pu être faite.
function garderCopie(brut) {
  try {
    const deja = copies().find((c) => localStorage.getItem(c.cle) === brut);
    if (deja) return deja.cle;
    const base = PREFIXE_COPIE + new Date().toISOString();
    let cle = base;
    for (let n = 2; localStorage.getItem(cle) !== null; n++) cle = `${base}-${n}`;
    localStorage.setItem(cle, brut);
    return localStorage.getItem(cle) === brut ? cle : "";
  } catch (e) {
    console.error("copie de la mémoire des notes illisible impossible", e);
    return "";
  }
}

// Toute écriture passe ici : relire la mémoire, la changer, la réécrire. Une mémoire illisible n'est réécrite que si
// son texte brut est gardé à part ; sinon RIEN n'est écrit, et l'écriture est refusée (« illisible »).
function changer(f) {
  const m = memoire();
  if (m.etat === "indisponible") return refus("indisponible");
  let copie = "";
  if (m.etat === "illisible") {
    copie = garderCopie(m.brut);
    if (!copie) return refus("illisible");
  }
  f(m.notes);
  try { localStorage.setItem(CLE, JSON.stringify(m.notes)); }
  catch (e) { console.error("enregistrement impossible", e); return refus(plein(e) ? "plein" : "indisponible", copie); }
  dernier = { ok: true, raison: "", copie };
  return true;
}

export function ecrire(note) {
  note.modifie = Date.now();
  return changer((notes) => { notes[note.id] = note; });
}

export function supprimer(id) {
  return changer((notes) => { delete notes[id]; });
}

function ident() {
  return (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2)).slice(0, 12);
}

export function nouvelle({ module, nom = "", chantier = "", dossier = "" }) {
  const n = { id: ident(), module, nom: nom || MODULES[module].nom, chantier, dossier, indice: "A", cree: Date.now(), modifie: Date.now(),
              etat: null, resume: null, vignette: "" };
  return n;
}

export function recentes(n = 6) {
  // une note d'un module que cette version ne connaît pas reste enregistrée, mais ne s'affiche pas
  return Object.values(toutes()).filter((x) => x && Object.hasOwn(MODULES, x.module)).sort((a, b) => b.modifie - a.modifie).slice(0, n);
}

// le fichier .stab : la note entière, lisible et rechargeable
export function exporter(note) {
  const nom = `${(note.chantier ? note.chantier + " - " : "") + (note.nom || "note")}`.replace(/[\\/:*?"<>|]/g, "_");
  return telecharger(`${nom}.stab`, JSON.stringify({ format: "stab", version: 1, note }, null, 1), "application/json");
}

// Un .stab vient d'ailleurs : on n'en garde que des DONNÉES (des textes, l'état que le moteur relira). Le dessin et le
// résumé qu'il porte sont jetés — c'est du HTML qu'un fichier piégé pourrait remplir — et le moteur les refait à
// l'ouverture de la note (verifier_site.js : « .stab piégé »).
const texte = (x, defaut = "") => (typeof x === "string" ? x : defaut).slice(0, 200);

export async function importer(fichier) {
  const d = JSON.parse(await fichier.text());
  const src = d && d.format === "stab" ? d.note : null;
  if (!src || !Object.hasOwn(MODULES, src.module)) throw new Error("ce fichier n'est pas une note .stab du site");
  const n = nouvelle({ module: src.module, nom: texte(src.nom), chantier: texte(src.chantier), dossier: texte(src.dossier) });
  Object.assign(n, { indice: texte(src.indice, "A") || "A", etat: src.etat && typeof src.etat === "object" ? src.etat : null });
  // une note lue mais pas enregistrée ne s'ouvre pas : l'établi l'aurait remplacée, sans un mot, par une note neuve
  if (!ecrire(n)) throw new Error(`la note n'a pas pu être enregistrée dans ce navigateur : ${RAISONS[dernier.raison].long}`);
  return n;
}

export function quand(t) {
  const d = new Date(t), j = new Date();
  const h = d.toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === j.toDateString()) return `aujourd'hui, ${h}`;
  return d.toLocaleDateString("fr-BE", { day: "numeric", month: "short" }) + ", " + h;
}
