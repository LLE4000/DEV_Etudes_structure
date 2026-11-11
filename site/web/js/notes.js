// Les notes du bureau, enregistrées DANS CE NAVIGATEUR (localStorage) en attendant les comptes partagés (Supabase).
// Une note : { id, module, nom, chantier, dossier, indice, cree, modifie, etat (l'état du moteur), resume, vignette,
//              auteur, par }.
// « chantier » est le nom de son PROJET, « dossier » celui de son SOUS-PROJET (js/projets.js les rassemble) ; « auteur »
// et « par » sont les SIGNATURES ({ id, ini, nom, c } : js/equipe.js) de qui l'a créée et de qui l'a modifiée en dernier.
// Chaque écriture réussie laisse une ligne au journal d'activité (js/journal.js) — qui ne peut jamais la faire échouer.
const CLE = "stab.notes.v1";
// La copie d'une mémoire ILLISIBLE : son texte brut, tel quel, sous cette clé suivie de l'heure de la copie
// (« stab.notes.v1.illisible.2026-11-07T17:08:03.120Z »). Elle n'est jamais réécrite par le site.
export const PREFIXE_COPIE = `${CLE}.illisible.`;

// Les modules du site viennent du registre (api_web/registre.py), écrits par construire.py dans js/modules.js :
// adresses SANS « .html » (Cloudflare sert poutre.html à l'adresse « poutre »), nom, famille.
import { MODULES } from "./modules.js";
import { telecharger } from "./commun.js";
import * as J from "./journal.js";
import { signature, signaturePropre } from "./equipe.js";
export { MODULES };

// Un module qu'un autre a ABSORBÉ (construire.REDIRECTIONS, écrit dans js/modules.js : « vers » ; docs/site/
// DALLES_FUSION.md § 2.4) : une note en mémoire ou un .stab du module « predalle » s'ouvre par la page des dalles, qui
// envoie son état tel quel au module « dalle » (api_web/dalle.normaliser le relit). Une seule table : celle du générateur.
export const moduleServi = (module) => (MODULES[module] && MODULES[module].vers) || module;
// la page qui ouvre une note de ce module (sans « .html »)
export const pageDe = (module) => MODULES[moduleServi(module)].page;

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

// ``date`` : la note a été MODIFIÉE, sa date de modification avance. Sans (date: false), elle est réécrite telle quelle —
// ses caches (résumé, vignette, état relu par le moteur) — et garde sa date : rouvrir une note sans rien y changer ne la
// fait pas remonter dans « Reprendre » (js/etabli.js décide, REFONTE_ETABLI.md, lot S3).
//
// ``journal`` : le geste à noter au journal (« note.importee ») ; sinon il se déduit — la note n'existait pas : créée ;
// son projet ou son sous-projet a changé : rangée ; son nom : renommée ; sinon, si elle est datée : modifiée.
const meme = (a, b) => String(a || "").trim().toLocaleLowerCase("fr") === String(b || "").trim().toLocaleLowerCase("fr");
export function ecrire(note, { date = true, journal = null } = {}) {
  if (date || !Number.isFinite(note.modifie)) note.modifie = Date.now();
  const sig = signature();
  let avant = null;
  const ok = changer((notes) => {
    avant = notes[note.id] && typeof notes[note.id] === "object" ? notes[note.id] : null;
    if (!avant && !note.auteur && sig) note.auteur = sig;       // la première écriture : qui l'a créée
    if (date && sig) note.par = sig;                            // une modification : qui l'a faite
    notes[note.id] = note;
  });
  if (ok) {
    if (journal) J.noter(journal, { note });
    else if (!avant) J.noter("note.creee", { note });
    else if (!meme(avant.chantier, note.chantier) || !meme(avant.dossier, note.dossier)) J.noter("note.classee", { note });
    else if ((avant.nom || "") !== (note.nom || "")) J.noter("note.renommee", { note, de: avant.nom || "" });
    else if (date) J.noter("note.modifiee", { note });
  }
  return ok;
}

export function supprimer(id) {
  let avant = null;
  const ok = changer((notes) => { avant = notes[id] && typeof notes[id] === "object" ? notes[id] : null; delete notes[id]; });
  if (ok && avant) J.noter("note.supprimee", { note: avant });
  return ok;
}

// Changer PLUSIEURS notes d'un coup, sans les dater (renommer un projet, un sous-projet : js/projets.js) : ``f(note)``
// reçoit chaque note lisible et la change en place. La même porte que ecrire et supprimer.
export function reclasser(f) {
  const ok = changer((notes) => { for (const n of Object.values(notes)) if (n && typeof n === "object") f(n); });
  return ok;
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

// le fichier .stab : la note entière, lisible et rechargeable. Il part comme un fichier BINAIRE (octet-stream) et non
// comme du JSON : un navigateur qui voit « application/json » ajoute « .json » au nom (« note.stab.json »).
export function exporter(note) {
  const nom = `${(note.chantier ? note.chantier + " - " : "") + (note.nom || "note")}`.replace(/[\\/:*?"<>|]/g, "_");
  const fait = telecharger(`${nom}.stab`, JSON.stringify({ format: "stab", version: 1, note }, null, 1), "application/octet-stream");
  J.noter("note.exportee", { note });
  return fait;
}

// Un .stab vient d'ailleurs : on n'en garde que des DONNÉES (des textes, l'état que le moteur relira). Le dessin et le
// résumé qu'il porte sont jetés — c'est du HTML qu'un fichier piégé pourrait remplir — et le moteur les refait à
// l'ouverture de la note (verifier_site.js : « .stab piégé »).
const texte = (x, defaut = "") => (typeof x === "string" ? x : defaut).slice(0, 200);

// Un ENREGISTREMENT DE L'ÉCRAN STREAMLIT « Poutre en béton armé » (poutre_ba.json : {"beams", "values"}) s'ouvre aussi :
// il devient une note « Poutres béton armé » dont l'état porte le fichier tel quel ({"streamlit": …}, des données
// seulement : nombres, textes, booléens) ; le moteur le traduit au premier calcul (api_web/poutre_streamlit.py).
const PRIMITIF = (x) => x === null || ["number", "string", "boolean"].includes(typeof x);
function depuisStreamlit(d, nomFichier) {
  if (!d || !Array.isArray(d.beams) || !d.values || typeof d.values !== "object" || Array.isArray(d.values)) return null;
  const values = {};
  for (const [k, v] of Object.entries(d.values)) {
    if (typeof k === "string" && k.length <= 120 && PRIMITIF(v) && (typeof v !== "string" || v.length <= 400)) values[k] = v;
  }
  if (!Object.keys(values).some((k) => /^b\d+_/.test(k))) return null;
  const beams = d.beams.slice(0, 50).map((b) => ({
    id: Number(b && b.id), nom: texte(b && b.nom),
    sections: (Array.isArray(b && b.sections) ? b.sections : []).slice(0, 50).map((s) => ({ id: Number(s && s.id), nom: texte(s && s.nom) })),
  }));
  const n = nouvelle({ module: "poutre", nom: texte(String(nomFichier || "").replace(/\.json$/i, "")) || MODULES.poutre.nom });
  n.etat = { streamlit: { beams, values } };
  return n;
}

export async function importer(fichier) {
  const d = JSON.parse(await fichier.text());
  const st = d && d.format !== "stab" ? depuisStreamlit(d, fichier.name) : null;
  if (st) {
    if (!ecrire(st, { journal: "note.importee" })) throw new Error(`la note n'a pas pu être enregistrée dans ce navigateur : ${RAISONS[dernier.raison].long}`);
    return st;
  }
  const src = d && d.format === "stab" ? d.note : null;
  if (!src || !Object.hasOwn(MODULES, src.module)) throw new Error("ce fichier n'est ni une note .stab du site, ni un enregistrement Streamlit « Poutre en béton armé »");
  const n = nouvelle({ module: src.module, nom: texte(src.nom), chantier: texte(src.chantier), dossier: texte(src.dossier) });
  Object.assign(n, { indice: texte(src.indice, "A") || "A", etat: src.etat && typeof src.etat === "object" ? src.etat : null });
  // qui l'avait créée (sa signature : des initiales, un nom — des textes, relus et bornés) ; celui qui l'importe ici la
  // signe comme dernière main
  const auteur = signaturePropre(src.auteur);
  if (auteur) n.auteur = auteur;
  // une note lue mais pas enregistrée ne s'ouvre pas : l'établi l'aurait remplacée, sans un mot, par une note neuve
  if (!ecrire(n, { journal: "note.importee" })) throw new Error(`la note n'a pas pu être enregistrée dans ce navigateur : ${RAISONS[dernier.raison].long}`);
  return n;
}

export function quand(t) {
  const d = new Date(t), j = new Date();
  const h = d.toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === j.toDateString()) return `aujourd'hui, ${h}`;
  return d.toLocaleDateString("fr-BE", { day: "numeric", month: "short" }) + ", " + h;
}
