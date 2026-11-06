// Les notes du bureau, enregistrées DANS CE NAVIGATEUR (localStorage) en attendant les comptes partagés (Supabase).
// Une note : { id, module, nom, chantier, dossier, indice, cree, modifie, etat (l'état du moteur), resume, vignette }.
const CLE = "stab.notes.v1";

// Les adresses SANS « .html » : Cloudflare Pages sert poutre.html à l'adresse « poutre » et redirige « poutre.html »
// vers elle — une redirection qui n'a pas à porter le « ?note=… » de la note ouverte (outils/serveur_local.py fait de
// même, en jetant la requête : un lien en « .html » perdrait sa note dans le parcours).
export const MODULES = {
  garde_corps: { page: "garde-corps", nom: "Garde-corps", famille: "autres" },
  poutre: { page: "poutre", nom: "Poutres béton armé", famille: "beton" },
};

export function toutes() {
  try { return JSON.parse(localStorage.getItem(CLE) || "{}") || {}; } catch (e) { return {}; }
}

function tout_ecrire(notes) {
  try { localStorage.setItem(CLE, JSON.stringify(notes)); return true; }
  catch (e) { console.error("enregistrement impossible", e); return false; }
}

export function lire(id) { return toutes()[id] || null; }

export function ecrire(note) {
  const notes = toutes();
  note.modifie = Date.now();
  notes[note.id] = note;
  return tout_ecrire(notes);
}

export function supprimer(id) {
  const notes = toutes();
  delete notes[id];
  tout_ecrire(notes);
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
  const blob = new Blob([JSON.stringify({ format: "stab", version: 1, note }, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${(note.chantier ? note.chantier + " - " : "") + (note.nom || "note")}.stab`.replace(/[\\/:*?"<>|]/g, "_");
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
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
  ecrire(n);
  return n;
}

export function quand(t) {
  const d = new Date(t), j = new Date();
  const h = d.toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === j.toDateString()) return `aujourd'hui, ${h}`;
  return d.toLocaleDateString("fr-BE", { day: "numeric", month: "short" }) + ", " + h;
}
