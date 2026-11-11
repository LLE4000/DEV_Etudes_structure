// Les PROJETS du bureau et leurs SOUS-PROJETS (docs/site/PROJETS_EQUIPE.md § 2 et 3).
//
// Un projet, c'est le nom d'un chantier. Les notes le portent déjà : « chantier » est le nom du projet d'une note,
// « dossier » celui de son sous-projet (js/notes.js) — et c'est ce NOM qui les lie, comparé sans la casse ni les espaces
// en trop (cle). Ce fichier n'y ajoute qu'un petit registre, dans ce navigateur : il fait EXISTER un projet ou un
// sous-projet qui n'a pas encore de note, et garde ce que les notes ne disent pas (créé quand et par qui, archivé).
//   registre : { [cle(nom)]: { nom, cree, par (une signature), archive, sous: [nom, …] } }
//
// modele() rassemble les deux : les projets du registre ET ceux que nomment les notes. Renommer un projet réécrit le
// nom dans toutes ses notes (N.reclasser) : le nom n'a qu'une vérité à la fois.
// Aucun effet au chargement ; pas de document ici (ces fonctions tournent aussi dans node).
import * as M from "./memoire.js";
import * as N from "./notes.js";
import * as J from "./journal.js";
import { signature, signaturePropre } from "./equipe.js";

export const CLE = "stab.projets.v1";
export const NOM_MAX = 80;

// le nom tel qu'on l'écrit (espaces resserrés), et la clé qui le compare
export const nomPropre = (t) => String(t ?? "").replace(/\s+/g, " ").trim().slice(0, NOM_MAX);
export const cle = (t) => nomPropre(t).toLocaleLowerCase("fr");

const vide = () => ({});
const valide = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const ECHEC = "Le projet n'a pas pu être enregistré dans ce navigateur.";

function registre() {
  const out = {};
  for (const [k, x] of Object.entries(M.lire(CLE, vide, valide).valeur)) {
    if (!x || typeof x !== "object") continue;
    const nom = nomPropre(x.nom);
    if (!nom || cle(nom) !== k) continue;
    const sous = [], vus = new Set();
    for (const s of Array.isArray(x.sous) ? x.sous : []) {
      const n = nomPropre(s);
      if (n && !vus.has(cle(n))) { vus.add(cle(n)); sous.push(n); }
    }
    out[k] = { nom, cree: Number(x.cree) || 0, par: signaturePropre(x.par), archive: x.archive === true, sous };
  }
  return out;
}

// Le MODÈLE que les écrans affichent : { projets: [...], sans: [notes sans projet], nbNotes }.
//   un projet : { cle, nom, cree, par, archive, connu (il est au registre), activite (l'heure de sa dernière note, ou sa
//                 création), derniere (l'heure de sa dernière note ; 0 : aucune), nb (ses notes), notes (celles qui n'ont pas de sous-projet), sous: [{ cle, nom, notes }],
//                 auteurs (les signatures de ceux qui y ont écrit, sans doublon) }
// Les projets les plus récemment ACTIFS d'abord ; les sous-projets dans l'ordre où ils ont été créés (ceux que seules
// les notes nomment, ensuite, par ordre alphabétique) ; les notes d'un groupe, les plus récentes d'abord.
// ``notes`` : la table des notes (par défaut, celles de ce navigateur).
export function modele(notes = N.toutes()) {
  const reg = registre();
  const P = new Map();
  const projet = (nom) => {
    const k = cle(nom);
    if (!P.has(k)) {
      const r = reg[k];
      P.set(k, { cle: k, nom: r ? r.nom : nomPropre(nom), cree: r ? r.cree : 0, par: r ? r.par : null, archive: !!(r && r.archive), connu: !!r,
                 activite: r ? r.cree : 0, derniere: 0, nb: 0, notes: [], sous: new Map((r ? r.sous : []).map((s) => [cle(s), { cle: cle(s), nom: s, notes: [] }])), auteurs: new Map() });
    }
    return P.get(k);
  };
  for (const k of Object.keys(reg)) projet(reg[k].nom);
  const sans = [];
  let nbNotes = 0;
  for (const n of Object.values(notes || {})) {
    if (!n || typeof n !== "object" || !Object.hasOwn(N.MODULES, n.module)) continue;
    nbNotes++;
    if (!cle(n.chantier)) { sans.push(n); continue; }
    const p = projet(n.chantier);
    p.nb++;
    p.derniere = Math.max(p.derniere, Number(n.modifie) || 0);
    p.activite = Math.max(p.activite, p.derniere);
    for (const s of [n.par, n.auteur]) { const g = signaturePropre(s); if (g && !p.auteurs.has(g.ini)) p.auteurs.set(g.ini, g); }
    const ks = cle(n.dossier);
    if (!ks) { p.notes.push(n); continue; }
    if (!p.sous.has(ks)) p.sous.set(ks, { cle: ks, nom: nomPropre(n.dossier), notes: [], vu: true });
    p.sous.get(ks).notes.push(n);
  }
  const recent = (a, b) => (Number(b.modifie) || 0) - (Number(a.modifie) || 0);
  const projets = [...P.values()].map((p) => {
    const connus = [...p.sous.values()].filter((s) => !s.vu), vus = [...p.sous.values()].filter((s) => s.vu).sort((a, b) => a.nom.localeCompare(b.nom, "fr"));
    return { ...p, notes: p.notes.sort(recent), sous: [...connus, ...vus].map((s) => ({ cle: s.cle, nom: s.nom, notes: s.notes.sort(recent) })), auteurs: [...p.auteurs.values()] };
  }).sort((a, b) => (b.activite - a.activite) || a.nom.localeCompare(b.nom, "fr"));
  return { projets, sans: sans.sort(recent), nbNotes };
}

export function trouver(nom, m = modele()) { return m.projets.find((p) => p.cle === cle(nom)) || null; }

// ------------------------------------------------------------------ les gestes (chacun rend { ok, … } ou { ok: false, erreur })
// Créer un projet : son nom suffit. Un projet du même nom existe déjà (au registre ou dans les notes) : il est rendu
// (existe: true), rien n'est créé en double.
export function creer(nom) {
  const n = nomPropre(nom);
  if (!n) return { ok: false, erreur: "Le nom du projet manque." };
  const deja = trouver(n);
  if (deja && deja.connu) return { ok: true, existe: true, nom: deja.nom };
  // (un projet que seules des notes nommaient entre au registre tel qu'il est : on ne sait ni quand ni par qui il est né)
  const ok = M.changer(CLE, vide, valide, (v) => { v[cle(n)] = deja ? { nom: deja.nom, cree: 0, par: null, archive: false, sous: [] } : { nom: n, cree: Date.now(), par: signature(), archive: false, sous: [] }; });
  if (!ok) return { ok: false, erreur: ECHEC };
  if (!deja) J.noter("projet.cree", { projet: n });
  return { ok: true, existe: !!deja, nom: deja ? deja.nom : n };
}

// le projet au registre (créé à la demande : un projet que seules des notes nommaient y entre au premier geste)
function assurer(v, p) {
  if (!v[p.cle] || typeof v[p.cle] !== "object") v[p.cle] = { nom: p.nom, cree: p.cree || 0, par: p.par || null, archive: false, sous: [] };
  if (!Array.isArray(v[p.cle].sous)) v[p.cle].sous = [];
  return v[p.cle];
}

// Renommer un projet : le registre ET toutes ses notes. Un autre projet porte déjà ce nom : refusé (on ne fusionne pas
// deux projets par mégarde). Changer seulement la casse (« réalex » → « Réalex ») est permis.
export function renommer(ancien, nouveau) {
  const m = modele(), p = trouver(ancien, m), n = nomPropre(nouveau);
  if (!p) return { ok: false, erreur: "Ce projet n'existe plus." };
  if (!n) return { ok: false, erreur: "Le nom du projet manque." };
  if (n === p.nom) return { ok: true, nom: n, inchange: true };
  if (cle(n) !== p.cle && trouver(n, m)) return { ok: false, erreur: `Un projet « ${trouver(n, m).nom} » existe déjà.` };
  // les notes d'abord (la porte des notes dit ses refus) ; puis le registre
  if (p.nb && !N.reclasser((x) => { if (cle(x.chantier) === p.cle) x.chantier = n; })) return { ok: false, erreur: "Les notes du projet n'ont pas pu être réécrites dans ce navigateur." };
  const ok = M.changer(CLE, vide, valide, (v) => {
    const r = assurer(v, p);
    delete v[p.cle];
    v[cle(n)] = { ...r, nom: n };
  });
  if (!ok) return { ok: false, erreur: ECHEC };
  J.noter("projet.renomme", { projet: n, de: p.nom });
  return { ok: true, nom: n };
}

export function archiver(nom, oui = true) {
  const p = trouver(nom);
  if (!p) return { ok: false, erreur: "Ce projet n'existe plus." };
  const ok = M.changer(CLE, vide, valide, (v) => { assurer(v, p).archive = !!oui; });
  if (!ok) return { ok: false, erreur: ECHEC };
  J.noter(oui ? "projet.archive" : "projet.rouvert", { projet: p.nom });
  return { ok: true };
}

// Supprimer un projet : seulement s'il n'a plus de note (on ne perd jamais une note en supprimant un projet : on les
// déplace ou on les supprime d'abord).
export function supprimer(nom) {
  const p = trouver(nom);
  if (!p) return { ok: true };
  if (p.nb) return { ok: false, erreur: `Ce projet a encore ${p.nb} note${p.nb > 1 ? "s" : ""} : déplacez-les ou supprimez-les d'abord.` };
  const ok = M.changer(CLE, vide, valide, (v) => { delete v[p.cle]; });
  if (!ok) return { ok: false, erreur: ECHEC };
  J.noter("projet.supprime", { projet: p.nom });
  return { ok: true };
}

// ------------------------------------------------------------------ les sous-projets
export function creerSous(projet, nom) {
  const p = trouver(projet), n = nomPropre(nom);
  if (!p) return { ok: false, erreur: "Ce projet n'existe plus." };
  if (!n) return { ok: false, erreur: "Le nom du sous-projet manque." };
  const deja = p.sous.find((s) => s.cle === cle(n));
  if (deja) return { ok: true, existe: true, nom: deja.nom };
  const ok = M.changer(CLE, vide, valide, (v) => { assurer(v, p).sous.push(n); });
  if (!ok) return { ok: false, erreur: ECHEC };
  J.noter("sous.cree", { projet: p.nom, sous: n });
  return { ok: true, nom: n };
}

export function renommerSous(projet, ancien, nouveau) {
  const p = trouver(projet), n = nomPropre(nouveau);
  const s = p && p.sous.find((x) => x.cle === cle(ancien));
  if (!p || !s) return { ok: false, erreur: "Ce sous-projet n'existe plus." };
  if (!n) return { ok: false, erreur: "Le nom du sous-projet manque." };
  if (n === s.nom) return { ok: true, nom: n, inchange: true };
  if (cle(n) !== s.cle && p.sous.some((x) => x.cle === cle(n))) return { ok: false, erreur: `Un sous-projet « ${n} » existe déjà dans ce projet.` };
  if (s.notes.length && !N.reclasser((x) => { if (cle(x.chantier) === p.cle && cle(x.dossier) === s.cle) x.dossier = n; })) {
    return { ok: false, erreur: "Les notes du sous-projet n'ont pas pu être réécrites dans ce navigateur." };
  }
  const ok = M.changer(CLE, vide, valide, (v) => {
    const r = assurer(v, p);
    const i = r.sous.findIndex((x) => cle(x) === s.cle);
    if (i >= 0) r.sous[i] = n; else r.sous.push(n);
  });
  if (!ok) return { ok: false, erreur: ECHEC };
  J.noter("sous.renomme", { projet: p.nom, sous: n, de: s.nom });
  return { ok: true, nom: n };
}

export function supprimerSous(projet, nom) {
  const p = trouver(projet);
  const s = p && p.sous.find((x) => x.cle === cle(nom));
  if (!p || !s) return { ok: true };
  if (s.notes.length) return { ok: false, erreur: `Ce sous-projet a encore ${s.notes.length} note${s.notes.length > 1 ? "s" : ""} : déplacez-les d'abord.` };
  const ok = M.changer(CLE, vide, valide, (v) => { const r = assurer(v, p); r.sous = r.sous.filter((x) => cle(x) !== s.cle); });
  if (!ok) return { ok: false, erreur: ECHEC };
  J.noter("sous.supprime", { projet: p.nom, sous: s.nom });
  return { ok: true };
}

// Ranger une note ailleurs : dans un projet et un sous-projet (« » : aucun). La note n'est pas datée — la ranger n'est
// pas la modifier (elle ne remonte pas dans « Reprendre ») ; le journal le note (« a rangé la note »).
export function deplacer(idNote, projet, sous = "") {
  const n = N.lire(idNote);
  if (!n) return { ok: false, erreur: "Cette note n'est plus dans ce navigateur." };
  const p = nomPropre(projet) ? trouver(projet) : null;
  n.chantier = p ? p.nom : nomPropre(projet);
  const s = p && p.sous.find((x) => x.cle === cle(sous));
  n.dossier = n.chantier ? (s ? s.nom : nomPropre(sous)) : "";
  if (!N.ecrire(n, { date: false })) return { ok: false, erreur: `La note n'a pas pu être rangée : ${(N.RAISONS[N.bilan().raison] || N.RAISONS.indisponible).long}.` };
  return { ok: true };
}
