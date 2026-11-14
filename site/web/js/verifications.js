// Les MODULES VÉRIFIÉS (cahier des charges § 4 ; docs/site/PROJETS_EQUIPE.md § 8). Un administrateur déclare qu'il a relu
// la MÉTHODE d'un module : la pastille « vérifié » le dit sur l'accueil, dans le volet des modules et dans le module, avec
// son nom et la date — on peut alors s'en servir sans la revérifier. Chaque note reste à relire par son auteur.
//
// La vérification vaut pour UNE VERSION du module : l'empreinte de son calcul le jour où elle est posée (js/versions.js,
// écrit à la construction du site). Le calcul change — une formule, un coefficient, une donnée — : l'empreinte change, la
// pastille tombe d'elle-même, et le module redevient « à revérifier ». Une pastille ne certifie jamais un calcul que
// personne n'a relu.
//
// ARGENT puis OR (demande du propriétaire, 14/11/2026 : « si un deuxième administrateur valide […] argenté puis doré ;
// deux personnes qui valident un module, c'est plus sûr qu'une ») : la marque est ARGENTÉE quand un administrateur a
// vérifié la version du jour, DORÉE quand deux administrateurs DIFFÉRENTS l'ont vérifiée.
//
// Enregistré DANS CE NAVIGATEUR, comme l'équipe (js/equipe.js) : { <module>: [{ par (signature), t, version }, …] } — une
// ligne par administrateur ; l'ancienne forme (une seule ligne, sans liste) se lit encore.
// Aucun effet au chargement ; pas de document ici (ces fonctions tournent aussi dans node : tests/test_site_bureau.py).
import * as M from "./memoire.js";
import * as E from "./equipe.js";
import * as J from "./journal.js";
import { MODULES } from "./modules.js";
import { VERSIONS } from "./versions.js";

export const CLE = "stab.verifications.v1";
const ECHEC = "La vérification n'a pas pu être enregistrée dans ce navigateur.";

const vide = () => ({});
const valide = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const EMPREINTE = /^[0-9a-f]{16}$/;

// les modules qu'on peut vérifier : ceux qui ont un écran de calcul et une version (ni masqués, ni absorbés par un autre)
export function modules() {
  return Object.keys(MODULES).filter((k) => !MODULES[k].masque && !MODULES[k].vers && Object.hasOwn(VERSIONS, k));
}

// ce qui est écrit pour un module, nettoyé (un registre réécrit à la main, une signature piégée) — ou null
function lue(x) {
  if (!x || typeof x !== "object") return null;
  const par = E.signaturePropre(x.par);
  if (!par || !Number.isFinite(x.t) || !EMPREINTE.test(String(x.version || ""))) return null;
  return { par, t: x.t, version: x.version };
}

// QUI a signé : l'identifiant de la personne (ou, dans un registre venu d'ailleurs, ses initiales et son nom)
const qui = (par) => par.id || `${par.ini}|${par.nom}`;

// toutes les lignes bien formées d'un module (l'ancienne forme : une seule), une par personne et par version — la plus récente
function lues(x) {
  const brutes = Array.isArray(x) ? x : [x];
  const vues = new Map();
  for (const v of brutes.map(lue).filter(Boolean)) {
    const k = `${qui(v.par)}@${v.version}`;
    if (!vues.has(k) || vues.get(k).t < v.t) vues.set(k, v);
  }
  return [...vues.values()].sort((a, b) => a.t - b.t);
}

// L'état d'un module : { etat, niveau, par, pars, t, version (celle de la vérification), actuelle (celle du site) }
//   « verifie » : vérifié, et le calcul n'a pas changé depuis — niveau « argent » (un administrateur) ou « or » (deux
//                 administrateurs différents, sur cette même version) ; ``pars`` : leurs signatures, ``par`` : la première ;
//   « change »  : vérifié, mais le calcul a changé depuis — à revérifier (``par`` : la dernière signature) ;
//   « non »     : personne ne l'a vérifié (ou le module n'a pas de version).
export function etat(cle, registre = M.lire(CLE, vide, valide).valeur) {
  const actuelle = Object.hasOwn(VERSIONS, cle) ? VERSIONS[cle] : "";
  const toutes = Object.hasOwn(registre, cle) ? lues(registre[cle]) : [];
  if (!toutes.length || !actuelle) return { etat: "non", niveau: "", par: null, pars: [], t: 0, version: "", actuelle };
  const du_jour = toutes.filter((v) => v.version === actuelle);
  if (!du_jour.length) {
    const v = toutes[toutes.length - 1];
    return { etat: "change", niveau: "", ...v, pars: [v.par], actuelle };
  }
  return { etat: "verifie", niveau: du_jour.length >= 2 ? "or" : "argent", par: du_jour[0].par, pars: du_jour.map((v) => v.par),
           t: du_jour[du_jour.length - 1].t, version: actuelle, actuelle };
}

// La personne a-t-elle déjà vérifié la version du jour ? (la page Équipe ne lui propose plus « Vérifier aussi »)
export function aSigne(e, membre) {
  return !!membre && e.etat === "verifie" && e.pars.some((p) => qui(p) === membre.id);
}

// tous les modules vérifiables, dans l'ordre du registre des modules : [{ cle, nom, famille, ...etat }]
export function tous() {
  const registre = M.lire(CLE, vide, valide).valeur;
  return modules().map((k) => ({ cle: k, nom: MODULES[k].nom, famille: MODULES[k].famille, ...etat(k, registre) }));
}

const jour = (t) => new Date(t).toLocaleDateString("fr-BE", { day: "2-digit", month: "2-digit", year: "numeric" });

// Ce qu'on en DIT : { court (la pastille), long (la phrase entière), qui (« LLE · 11/11/2026 », « LLE, MDU · 14/11/2026 »,
// ou « ») }
export function dire(cle, e = etat(cle)) {
  const nom = Object.hasOwn(MODULES, cle) ? MODULES[cle].nom : cle;
  if (e.etat === "verifie" && e.niveau === "or") {
    const noms = e.pars.map((p) => p.nom);
    return { court: "Vérifié deux fois", qui: `${e.pars.map((p) => p.ini).join(", ")} · ${jour(e.t)}`,
             long: `Module « ${nom} » vérifié par ${noms.slice(0, -1).join(", ")} et ${noms[noms.length - 1]} (dernière vérification le ${jour(e.t)}) : `
                   + `sa méthode a été relue par deux administrateurs. Chaque note reste à relire par son auteur.` };
  }
  if (e.etat === "verifie") {
    return { court: "Vérifié", qui: `${e.par.ini} · ${jour(e.t)}`,
             long: `Module « ${nom} » vérifié par ${e.par.nom} le ${jour(e.t)} : sa méthode a été relue par un administrateur. Chaque note reste à relire par son auteur.` };
  }
  if (e.etat === "change") {
    return { court: "À revérifier", qui: `${e.par.ini} · ${jour(e.t)}`,
             long: `Module « ${nom} » à revérifier : son calcul a changé depuis la vérification de ${e.par.nom} (${jour(e.t)}).` };
  }
  return { court: "À vérifier", qui: "", long: `Module « ${nom} » à vérifier : aucun administrateur n'a encore déclaré sa méthode vérifiée.` };
}

// Déclarer un module vérifié : la personne du poste, si elle est administrateur. La vérification porte la version du jour ;
// elle s'AJOUTE à celle d'un autre administrateur sur la même version (la marque passe à l'or) et remplace la sienne ; les
// vérifications d'une version périmée tombent.
export function verifier(cle) {
  const moi = E.moi();
  if (!modules().includes(cle)) return { ok: false, erreur: "Ce module ne se vérifie pas (il n'a pas d'écran de calcul)." };
  if (!moi) return { ok: false, erreur: "Présentez-vous d'abord (page Équipe) : la vérification porte votre nom." };
  if (!E.estAdmin(moi)) return { ok: false, erreur: "Seul un administrateur peut déclarer un module vérifié." };
  const v = { par: E.signatureDe(moi), t: Date.now(), version: VERSIONS[cle] };
  const garder = (x) => lues(x).filter((w) => w.version === v.version && qui(w.par) !== qui(v.par));
  if (!M.changer(CLE, vide, valide, (r) => { r[cle] = [...garder(r[cle]), v]; })) return { ok: false, erreur: ECHEC };
  J.noter("module.verifie", { module: MODULES[cle].nom });
  return { ok: true, ...v, niveau: etat(cle).niveau };
}

// Retirer la vérification d'un module (un administrateur) : il redevient « à vérifier ».
export function retirer(cle) {
  const moi = E.moi();
  if (!E.estAdmin(moi)) return { ok: false, erreur: "Seul un administrateur peut retirer une vérification." };
  const avant = etat(cle);
  if (avant.etat === "non") return { ok: true, inchange: true };
  if (!M.changer(CLE, vide, valide, (r) => { delete r[cle]; })) return { ok: false, erreur: ECHEC };
  J.noter("module.retire", { module: Object.hasOwn(MODULES, cle) ? MODULES[cle].nom : cle });
  return { ok: true };
}
