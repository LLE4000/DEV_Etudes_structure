// Les MODULES VÉRIFIÉS (cahier des charges § 4 ; docs/site/PROJETS_EQUIPE.md § 8). Un administrateur déclare qu'il a relu
// la MÉTHODE d'un module : la pastille « vérifié » le dit sur l'accueil, dans le volet des modules et dans le module, avec
// son nom et la date — on peut alors s'en servir sans la revérifier. Chaque note reste à relire par son auteur.
//
// La vérification vaut pour UNE VERSION du module : l'empreinte de son calcul le jour où elle est posée (js/versions.js,
// écrit à la construction du site). Le calcul change — une formule, un coefficient, une donnée — : l'empreinte change, la
// pastille tombe d'elle-même, et le module redevient « à revérifier ». Une pastille ne certifie jamais un calcul que
// personne n'a relu.
//
// Enregistré DANS CE NAVIGATEUR, comme l'équipe (js/equipe.js) : { <module>: { par (signature), t, version } }.
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

// L'état d'un module : { etat, par, t, version (celle de la vérification), actuelle (celle du site) }
//   « verifie » : vérifié, et le calcul n'a pas changé depuis ;
//   « change »  : vérifié, mais le calcul a changé depuis — à revérifier ;
//   « non »     : personne ne l'a vérifié (ou le module n'a pas de version).
export function etat(cle, registre = M.lire(CLE, vide, valide).valeur) {
  const actuelle = Object.hasOwn(VERSIONS, cle) ? VERSIONS[cle] : "";
  const v = Object.hasOwn(registre, cle) ? lue(registre[cle]) : null;
  if (!v || !actuelle) return { etat: "non", par: null, t: 0, version: "", actuelle };
  return { etat: v.version === actuelle ? "verifie" : "change", ...v, actuelle };
}

// tous les modules vérifiables, dans l'ordre du registre des modules : [{ cle, nom, famille, ...etat }]
export function tous() {
  const registre = M.lire(CLE, vide, valide).valeur;
  return modules().map((k) => ({ cle: k, nom: MODULES[k].nom, famille: MODULES[k].famille, ...etat(k, registre) }));
}

const jour = (t) => new Date(t).toLocaleDateString("fr-BE", { day: "2-digit", month: "2-digit", year: "numeric" });

// Ce qu'on en DIT : { court (la pastille), long (la phrase entière), qui (« LLE · 11/11/2026 », ou « ») }
export function dire(cle, e = etat(cle)) {
  const nom = Object.hasOwn(MODULES, cle) ? MODULES[cle].nom : cle;
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

// Déclarer un module vérifié : la personne du poste, si elle est administrateur. La vérification porte la version du jour.
export function verifier(cle) {
  const moi = E.moi();
  if (!modules().includes(cle)) return { ok: false, erreur: "Ce module ne se vérifie pas (il n'a pas d'écran de calcul)." };
  if (!moi) return { ok: false, erreur: "Présentez-vous d'abord (page Équipe) : la vérification porte votre nom." };
  if (!E.estAdmin(moi)) return { ok: false, erreur: "Seul un administrateur peut déclarer un module vérifié." };
  const v = { par: E.signatureDe(moi), t: Date.now(), version: VERSIONS[cle] };
  if (!M.changer(CLE, vide, valide, (r) => { r[cle] = v; })) return { ok: false, erreur: ECHEC };
  J.noter("module.verifie", { module: MODULES[cle].nom });
  return { ok: true, ...v };
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
