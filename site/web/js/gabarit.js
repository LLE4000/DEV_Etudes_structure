// Le GABARIT DU BUREAU (lot 6, 14/11/2026 ; docs/site/MODELES_DIMENSIONNEMENT.md, api_web/gabarit.py) : les options de
// départ de chaque module, fixées par un administrateur sur la page « Paramètres » (js/ecran_parametres.js), que chaque
// note NEUVE reçoit — js/etabli.js les envoie avec sa requête « nouveau », le moteur les écrit et les contrôle.
//
// Ce fichier GARDE ce que la page a fixé — les ÉCARTS à l'usine, tels que le moteur les a rendus — et rien d'autre : ni
// valeur d'usine, ni borne, ni libellé (le moteur les a, R2). Une valeur égale à l'usine n'est pas gardée : elle suit
// l'usine, même quand l'usine change.
//
// Dans ce navigateur, comme l'équipe : `stab.gabarit.v1` = { modules: { <module>: { <option>: valeur } }, par: la
// signature de qui l'a changé en dernier, t: la date } — il deviendra celui du bureau avec les comptes. Une mémoire
// abîmée n'est jamais écrasée sans copie (js/memoire.js). Aucun effet au chargement ; rien ici ne touche au document
// (ces fonctions tournent aussi dans node : tests/test_site_bureau.py).
import * as M from "./memoire.js";
import { signature, signaturePropre } from "./equipe.js";

export const CLE = "stab.gabarit.v1";
const vide = () => ({ modules: {} });
const objet = (x) => !!x && typeof x === "object" && !Array.isArray(x);
const valide = (g) => objet(g) && objet(g.modules);

// le gabarit d'un module : ses écarts à l'usine, ou null (aucun : la note neuve est celle d'usine)
export function lire(module) {
  const v = M.lire(CLE, vide, valide).valeur.modules[module];
  return objet(v) && Object.keys(v).length ? v : null;
}

// qui l'a changé en dernier, et quand : { par (signature propre, ou null), t } — ou null s'il n'a jamais été écrit
export function dernier() {
  const g = M.lire(CLE, vide, valide).valeur;
  return Number.isFinite(g.t) ? { par: signaturePropre(g.par), t: g.t } : null;
}

// garder les ÉCARTS d'un module (ceux que le moteur a rendus) ; vides : le module revient à l'usine. Rend vrai si c'est
// écrit (un stockage refusé ou plein : faux, rien n'est écrit)
export function ecrire(module, ecarts, quand = Date.now()) {
  return M.changer(CLE, vide, valide, (g) => {
    if (objet(ecarts) && Object.keys(ecarts).length) g.modules[module] = ecarts;
    else delete g.modules[module];
    g.par = signature();
    g.t = quand;
  });
}
