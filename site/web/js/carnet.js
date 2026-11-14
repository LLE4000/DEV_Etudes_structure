// Le CARNET d'un module : ce qui sert d'une note à l'AUTRE (les grues enregistrées du « Générateur SAF », 14/11/2026) n'est
// pas dans une note. L'écran le garde ici, dans ce navigateur, et js/etabli.js le joint à chaque requête d'un module qui en
// a un (body data-carnet, api_web/registre.carnets) ; le moteur le relit, le change à la demande et le rend dans sa réponse
// (api_web/routeur.py), que l'écran enregistre. Ce fichier ne sait rien de ce qu'un carnet contient.
//
// `stab.carnets.v1` = { modules: { <module>: [...] } } — il deviendra celui du bureau avec les comptes, comme le gabarit
// (js/gabarit.js). Une mémoire abîmée n'est jamais écrasée sans copie (js/memoire.js). Aucun effet au chargement ; rien
// ici ne touche au document.
import * as M from "./memoire.js";

export const CLE = "stab.carnets.v1";
const vide = () => ({ modules: {} });
const valide = (c) => !!c && typeof c === "object" && !Array.isArray(c) && !!c.modules && typeof c.modules === "object" && !Array.isArray(c.modules);

// le carnet d'un module : une liste (vide s'il n'a jamais été écrit)
export function lire(module) {
  const v = M.lire(CLE, vide, valide).valeur.modules[module];
  return Array.isArray(v) ? v : [];
}

// garder le carnet d'un module, tel que le moteur l'a rendu ; rend vrai si c'est écrit (ou s'il n'a pas changé)
export function ecrire(module, liste) {
  if (!Array.isArray(liste)) return false;
  if (JSON.stringify(lire(module)) === JSON.stringify(liste)) return true;
  return M.changer(CLE, vide, valide, (c) => {
    if (liste.length) c.modules[module] = liste;
    else delete c.modules[module];
  });
}
