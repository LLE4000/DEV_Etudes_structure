// La mémoire du BUREAU dans ce navigateur (localStorage) : l'équipe, les projets, le journal d'activité
// (docs/site/PROJETS_EQUIPE.md § 3). Les notes ont leur propre porte (js/notes.js), plus stricte ; celle-ci sert trois
// petits registres, avec la même règle : une mémoire ILLISIBLE (abîmée, réécrite par autre chose) n'est jamais écrasée
// sans que son texte brut soit d'abord copié à part — sous sa clé suivie de « .illisible.<heure> ».
//
// Aucun effet au chargement ; rien ici ne touche au document : ces fonctions tournent aussi dans node (tests).

// { etat: "absente" | "lue" | "illisible" | "indisponible", valeur, brut }
// ``valide(v)`` dit si la valeur lue a la forme attendue (un objet, un tableau) : sinon elle est « illisible ».
export function lire(cle, defaut, valide = () => true) {
  let brut;
  try { brut = localStorage.getItem(cle); } catch (e) { return { etat: "indisponible", valeur: defaut() }; }
  if (brut === null || !brut.trim()) return { etat: "absente", valeur: defaut() };
  let v;
  try { v = JSON.parse(brut); } catch (e) { return { etat: "illisible", valeur: defaut(), brut }; }
  if (!valide(v)) return { etat: "illisible", valeur: defaut(), brut };
  return { etat: "lue", valeur: v };
}

// Copier UNE fois le texte brut d'une mémoire illisible ; rend la clé de la copie, ou « » si elle n'a pas pu être faite.
function garderCopie(cle, brut) {
  const prefixe = `${cle}.illisible.`;
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(prefixe) && localStorage.getItem(k) === brut) return k;
    }
    const base = prefixe + new Date().toISOString();
    let k = base;
    for (let n = 2; localStorage.getItem(k) !== null; n++) k = `${base}-${n}`;
    localStorage.setItem(k, brut);
    return localStorage.getItem(k) === brut ? k : "";
  } catch (e) { return ""; }
}

// Relire, changer, réécrire. ``f(valeur)`` change la valeur en place (ou en rend une autre). Rend vrai si c'est écrit.
// Une mémoire illisible n'est réécrite que si son texte est gardé à part ; un stockage refusé ou plein : faux, rien écrit.
export function changer(cle, defaut, valide, f) {
  const m = lire(cle, defaut, valide);
  if (m.etat === "indisponible") return false;
  if (m.etat === "illisible" && !garderCopie(cle, m.brut)) return false;
  const rendu = f(m.valeur);
  const v = rendu === undefined ? m.valeur : rendu;
  try { localStorage.setItem(cle, JSON.stringify(v)); } catch (e) { console.error(`enregistrement impossible (${cle})`, e); return false; }
  return true;
}

// un identifiant court, propre à ce navigateur
export function ident() {
  const c = globalThis.crypto;
  return (c && c.randomUUID ? c.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2)).replace(/-/g, "").slice(0, 12);
}
