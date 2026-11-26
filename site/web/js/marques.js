// Ce que TOUTE page porte du bureau, quel que soit son écran : la PASTILLE de la personne du poste (au pied de la barre)
// et la MARQUE des modules vérifiés (cahier des charges § 4) — sur les tuiles de l'accueil, dans le volet des modules, dans
// « Nouvelle note », et au pied du Résultat d'un module vérifié. Les gabarits (site/web/construire.py : marque_verifie, rail,
// page_module) écrivent ces marques CACHÉES ou vides ; ce fichier les montre d'après ce navigateur (js/equipe.js,
// js/verifications.js).
//
// L'APPLICATION (js/app.js) appelle marquer() après chaque montage ; un écran qui change ces données (se présenter,
// vérifier un module) rappelle marquer(). Ces marques se LISENT : déclarer un module vérifié se fait sur la page Équipe
// (js/ecran_equipe.js). Rien ici ne doit pouvoir faire échouer un montage : tout est gardé.
import * as E from "./equipe.js";
import * as V from "./verifications.js";

// Montrer, dans ``racine``, la pastille du poste et les marques des modules vérifiés.
export function marquer(racine = typeof document === "undefined" ? null : document) {
  try {
    if (!racine || typeof racine.querySelectorAll !== "function") return;
    const moi = E.moi();
    for (const el of racine.querySelectorAll("[data-moi-rail]")) {
      el.innerHTML = E.pastille(moi);
      el.dataset.tip = moi ? `${moi.nom} · mon profil et l'équipe` : "Se présenter · l'équipe";
    }
    const marques = racine.querySelectorAll("[data-verifie]");
    if (marques.length) {
      const etats = new Map(V.tous().map((m) => [m.cle, m]));
      for (const el of marques) {
        const e = etats.get(el.dataset.verifie);
        const vu = !!e && e.etat === "verifie";
        el.hidden = !vu;
        // (argent : un administrateur ; or : deux administrateurs différents — js/verifications.js, 14/11/2026)
        if (vu) { const d = V.dire(e.cle, e); el.title = `Vérifié · ${d.qui}`; el.setAttribute("aria-label", d.long); el.dataset.niveau = e.niveau; }
        else { el.removeAttribute("title"); el.removeAttribute("aria-label"); delete el.dataset.niveau; }
      }
    }
    for (const el of racine.querySelectorAll(".et-verif[data-verif-module]")) peindre(el);
  } catch (e) { console.error(e); }
}

// la marque d'un module, au pied de son Résultat : « Module vérifié · LLE · 11/11/2026 » (argentée), « Module vérifié deux
// fois · LLE, MDU · 14/11/2026 » (dorée), « Module à revérifier » (son calcul a changé depuis). Un module que personne n'a
// vérifié n'en porte PAS (``poserVerification``) : « Module à vérifier » est parti avec le cadre du pied (25/11/2026).
function peindre(el) {
  const cle = el.dataset.verifModule;
  const e = V.etat(cle), d = V.dire(cle, e);
  el.dataset.etat = e.etat;
  el.dataset.niveau = e.niveau || "";
  el.title = d.long;
  const t = el.querySelector(".et-verif-t");
  if (t) t.textContent = e.etat === "verifie" ? `Module ${d.court.toLowerCase()} · ${d.qui}` : e.etat === "change" ? "Module à revérifier" : "Module à vérifier";
}

// Poser la marque au pied du Résultat d'un établi (js/etabli.js, après chaque rendu du résultat, qui la remporte) : la
// DERNIÈRE ligne du cadre, sous le corps qui défile. Elle ne se pose que pour un module VÉRIFIÉ, ou vérifié puis changé
// (« à revérifier ») : il n'y a plus de cadre au pied du résultat (le propriétaire, 25/11/2026 : « en bas à droite … il y
// a un code couleur. J'aimerais qu'on vire ce cadre-là, il ne sert à rien. Et ça peut être pour tous les modules »), et
// « Module à vérifier » ne se dit plus — sans marque, un module n'est pas vérifié, comme sur sa tuile de l'accueil. Un
// module qui ne se vérifie pas (masqué, absorbé par un autre) n'en porte jamais.
export const ETATS_MARQUES = ["verifie", "change"];
export function poserVerification(resultat, cle) {
  try {
    if (!resultat || !V.modules().includes(cle)) return;
    for (const vieux of resultat.querySelectorAll(":scope > .et-verif")) vieux.remove();
    const gabarit = document.getElementById("et-verif-gabarit");
    if (!gabarit || !ETATS_MARQUES.includes(V.etat(cle).etat)) return;
    resultat.appendChild(gabarit.content.cloneNode(true));
    const el = resultat.querySelector(":scope > .et-verif");
    if (el) peindre(el);
  } catch (e) { console.error(e); }
}
