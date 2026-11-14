// Ce que TOUTE page porte du bureau, quel que soit son écran : la PASTILLE de la personne du poste (au pied de la barre)
// et la MARQUE des modules vérifiés (cahier des charges § 4) — sur les tuiles de l'accueil, dans le volet des modules, dans
// « Nouvelle note », et au pied du cadre Résultat d'un module. Les gabarits (site/web/construire.py : marque_verifie, rail,
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

// la marque d'un module, au pied de son cadre Résultat : « Module vérifié · LLE · 11/11/2026 » (argentée), « Module vérifié
// deux fois · LLE, MDU · 14/11/2026 » (dorée), « Module à vérifier », « Module à revérifier » (son calcul a changé depuis)
function peindre(el) {
  const cle = el.dataset.verifModule;
  const e = V.etat(cle), d = V.dire(cle, e);
  el.dataset.etat = e.etat;
  el.dataset.niveau = e.niveau || "";
  el.title = d.long;
  const t = el.querySelector(".et-verif-t");
  if (t) t.textContent = e.etat === "verifie" ? `Module ${d.court.toLowerCase()} · ${d.qui}` : e.etat === "change" ? "Module à revérifier" : "Module à vérifier";
}

// Poser la marque au pied du cadre Résultat d'un établi (js/etabli.js, après chaque rendu du résultat : le pied « .pfoot »
// vient du moteur à chaque calcul). Un module qui ne se vérifie pas (masqué, absorbé par un autre) n'en porte pas.
export function poserVerification(resultat, cle) {
  try {
    if (!resultat || !V.modules().includes(cle)) return;
    const pied = resultat.querySelector(".pfoot");
    const gabarit = document.getElementById("et-verif-gabarit");
    if (!pied || !gabarit || pied.querySelector(".et-verif")) return;
    pied.appendChild(gabarit.content.cloneNode(true));
    const el = pied.querySelector(".et-verif");
    if (el) peindre(el);
  } catch (e) { console.error(e); }
}
