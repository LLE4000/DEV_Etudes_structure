// Le panneau « Bibliothèque » d'une NOTE (docs/site/BIBLIOTHEQUES.md § 2.4 et § 2.5 ; construire.compact l'écrit pour
// un module dont la famille a une bibliothèque et qui sait la préremplir). Il montre la bibliothèque rendue pour CETTE
// note : le moteur préremplit ses paramètres avec ce que la note sait (action « bibliotheque » : le béton, les Ø, la
// largeur…). Rien du calcul ne change tant qu'on ne clique pas « Utiliser cette valeur » — une saisie ordinaire
// (« modifier »), vers la cible que le moteur a nommée — ; « Inclure dans la note » copie la fiche figée dans l'état
// (action « fiches »), que la note montre à gauche (bloc « Justifications ») et au centre (vue « J1 »). Le câblage de la
// bibliothèque elle-même (rubriques, paramètres, tableau, « Pourquoi ? ») est celui de la page : js/biblio.js.
import { cablerBibliotheque } from "./biblio.js";
import { toast } from "./commun.js";

// module : la clé du module ; note() : la note ouverte ; moteur : le CLIENT du moteur de l'écran (js/moteur.js : ses
// lectures sont abandonnées à la fermeture) ; envoyer(action, extra) : la requête qui met la note à jour (js/etabli.js) ;
// avant() : ce que l'hôte referme d'abord (volets, menu du téléphone) ; signal : l'AbortSignal du montage de l'écran
export function cablerPanneauBibliotheque({ module, note, moteur, envoyer, avant = () => {}, signal = null }) {
  const panneau = document.getElementById("et-bibliotheque");
  if (!panneau) return { ouvrir() {}, fermer() {}, ouvert: () => false };
  let bib = null, retour = null;

  // « Utiliser cette valeur » : vers quel champ, quelle valeur — ou grisé, avec la raison du moteur
  function apres(r) {
    const b = panneau.querySelector("[data-utiliser]");
    if (!b) return;
    const c = (r.cibles || [])[0];
    const possible = !!c && c.possible !== false;
    b.disabled = !possible;
    b.title = !c ? "Aucun champ de cette note ne reçoit cette valeur"
      : (possible ? `Écrire ${c.texte} dans ${c.libelle}${c.actuel ? ` (aujourd'hui ${c.actuel})` : ""}` : c.raison || "");
    const l = b.querySelector("span");
    if (l) l.textContent = possible ? `Utiliser cette valeur · ${c.texte}` : "Utiliser cette valeur";
  }

  function ouvrir(depuis) {
    retour = depuis || null;
    avant();
    panneau.hidden = false;
    document.documentElement.classList.add("volet-ouvert");
    if (!bib) {
      bib = cablerBibliotheque({
        racine: panneau, apres, signal,
        demander: (rubrique, params) => moteur.demander(() => ({ module, action: "bibliotheque", etat: note().etat, rubrique, params })),
      });
    }
    bib.oublier();                // les paramètres repartent de la note : le moteur préremplit
    bib.consulter();
    panneau.querySelector("[data-fermer-bibliotheque]")?.focus();
  }

  function fermer() {
    if (panneau.hidden) return;
    if (bib && bib.pourquoiOuvert()) bib.pourquoi(false);
    panneau.hidden = true;
    document.documentElement.classList.remove("volet-ouvert");
    if (retour && retour.isConnected && !retour.closest("[hidden]")) retour.focus();
  }

  panneau.addEventListener("click", async (ev) => {
    if (ev.target === panneau || ev.target.closest("[data-fermer-bibliotheque]")) { fermer(); return; }
    const r = bib && bib.derniere();
    if (!r) return;
    const u = ev.target.closest("[data-utiliser]");
    if (u) {
      const c = (r.cibles || [])[0];
      if (!c || c.possible === false) return;
      const rep = await envoyer("modifier", { chemin: c.chemin, valeur: c.valeur });
      if (signal && signal.aborted) return;   // l'écran s'est fermé pendant le calcul : rien ne s'écrit dans le suivant
      if (rep && !rep.message) toast(`${c.texte} écrit dans ${c.libelle}`);
      // la note a changé : la bibliothèque se relit depuis la note (la case allumée suit)
      bib.oublier();
      bib.consulter();
      return;
    }
    const inc = ev.target.closest("[data-inclure]");
    if (inc && r.fiche) {
      const fiche = { ...r.fiche, sorte: inc.dataset.inclure, rubrique: r.rubrique, bibliotheque: r.bibliotheque };
      const rep = await envoyer("fiches", { quoi: "ajouter", fiche });
      if (signal && signal.aborted) return;
      if (rep && !rep.message) toast(`« ${fiche.titre} » inclus dans la note (Justifications)`);
    }
  }, { signal });
  // Échap referme le panneau — après « Pourquoi ? », que js/biblio.js referme d'abord
  panneau.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || (bib && bib.pourquoiOuvert())) return;
    ev.stopPropagation();
    fermer();
  }, { signal });
  return { ouvrir, fermer, ouvert: () => !panneau.hidden };
}
