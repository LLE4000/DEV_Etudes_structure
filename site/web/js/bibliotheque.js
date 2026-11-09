// La page d'une BIBLIOTHÈQUE (site/web/construire.py, page_bibliotheque ; docs/site/BIBLIOTHEQUES.md § 2.2) : à gauche
// le menu des rubriques, à droite la barre de paramètres, le tableau, la bande de la valeur retenue, les sources. Le
// câblage est celui de js/biblio.js (le même que le panneau « Bibliothèque » d'une note) ; ce script ne fait que créer le
// moteur, relire l'adresse (?rubrique=…&…) et la tenir : une adresse se partage. Aucune note n'est créée ni
// enregistrée : une consultation n'en est pas une.
import { Moteur } from "./moteur.js";
import { cablerRail } from "./commun.js";
import { cablerBibliotheque } from "./biblio.js";

cablerRail();
const BIB = document.body.dataset.bibliotheque;          // « beton »
const etapes = document.getElementById("demarrage-etape");
const moteur = new Moteur((t) => { if (etapes) etapes.textContent = t; }, "");

// l'adresse ouvre une rubrique avec ses paramètres ; sans elle, la première rubrique du menu
const q = new URLSearchParams(location.search);
const bib = cablerBibliotheque({
  racine: document,
  rubrique: q.get("rubrique"),
  demander: (rubrique, params) => moteur.demander({ action: "consulter", bibliotheque: BIB, rubrique, params }),
  apres: (r) => {
    // l'adresse suit : la rubrique, puis chaque paramètre ; le titre de l'onglet aussi
    const u = new URL(location.href);
    u.search = "";
    u.searchParams.set("rubrique", r.rubrique);
    for (const [k, v] of Object.entries(r.params)) if (v !== null && v !== undefined && v !== "") u.searchParams.set(k, String(v));
    history.replaceState(null, "", u);
    document.title = `${r.titre} · ${document.body.dataset.titre} · Stab Méthodes`;
  },
});

// ------------------------------------------------------------------ le démarrage
moteur.pret.then(() => {
  document.getElementById("demarrage").hidden = true;
  return bib.ouvrir(q.get("rubrique"), Object.fromEntries([...q.entries()].filter(([k]) => k !== "rubrique")));
}).catch((e) => {
  const d = document.getElementById("demarrage");
  d.classList.add("erreur");
  if (etapes) etapes.textContent = `Le moteur n'a pas démarré : ${e.message}`;
});
