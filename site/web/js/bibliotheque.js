// La page d'une BIBLIOTHÈQUE (site/web/construire.py, page_bibliotheque ; docs/site/BIBLIOTHEQUES.md § 2.2) : à gauche
// le menu des rubriques, à droite la barre de paramètres, le tableau, la bande de la valeur retenue, les sources. Le
// câblage est celui de js/biblio.js (le même que le panneau « Bibliothèque » d'une note) ; cet écran ne fait que parler au
// moteur, relire l'adresse (?rubrique=…&…) et la tenir : une adresse se partage. Aucune note n'est créée ni enregistrée :
// une consultation n'en est pas une.
//
// Un ÉCRAN MONTABLE (lot S4, REFONTE_ETABLI.md § 3.1, § 3.4) : aucun effet au chargement du fichier ; js/app.js appelle
// ouvrirBibliotheque(hote, ctx). Le moteur est celui de l'application (ctx.moteur), par le client de l'écran ; l'adresse
// et le titre passent par ctx.adresse et ctx.titre ; à la fermeture, la lecture pas encore partie est abandonnée et une
// réponse tardive n'est plus posée (js/biblio.js). Pendant le démarrage : le même bandeau que les établis, sans aperçu,
// le menu et la barre en lecture seule, « Chargement… » du tableau avec aria-busy.
import { cablerBibliotheque } from "./biblio.js";
import { bandeauDemarrage, lectureSeule, TEXTE_DEMARRAGE, ACTIONS_SANS_MOTEUR, texteEchecDemarrage } from "./commun.js";

const ZONES_LECTURE = ["#bib-menu", "#bib-barre", ".top", ".mhead"];
const PANNEAUX_OCCUPES = ["#bib-tableau"];

// ``hote`` : le corps de la page ; ``ctx`` : { params, bibliotheque, moteur, signal, adresse, titre, annoncer… }.
// Rend { fermer } tout de suite.
export function ouvrirBibliotheque(hote, ctx) {
  const { signal } = ctx;
  const BIB = ctx.bibliotheque;              // « beton »
  const TITRE = hote.dataset.titre;          // « Béton armé »
  const client = ctx.moteur.client();
  hote.dataset.ecran = "demarrage";
  const bandeau = bandeauDemarrage({ signal });
  const lecture = lectureSeule(ZONES_LECTURE, PANNEAUX_OCCUPES);
  let pret = false;

  // l'adresse ouvre une rubrique avec ses paramètres ; sans elle, la première rubrique du menu
  const q = ctx.params;
  const bib = cablerBibliotheque({
    racine: document,
    signal,
    rubrique: q.get("rubrique"),
    demander: (rubrique, params) => client.demander({ action: "consulter", bibliotheque: BIB, rubrique, params }),
    apres: (r) => {
      // l'adresse suit : la rubrique, puis chaque paramètre ; le titre de l'onglet aussi
      const u = new URLSearchParams();
      u.set("rubrique", r.rubrique);
      for (const [k, v] of Object.entries(r.params)) if (v !== null && v !== undefined && v !== "") u.set(k, String(v));
      ctx.adresse(`?${u}`);
      ctx.titre(`${r.titre} · ${TITRE} · Stab Méthodes`);
      if (!pret) {
        pret = true;
        bandeau.retirer();
        lecture.lever();
        hote.dataset.ecran = "pret";
        ctx.annoncer("prêt");
        // le défilement d'un retour, quand le tableau est là (comme l'établi : l'application le rendait à 1,5 s)
        requestAnimationFrame(() => { if (!signal.aborted && typeof ctx.rendreDefilement === "function") ctx.rendreDefilement(); });
      }
    },
  });

  // ---------------------------------------------------------------- le démarrage
  (async () => {
    try {
      await ctx.moteur.pret;
      if (signal.aborted) return;
      const r = await bib.ouvrir(q.get("rubrique"), Object.fromEntries([...q.entries()].filter(([k]) => k !== "rubrique")));
      if (signal.aborted || pret) return;
      if (!r) throw new Error("la première consultation a échoué");
    } catch (e) {
      if (signal.aborted) return;
      hote.dataset.ecran = "erreur";
      // le même texte et le même geste que les établis (commun.js) : une erreur nommée de version se dit seule ; ce qui
      // marche sans le moteur redevient actif (aucune action de ce genre dans une bibliothèque aujourd'hui : la règle est
      // la même pour tous les écrans)
      const raison = texteEchecDemarrage(e, ctx.moteur);
      bandeau.erreur(raison);
      lecture.rendre(ACTIONS_SANS_MOTEUR);
      ctx.annoncer(raison);
    }
  })();

  return {
    fermer() {
      // une consultation n'écrit rien : les lectures en attente sont abandonnées, rien n'est gardé
      return { note: null, enregistree: client.fermer().then(() => ({ refus: [] })) };
    },
  };
}
