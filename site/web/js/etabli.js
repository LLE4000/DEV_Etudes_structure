// L'établi d'un module (garde-corps.html, poutre.html…) : il envoie au moteur ce que l'utilisateur saisit et affiche les
// fragments qu'il reçoit (données, résultat, fenêtres de calcul, sélecteurs, dessins). Il ne calcule rien.
//
// Une seule page pour les trois formats : les fonctions sont les mêmes au PC, à la tablette et au téléphone, seule la
// disposition change (css/site.css). Au téléphone, les onglets Données · Dessin · Résultat choisissent le cadre visible,
// les actions de la barre du haut passent dans un menu, et les volets montent du bas.
//
// Un ÉCRAN MONTABLE (lot S4, REFONTE_ETABLI.md § 3.1, C1, C2) : ce fichier n'a AUCUN effet à son chargement ; js/app.js
// appelle ouvrirEtabli(hote, ctx) quand le corps d'un établi est en place, et fermer() quand il le quitte.
//   - Tout écouteur posé sur le document, la fenêtre ou une requête média l'est avec le signal du montage (ctx.signal) ;
//     les observateurs se déconnectent et les minuteries s'arrêtent à son abandon.
//   - Toute suite d'un « await » s'arrête si l'écran est fermé (fini()) : le corps est déjà celui de l'écran suivant.
//   - L'écran ne touche ni à l'historique, ni à l'adresse, ni au titre : ctx.adresse, ctx.titre, ctx.naviguer.
//   - Le moteur est celui de l'application, partagé ; l'écran lui parle par SON client (ctx.moteur.client()).
//   - La fermeture (fermer) ne bloque pas et ne jette rien : les LECTURES pas encore parties sont abandonnées, les
//     MODIFICATIONS gardées partent et leurs réponses s'appliquent à la NOTE seule (jamais au DOM) ; un refus est remis à
//     l'application (enregistree se résout avec { refus }), qui le dit dans l'écran suivant.
//   - Le DÉMARRAGE est visible (plus de voile, § 3.4) : le bandeau .et-demarre, les contrôles en lecture seule, l'APERÇU
//     d'un calcul neuf (moteur/apercus/<clé>.json) ou le dernier enregistrement d'une note rouverte, montré neutre ;
//     body[data-ecran] dit l'état : demarrage, apercu, pret, erreur.
import * as N from "./notes.js";
import { toast, esc, telecharger, garderGrec, texteTaux, lireManifeste, bandeauDemarrage, lectureSeule, TEXTE_DEMARRAGE, SUITE_LECTURE, SUITE_APERCU,
  ACTIONS_SANS_MOTEUR, texteEchecDemarrage } from "./commun.js";
import { cablerToile } from "./toile.js";
import { recupererMeteo } from "./meteo.js";
import { SONDE, marquer, marquerUneFois } from "./sonde.js";
import { cablerCompact } from "./compact.js";
import { cablerRangement } from "./rangement.js";
import { cablerPanneauBibliotheque } from "./biblio_note.js";

// les actions qui MODIFIENT la note (une note provisoire s'écrit à la première) ; « nouveau », « calculer »,
// « selecteur », « bibliotheque » ne font que la lire
const MODIFIENT = new Set(["modifier", "choisir", "ajouter", "dupliquer", "supprimer", "activer", "fiches"]);
// … et parmi elles, celles qui changent sa DATE (« modifiée le … », l'ordre de « Reprendre ») — toutes, sauf ouvrir un
// autre élément de la note : c'est une lecture (REFONTE_ETABLI.md, lot S3). Rouvrir une note sans rien y changer ne
// change pas sa date : le premier calcul n'en réécrit que les caches (résumé, vignette).
const DATENT = new Set([...MODIFIENT].filter((a) => a !== "activer"));
// SUITE : ce que l'utilisateur peut faire, selon la raison du refus (notes.RAISONS dit la raison)
const SUITE = {
  plein: "Cette note n'est gardée nulle part : exportez-la en fichier .stab pour ne pas la perdre.",
  illisible: "Sa copie de sauvegarde n'a pas pu être faite : rien n'y est écrit, pour ne pas effacer ce qu'elle contient. "
    + "Exportez cette note en fichier .stab pour la garder.",
  indisponible: "Cette note n'est gardée nulle part : exportez-la en fichier .stab pour ne pas la perdre.",
};
const HIST_MAX = 50;
const memoire = { lire(k, d) { try { return JSON.parse(sessionStorage.getItem(k) || "null") ?? d; } catch (e) { return d; } },
  ecrire(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* stockage bloqué */ } } };
// l'heure seule aujourd'hui, la date sinon (l'indicateur est court ; la date entière est au survol)
const heure = (t) => (new Date(t).toDateString() === new Date().toDateString()
  ? new Date(t).toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit" })
  : new Date(t).toLocaleDateString("fr-BE", { day: "numeric", month: "short" }));
const estDessin = (v) => /^\s*<svg/.test(v.svg || "");
// pendant le démarrage, les CONTRÔLES de ces zones sont en lecture seule (inert) — pas les panneaux, qui restent lisibles
// et défilables (aria-busy) ; ni les onglets du téléphone, ni les vues du dessin, ni la navigation
const ZONES_LECTURE = ["#donnees", "#volets", ".top", ".mtop", ".mbar", "#mmenu", ".panel.left .phead .et-elt"];
const PANNEAUX_OCCUPES = ["#donnees", "#resultat", "#stage"];
// la mention de l'APERÇU, en tête du résultat (aux trois formats : revue S4, ecrans-apercu-mention — au PC et à la
// tablette, la barre du haut n'a pas la place de la suite du bandeau)
export const MENTION_APERCU = "Aperçu du calcul par défaut, en lecture seule tant que le moteur démarre";
// les onglets du téléphone (l'onglet ouvert est gardé dans l'entrée d'historique de l'écran : un retour le rend)
const ONGLETS = ["donnees", "dessin", "resultat"];

// ``hote`` : le corps de la page ; ``ctx`` : ce que l'application donne (js/app.js) — { jeton, params, module, moteur,
// signal, adresse(url, {remplacer}), titre(t), naviguer(url, opts), attendreEnregistrement(id), annoncer(t) }.
// Rend { fermer } TOUT DE SUITE ; fermer() rend { note, enregistree }.
export function ouvrirEtabli(hote, ctx) {
  const { signal } = ctx;
  let ferme = false;
  const fini = () => ferme || signal.aborted;
  const image = (f) => requestAnimationFrame(() => { if (!fini()) f(); });
  const MODULE = ctx.module;
  const $ = (s, r = document) => r.querySelector(s);
  marquer("etabli_debut");   // la sonde du démarrage (js/sonde.js) : rien sans « ?sonde=1 »

  // ---------------------------------------------------------------- le démarrage, visible tout de suite
  hote.dataset.ecran = "demarrage";
  // (« aperçu » seulement quand l'aperçu est peint : une note rouverte n'en est jamais un, C11)
  const bandeau = bandeauDemarrage({ texte: `${TEXTE_DEMARRAGE} · ${SUITE_LECTURE}`, signal });
  const lecture = lectureSeule(ZONES_LECTURE, PANNEAUX_OCCUPES);
  const client = ctx.moteur.client();
  const refus = [];          // les refus du moteur arrivés après la fermeture : l'application les dit dans l'écran suivant
  let modifsEnVol = 0;       // les modifications envoyées (en file ou parties) dont la réponse n'est pas arrivée
  let note = null;
  const demandee = ctx.params.get("note");   // l'adresse nomme une note (un lien, la carte « Reprendre », un import)

  let fermeture = null;
  function fermer() {
    if (fermeture) return fermeture;
    ferme = true;
    // les lectures pas encore parties sont abandonnées ; les modifications gardées partent et s'appliquent à la note.
    // Sans modification gardée, rien ne s'enregistre après la fermeture : la note n'est pas nommée (l'accueil, monté dans
    // la même tâche, la montrerait « enregistrement en cours » pour rien)
    const garde = modifsEnVol > 0 && !!note;
    const vide = client.fermer();
    const enregistree = garde ? vide.then(() => new Promise((ok) => setTimeout(ok, 0))).then(() => ({ refus: refus.slice() }))
      : Promise.resolve({ refus: [] });
    fermeture = { note: garde ? note.id : null, enregistree };
    return fermeture;
  }

  // la note demandée est peut-être encore en train de s'écrire (l'écran qui l'avait ouverte vient de se fermer) : on
  // l'attend avant de la lire, sinon le premier calcul partirait de l'état d'avant la dernière saisie
  (async () => {
    if (demandee) await ctx.attendreEnregistrement(demandee);
    if (fini()) return;
    monter();
  })();
  return { fermer };

  function monter() {
    const bench = $(".bench");
    const telephone = matchMedia("(max-width: 700px)");
    // l'établi COMPACT (les trois modules béton armé, construire.ETABLI_COMPACT) : l'arborescence de la note à gauche, le
    // dessin au centre, un bloc de résultats par section à droite (js/compact.js) ; les autres modules gardent l'établi d'avant
    const COMPACT = hote.dataset.etabli === "compact";
    // (colonne de gauche compacte, construire.ARBRE_COMPACT) l'emplacement de la ligne de l'élément ouvert, dans l'en-tête
    // du panneau des données
    const ELT = $(".panel.left .phead .et-elt");
    // le module fait sa note PDF dans le navigateur (construire.NOTE_PDF) ; sinon « Note PDF » dit qu'elle arrive bientôt
    const NOTE_PDF = hote.dataset.notePdf === "1";

    // -------------------------------------------------------------- la note
    note = demandee && N.lire(demandee);
    // une note d'un AUTRE module (un lien recopié d'une autre page) : elle s'ouvre par la page de son module — sans quoi
    // celle-ci ouvrait, sans un mot, un calcul neuf à sa place (l'application y va, en remplaçant cette adresse)
    const ailleurs = !!note && Object.hasOwn(N.MODULES, note.module) && N.moduleServi(note.module) !== MODULE;
    if (ailleurs) { ctx.naviguer(`${N.pageDe(note.module)}?note=${encodeURIComponent(note.id)}`, { remplacer: true }); return; }
    // une note déjà enregistrée, rouverte — ou une note d'un module que celui-ci a ABSORBÉ (notes.moduleServi : « predalle »
    // est servi par la page des dalles) : son état part tel quel au moteur de ce module, qui sait le relire
    const relue = !!note && N.moduleServi(note.module) === MODULE;
    // la note demandée n'est PAS dans ce navigateur (supprimée, enregistrée sur un autre poste, lien partagé) : le message le
    // dit, avec « Importer .stab » ; RIEN n'est créé à sa place — le calcul ouvert est provisoire, comme depuis une carte
    const absente = !!demandee && !note;
    // PROVISOIRE (décision D1, tous les modules depuis le 10/11/2026 ; c'était la poutre et la dalle) : le calcul ouvert
    // depuis sa carte, sans nom ni chantier demandés. La note n'existe qu'en mémoire : RIEN n'est écrit tant que
    // l'utilisateur ne modifie rien (ouvrir pour regarder ne remplit pas la liste des notes). La première saisie, un renommage
    // ou « Enregistrer comme note… » l'écrivent, et l'adresse prend alors ?note=. (« Nouvelle note » de l'accueil, ou une
    // adresse ?nom=…&chantier=…, nomme la note d'abord : elle n'est pas provisoire.)
    const q = ctx.params;
    let provisoire = false;
    // la note EST dans le stockage (rouverte, ou écrite depuis par cet écran) : si elle n'y est plus quand une modification
    // gardée revient, c'est qu'elle a été supprimée — on ne la réécrit pas (revue S4, etat-03)
    let ecrite = relue;
    if (!relue) {
      note = N.nouvelle({ module: MODULE, nom: q.get("nom") || "", chantier: q.get("chantier") || "", dossier: q.get("dossier") || "" });
      provisoire = !["nom", "chantier", "dossier"].some((k) => q.get(k));
      if (!provisoire) ctx.adresse(`?note=${note.id}`);
    }
    // le nom de la note : le champ de la barre du haut (PC, tablette) et celui de l'en-tête du téléphone
    const noms = [...document.querySelectorAll("[data-nom-note]")];
    // ce que la note EST, en une phrase (décision D9 : l'indice réel, à la place de l'étiquette figée « Brouillon ») — une
    // source pour l'étiquette de la barre du haut (construire.etiquette_note) et la 2ᵉ ligne de l'en-tête du téléphone :
    // « Indice B » ; « Non enregistrée » tant que la note est provisoire (au téléphone, l'indicateur d'enregistrement qui
    // précède le dit déjà : « Pas encore enregistré » — la 2ᵉ ligne ne le répète pas)
    const identite = () => (provisoire ? "Non enregistrée" : `Indice ${note.indice || "A"}`);
    function afficherEntete() {
      noms.forEach((n) => { if (n !== document.activeElement) n.value = note.nom || ""; });
      ctx.titre(`${note.nom || N.MODULES[MODULE].nom} · Stab Méthodes`);
      const fil = [note.chantier, note.dossier].filter(Boolean).join(" › ");
      $("#fil-chantier").textContent = fil || "Sans chantier";
      $("#fil-m").textContent = [provisoire ? "" : identite(), fil || "Sans chantier", N.MODULES[MODULE].nom].filter(Boolean).join(" · ");
      const ind = $("#et-ind");
      ind.textContent = identite();
      ind.dataset.provisoire = provisoire ? "1" : "";
      ind.title = provisoire ? "Ce calcul n'est pas encore une note : rien n'est écrit tant que vous ne modifiez rien (ou « Enregistrer comme note… »)"
        : `Indice ${note.indice || "A"} de la note (celui de la note PDF et du cartouche)`;
    }
    afficherEntete();
    noms.forEach((n) => n.addEventListener("change", () => { note.nom = n.value.trim() || N.MODULES[MODULE].nom; enregistrer(); apresEntete(); }, { signal }));
    // renommée ou classée : l'en-tête, la liste (établi compact) et le cartouche du dessin le disent tout de suite
    function apresEntete() {
      afficherEntete();
      if (compact) compact.majNote();
      document.querySelectorAll('#stage .cart [data-cart="nom"]').forEach((b) => { b.textContent = note.nom || ""; });
      document.querySelectorAll('#stage .cart [data-cart="chantier"]').forEach((b) => { b.textContent = note.chantier || "—"; });
    }
    noms.forEach((n) => n.addEventListener("keydown", (ev) => { if (ev.key === "Enter") n.blur(); }, { signal }));

    // -------------------------------------------------------------- l'enregistrement : visible aux trois formats
    // Chaque indicateur .et-enr (la barre du haut au PC et à la tablette, la 2ᵉ ligne de l'en-tête au téléphone : le même
    // composant, construire.enregistrement) dit l'état de la DERNIÈRE écriture. Un échec y reste, en rouge, jusqu'à la
    // prochaine écriture réussie, et ouvre une fois le message #et-alerte, qui offre « Exporter .stab ». Aucune règle de la
    // feuille ne les cache : l'ancien texte (.saved .txt) l'était sous 1 360 px par la charte, et l'échec y était muet
    // (verifier_site.js, filtre « enregistrement »).
    // Le texte d'un indicateur : l'ÉTAT (.et-enr-etat), puis, en échec, la RAISON (.et-enr-raison, « : stockage plein »). Au
    // téléphone, la 2ᵉ ligne n'écrit que l'état (la raison n'y tient pas à côté de la pastille de la note) : l'entrée du
    // menu ⋯ (revoir) dit l'échec en entier et rouvre le message.
    const indicateurs = [...document.querySelectorAll(".et-enr")];
    const alerte = $("#et-alerte");
    const revoir = $("[data-revoir-alerte]");
    let alerteDite = "";   // la raison que le message a déjà dite : il ne revient pas à chaque saisie qui échoue
    let alerteEchec = "";  // le texte du message du dernier échec : « revoir » le rouvre
    function direEnregistrement(etat, mot, texte, raison, titre) {
      indicateurs.forEach((e) => {
        e.dataset.enr = etat;
        e.querySelector(".et-enr-mot").textContent = mot;
        e.querySelector(".et-enr-etat").textContent = texte;
        e.querySelector(".et-enr-raison").textContent = raison;
        e.title = titre;
      });
      revoir.hidden = etat !== "echec";
      revoir.querySelector(".et-mi-enr-txt").textContent = etat === "echec" ? texte + raison : "";
    }
    // sorte : « echec » (avec « Exporter .stab »), « info », « absente » (une note demandée qui n'est pas dans ce navigateur :
    // avec « Importer .stab ») ; ``html`` : le texte est un fragment du MOTEUR (un avis d'ouverture, ses indices <sub>), posé
    // comme ses autres fragments
    function direAlerte(sorte, texte, html = false) {
      alerte.classList.toggle("et-info", sorte !== "echec");
      alerte.querySelector(".et-alerte-txt")[html ? "innerHTML" : "textContent"] = texte;
      alerte.querySelector(".et-alerte-stab").hidden = sorte !== "echec";
      alerte.querySelector(".et-alerte-importer").hidden = sorte !== "absente";
      alerte.hidden = false;
    }
    // écrire la note ; ``date`` : sa date de modification avance (une vraie modification) — sinon seuls ses caches (résumé,
    // vignette, l'état relu par le moteur) sont réécrits, à la date d'avant
    function enregistrer({ date = true } = {}) {
      if (provisoire) { provisoire = false; ctx.adresse(`?note=${note.id}`); rangement.libeller(); afficherEntete(); }
      const ok = N.ecrire(note, { date }), b = N.bilan();
      if (ok) {
        ecrite = true;
        direEnregistrement("ok", "Enregistré · ", heure(note.modifie), "", `Enregistré dans ce navigateur · ${N.quand(note.modifie)}`);
        if (alerteDite) { alerteDite = ""; alerteEchec = ""; alerte.hidden = true; }
        // la mémoire des notes était illisible : son texte vient d'être gardé à part (notes.js), on le dit une fois
        if (b.copie) {
          direAlerte("info", "La mémoire des notes de ce navigateur était illisible : son texte est gardé à part, tel quel (l'accueil permet "
            + "de le télécharger). Cette note est enregistrée dans une mémoire neuve.");
        }
        return true;
      }
      const r = N.RAISONS[b.raison] || N.RAISONS.indisponible;
      direEnregistrement("echec", "", "Non enregistré", ` : ${r.court}`, `Non enregistré : ${r.long}`);
      alerteEchec = `Non enregistré : ${r.long}. ${SUITE[b.raison] || SUITE.indisponible}`;
      if (alerteDite !== b.raison) {
        alerteDite = b.raison;
        direAlerte("echec", alerteEchec);
      }
      return false;
    }
    if (relue) direEnregistrement("ok", "Enregistré · ", heure(note.modifie), "", `Enregistré dans ce navigateur · ${N.quand(note.modifie)}`);
    else if (provisoire) {
      direEnregistrement("neuf", "", "Pas encore enregistré", "",
        "Rien n'est écrit tant que vous ne modifiez rien : la note s'enregistre dans ce navigateur à la première saisie");
    } else direEnregistrement("neuf", "", "Pas encore enregistré", "", "La note s'enregistre dans ce navigateur dès le premier calcul");
    if (absente) {
      direAlerte("absente", "Cette note n'est pas dans ce navigateur (supprimée, ou enregistrée sur un autre poste) : importez son fichier .stab "
        + "pour l'ouvrir. Le calcul ouvert ici est neuf ; rien n'est enregistré tant que vous ne le modifiez pas.");
    }
    alerte.addEventListener("click", (ev) => { if (ev.target.closest("[data-fermer-alerte]")) alerte.hidden = true; }, { signal });
    // le menu ⋯ (téléphone) : l'échec en entier ; le toucher rouvre son message (le menu se referme seul, comme pour ses
    // autres entrées)
    revoir.addEventListener("click", () => { if (alerteEchec) direAlerte("echec", alerteEchec); }, { signal });
    // « Enregistrer comme note… » (toute page d'établi) : nommer, ranger, écrire la note provisoire
    const rangement = cablerRangement({ note: () => note, provisoire: () => provisoire,
      apres: () => { const ok = enregistrer(); apresEntete(); return ok; }, signal });

    // -------------------------------------------------------------- le moteur
    let enCours = 0;
    let meteoEnCours = null;   // le chemin du bouton météo dont la récupération court (une à la fois)
    function occupe(plus) {
      if (fini()) return;
      enCours += plus;
      const on = enCours > 0 || !!meteoEnCours;
      document.querySelectorAll(".calcule").forEach((i) => {
        i.classList.toggle("on", on);
        i.textContent = on && !i.closest(".mtop") ? (meteoEnCours ? "Récupération de la météo…" : "Calcul…") : "";
      });
    }

    // ``modification`` : la requête rend un état que l'utilisateur a voulu (annuler, rétablir), quelle que soit l'action.
    // Une requête qui MODIFIE la note est gardée si l'écran se ferme avant sa réponse : la réponse s'applique alors à la
    // note seule (garderApresFermeture) ; une lecture est abandonnée (EcranFerme, avalée sans message).
    async function envoyer(action, extra = {}, { sansHistorique = false, modification = false } = {}) {
      if (fini()) return null;
      const modifie = MODIFIENT.has(action) || modification;
      // GARDE : rien ne modifie la note tant que l'écran n'est pas prêt — l'aperçu, le démarrage, l'erreur ne montrent pas
      // l'état de la note (revue S4, resultats-apercu-dessin-modifiable : sans effet aujourd'hui, aucun contrôle actif de
      // l'aperçu ne modifie ; elle protège un aperçu futur éditable — tests/test_site_web.py la fige)
      if (modifie && hote.dataset.ecran !== "pret") return null;
      occupe(1);
      if (modifie) modifsEnVol++;
      let avant = null, r;
      try {
        marquerUneFois("requete1_envoyee");
        r = await client.demander(() => { avant = note.etat; return { module: MODULE, action, etat: note.etat, ...extra }; }, { modifie });
      } catch (e) {
        if (e && e.name === "EcranFerme") return null;
        throw e;
      } finally { occupe(-1); if (modifie) modifsEnVol--; }
      if (fini()) { if (modifie) garderApresFermeture(r, action, avant, modification); return null; }
      marquerUneFois("reponse1_recue");
      if (!r.ok) {
        console.error(r.trace || r.erreur);
        dernierRefus = { erreur: String(r.erreur || ""), mort: !!r.mort };
        if (!modifie) { toast(`Le moteur a refusé : ${r.erreur}`, true); return null; }
        // une MODIFICATION refusée (ou un moteur tombé) : la note n'a pas changé — les cadres reviennent à la dernière
        // réponse du moteur (les champs à ses valeurs, le dessin, le résultat qui va avec), et le message le dit : la valeur
        // tapée restait dans son champ, face au résultat de l'ancienne (revue S4, etat-02)
        if (derniere) peindre(derniere);
        toast(`Le moteur a refusé : ${String(r.erreur || "").replace(/[.\s]+$/, "")}. La saisie n'est pas prise : les données sont revenues au dernier calcul.`, true);
        return null;
      }
      if (r.etat) {
        if (!sansHistorique) retenir(action, avant, r.etat);
        // la date de la note n'avance que si son état CHANGE par une action qui modifie (une saisie refusée, un choix
        // identique, le premier calcul d'une note rouverte ne la changent pas)
        const date = (modification || DATENT.has(action)) && JSON.stringify(avant) !== JSON.stringify(r.etat);
        appliquer(r, action, date);
      }
      return r;
    }

    // une modification partie avant la fermeture, dont la réponse arrive après (C1) : la NOTE seule — son état, ses caches,
    // son écriture, et l'adresse de SA propre entrée d'historique si elle était provisoire (C3) ; jamais le DOM ni le titre.
    // Un refus est gardé pour l'écran suivant.
    // Chaque refus NOMME la note et son module : dit dans l'écran suivant, « Saisie refusée : h : … » se lisait comme un
    // refus du module ouvert (revue S4, resultats-refus-ecran-suivant-anonyme ; app.js ne le préfixe pas une seconde fois).
    // Une note SUPPRIMÉE entre-temps (l'accueil, un autre onglet) n'est pas réécrite : on le dit (etat-03).
    function garderApresFermeture(r, action, avant, modification) {
      const de = `« ${note.nom || N.MODULES[MODULE].nom} » (${N.MODULES[MODULE].nom}) : `;
      if (!r || !r.ok) {
        refus.push(`${de}Le moteur a refusé : ${String(r ? r.erreur : "pas de réponse").replace(/[.\s]+$/, "")}. La dernière saisie n'est pas enregistrée.`);
        return;
      }
      if (r.message) refus.push(de + r.message);
      if (!r.etat) return;
      if (ecrite && ["lue", "absente"].includes(N.memoire().etat) && !N.lire(note.id)) {
        refus.push(`${de}la note a été supprimée de ce navigateur : sa dernière saisie n'y est pas réécrite.`);
        return;
      }
      const date = (modification || DATENT.has(action)) && JSON.stringify(avant) !== JSON.stringify(r.etat);
      note.etat = r.etat;
      note.resume = r.resume;
      if (note.module !== MODULE) note.module = MODULE;
      const v = (r.vues || []).find((x) => /^\s*<svg/.test(x.svg || ""));
      note.vignette = v ? v.svg : "";
      const etait = provisoire;
      provisoire = false;
      if (!N.ecrire(note, { date: etait || date })) {
        const raison = N.RAISONS[N.bilan().raison] || N.RAISONS.indisponible;
        refus.push(`${de}la dernière saisie n'a pas été enregistrée : ${raison.long}.`);
        return;
      }
      ecrite = true;
      if (etait) ctx.adresse(`?note=${note.id}`);
    }

    // -------------------------------------------------------------- annuler, rétablir (établi compact)
    // Chaque modification RÉUSSIE garde l'état d'avant (50 au plus) ; « Annuler » le rend au moteur, qui recalcule, et la
    // note s'enregistre. Une nouvelle modification efface ce qu'on pouvait rétablir. Ouvrir un autre élément n'est pas une
    // modification. Dans un champ, Ctrl+Z reste celui du champ.
    const hist = { avant: [], apres: [] };
    function retenir(action, avant, apres) {
      if (!COMPACT || !avant || !MODIFIENT.has(action) || action === "activer") return;
      if (JSON.stringify(avant) === JSON.stringify(apres)) return;
      hist.avant.push(avant);
      if (hist.avant.length > HIST_MAX) hist.avant.shift();
      hist.apres.length = 0;
      majHistorique();
    }
    function majHistorique() {
      document.querySelectorAll("[data-historique]").forEach((b) => {
        b.disabled = !(b.dataset.historique === "annuler" ? hist.avant : hist.apres).length;
      });
    }
    let enHistorique = false;
    async function historiser(sens) {
      const de = sens === "annuler" ? hist.avant : hist.apres, vers = sens === "annuler" ? hist.apres : hist.avant;
      if (!de.length || enHistorique) return;
      enHistorique = true;
      try {
        vers.push(note.etat);
        const voulu = de.pop();
        majHistorique();
        // (l'état d'avant part au moteur : « calculer » le rend, la note change — sa date avance)
        const r = await envoyer("calculer", { etat: voulu }, { sansHistorique: true, modification: true });
        if (r) toast(sens === "annuler" ? "Modification annulée" : "Modification rétablie");
      } finally { enHistorique = false; }
    }
    document.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-historique]");
      if (b && !b.disabled) historiser(b.dataset.historique);
    }, { signal });
    document.addEventListener("keydown", (ev) => {
      if (!COMPACT || !(ev.ctrlKey || ev.metaKey) || ev.altKey) return;
      if (ev.target.closest && ev.target.closest("input, textarea, select, [contenteditable]")) return;
      const k = ev.key.toLowerCase();
      if (k === "z" && !ev.shiftKey) { ev.preventDefault(); historiser("annuler"); }
      else if ((k === "z" && ev.shiftKey) || k === "y") { ev.preventDefault(); historiser("retablir"); }
    }, { signal });

    // -------------------------------------------------------------- l'affichage
    const ouvertes = new Set(memoire.lire(`stab.sec.${MODULE}`, []));
    // (établi compact) les rubriques ouvertes À LA MAIN (un clic sur leur en-tête) : elles restent ouvertes quand le dessin
    // amène à une autre rubrique ; celles que le dessin a ouvertes se replient quand il en ouvre une autre
    const voulues = new Set(COMPACT ? memoire.lire(`stab.sec.${MODULE}.voulues`, []) : []);
    // un VOLET dépliable des données (<details data-volet="k"> : le volet « Détails » de la fiche du catalogue des cornières,
    // api_web/corniere.py) garde son état d'ouverture d'un calcul à l'autre — le cadre est refait à chaque saisie — et pour la
    // session, comme les rubriques ; replié par défaut (le moteur ne l'ouvre jamais)
    const volets = new Set(memoire.lire(`stab.volets.${MODULE}`, []));
    function poserVolets(zone) {
      zone.querySelectorAll("details[data-volet]").forEach((d) => { d.open = volets.has(d.dataset.volet); });
    }
    function garderRubriques() {
      memoire.ecrire(`stab.sec.${MODULE}`, [...ouvertes]);
      if (COMPACT) memoire.ecrire(`stab.sec.${MODULE}.voulues`, [...voulues]);
    }
    let premier = true;
    let vueCourante = null;
    let derniere = null;       // la dernière réponse du moteur peinte (pas l'aperçu) : une modification refusée y revient
    let dernierRefus = null;   // le dernier refus du moteur : { erreur, mort }
    const toile = cablerToile({ bench, stage: $("#stage"), barre: $("#zoom"), signal });
    // l'établi compact : la sélection données ↔ dessin, la section courante, les colonnes réglables (js/compact.js)
    const compact = COMPACT ? cablerCompact({
      cle: MODULE, note: () => note, ouvrirRubrique, envoyer, signal,
      vue: { courante: () => vueCourante, choisir: choisirVue, ajuster: () => image(() => toile.vue(vueCourante)) },
    }) : null;

    function appliquer(r, action, date = false) {
      note.etat = r.etat;
      note.resume = r.resume;
      // une note d'un module absorbé (« predalle ») : le moteur de CE module l'a relue et rend son état dans sa propre
      // forme — la note devient une note de ce module, une fois pour toutes (l'accueil, le .stab et les prochaines
      // ouvertures la connaissent ainsi ; rien de l'ancienne note n'est perdu, docs/site/DALLES_FUSION.md § 2.1)
      if (note.module !== MODULE) note.module = MODULE;
      // la vignette de l'accueil : le premier DESSIN (une vue peut être un tableau, illisible en vignette)
      const v = (r.vues || []).find((x) => /^\s*<svg/.test(x.svg || ""));
      note.vignette = v ? v.svg : "";
      // une note provisoire ne s'écrit qu'à la première action qui la modifie ; une note écrite garde ses caches à jour, et
      // sa date n'avance qu'à une vraie modification
      if (provisoire ? MODIFIENT.has(action) : true) enregistrer({ date: provisoire || date });
      peindre(r);
      // l'AVIS d'ouverture d'un module (une note d'avant dont les chiffres changent, un nom coupé : « avis », dit une fois, en
      // tête de son résultat) : là où le résultat ne se VOIT pas — le téléphone s'ouvre sur l'onglet Données —, le même texte
      // dans le message de la page (#et-alerte, un composant des trois formats), qui reste jusqu'à ce qu'on le ferme (un
      // échec d'enregistrement déjà dit garde la place)
      if (Array.isArray(r.avis) && r.avis.length && !$("#resultat").checkVisibility() && (alerte.hidden || alerte.classList.contains("et-info"))) {
        direAlerte("info", r.avis.join(" "), true);
      }
      if (r.message) toast(r.message, true);
    }
    // les CADRES d'une réponse : statut, résumé, résultat, volets, vues, données. Le premier calcul et chaque calcul les
    // peignent par ici ; l'APERÇU aussi (``apercu`` : ni la note, ni « premier », ni l'historique, ni l'enregistrement — un
    // aperçu ne décide pas de ce que l'écran garde)
    function peindre(r, apercu = false) {
      if (!apercu) derniere = r;
      $("#statut").innerHTML = r.statut_html || "";
      afficherResume(r.resume || {});
      // le résultat refait garde sa position de lecture (une liste de contrôle dépliée, une justification qu'on vient de
      // valider : on reste à l'endroit où l'on travaillait)
      const defR = $("#resultat .pbody") ? $("#resultat .pbody").scrollTop : 0;
      // (le nom de l'élément peut être un champ de la tête des résultats, colonne de gauche compacte : la frappe en cours y survit aussi)
      const saisieR = saisieEnCours($("#resultat"));
      $("#resultat").innerHTML = r.resultat;
      if (defR && $("#resultat .pbody")) $("#resultat .pbody").scrollTop = defR;
      reprendreSaisie($("#resultat"), saisieR);
      const ouvert = $(".flyout.on.calcul");
      $("#volets").innerHTML = r.volets || "";
      if (ouvert) ouvrir(ouvert.id, $(`#resultat [data-calcul="${ouvert.id.slice(3)}"]`), true);
      afficherVues(r.vues || [], r.phrase || "");
      afficherDonnees(r.elements || "", r.donnees, apercu);
      [$("#resultat"), $("#volets"), $("#donnees")].forEach(garderGrec);
    }
    // l'APERÇU d'un calcul neuf (S4-D7, C12) : la réponse « nouveau » calculée à la construction (construire.py, un
    // processus neuf par module) — égale à la première réponse du moteur —, peinte pendant qu'il démarre, sans toucher la note
    // Il le DIT, en tête du résultat (un composant des trois formats) : la première vraie réponse refait le résultat et
    // efface la mention ; le bandeau prend « · aperçu en lecture seule » (au téléphone, il se lit en entier)
    function peindreApercu(r) {
      peindre(r, true);
      const m = document.createElement("p");
      m.className = "et-apercu-mention";
      m.setAttribute("role", "note");
      m.textContent = MENTION_APERCU;
      $("#resultat").prepend(m);
      bandeau.texte(`${TEXTE_DEMARRAGE} · ${SUITE_APERCU}`);
    }

    // le taux de la note dans l'en-tête du téléphone (le moteur l'a déjà arrondi : on l'écrit à la française)
    function afficherResume(res) {
      const c = $("#chip-m");
      const avecTaux = res.taux !== null && res.taux !== undefined;
      c.className = `mchip chip ${avecTaux && ["p", "f", "v", "r"].includes(res.niveau) ? res.niveau : "c"}`;
      // un module de consultation (niveau « i ») n'a pas de taux : son statut n'apprend rien, la place va au nom ; un taux
      // de plus de 999 % s'écrit « > 999 % » (la valeur exacte au survol), comme les pastilles du moteur ; un statut trop
      // long pour l'en-tête a sa forme courte (res.court, commun.STATUTS_COURTS), le statut entier au survol
      const tx = texteTaux(res.taux);
      c.textContent = avecTaux ? tx.texte : (res.niveau === "i" ? "" : res.court || res.statut || "");
      c.title = tx.titre || (res.court ? `${res.statut} · voir le résultat` : "Voir le résultat");
      c.hidden = !c.textContent;
    }

    // une note ROUVERTE pendant le démarrage (C11) : sa vignette et son dernier résumé, NEUTRES — aucune pastille de niveau
    // (c'est un cache, peut-être calculé par un moteur plus ancien) —, la liste en squelette ; jamais les valeurs par défaut
    function montrerRouverte() {
      const res = note.resume || {};
      const tx = texteTaux(res.taux);
      $("#statut").textContent = "";
      const c = $("#chip-m");
      c.className = "mchip chip c";
      c.textContent = "";
      c.hidden = true;
      $("#resultat").innerHTML = `<div class="pbody"><div class="et-rouverte"><b>${esc(note.nom || "")}</b>`
        + (tx.texte ? `<span class="et-rv-taux">${esc(tx.texte)}</span>` : "")
        + (res.statut ? `<span>${esc(res.statut)}</span>` : "")
        + `<small>dernier enregistrement · ${esc(N.quand(note.modifie))} · recalcul en cours</small></div></div>`;
      $("#donnees").innerHTML = `<div class="et-squelette" aria-hidden="true">${"<i></i>".repeat(9)}</div>`;
      if (/^\s*<svg/.test(note.vignette || "")) {
        $("#stage").innerHTML = `<div class="view on"><div class="sheet"><div class="draw dessin">${note.vignette}</div></div></div>`;
      }
    }

    // le champ en cours de frappe survit au rafraîchissement d'un cadre (#donnees ; #resultat, où le nom de l'élément peut
    // être un champ) : un calcul qui revient pendant qu'on tape dans un autre champ ne doit ni effacer ce qu'on tape, ni le
    // valider à notre place
    function saisieEnCours(zone) {
      const el = document.activeElement && zone.contains(document.activeElement) ? document.activeElement : null;
      if (!el || !el.dataset.chemin) return null;
      const brouillon = (el.tagName === "INPUT" || el.tagName === "TEXTAREA") && el.type !== "checkbox" && el.type !== "file" && el.value !== el.defaultValue
        ? { valeur: el.value, debut: el.selectionStart, fin: el.selectionEnd } : null;
      return { chemin: el.dataset.chemin, brouillon };
    }
    function reprendreSaisie(zone, s) {
      if (!s) return;
      const el = zone.querySelector(`[data-chemin="${CSS.escape(s.chemin)}"]:not(button)`);
      if (!el) return;
      if (s.brouillon) el.value = s.brouillon.valeur;
      el.focus({ preventScroll: true });
      if (s.brouillon) { try { el.setSelectionRange(s.brouillon.debut, s.brouillon.fin); } catch (e) { /* champ sans curseur */ } }
    }

    function afficherDonnees(elements, html, apercu = false) {
      const zone = $("#donnees");
      const saisie = saisieEnCours(zone);
      const saisieElt = ELT ? saisieEnCours(ELT) : null;
      const defile = zone.scrollTop;
      zone.innerHTML = elements + html;
      // (colonne compacte) la LIGNE de l'élément ouvert — son nom, la liste, copier, supprimer — monte dans l'en-tête du
      // panneau, à côté d'« Annuler » et « Rétablir » (construire.compact : l'emplacement .et-elt) ; un nom en cours de
      // frappe survit au calcul qui revient
      if (ELT) {
        const ligne = zone.querySelector(".et-elsel");
        ELT.replaceChildren(...(ligne ? [ligne] : []));
        reprendreSaisie(ELT, saisieElt);
      }
      separerNoms(zone);
      // les sections gardent leur état (ouvert, fermé) d'un calcul à l'autre (l'aperçu les montre ainsi, sans rien en garder)
      const ouv = apercu ? new Set(ouvertes) : ouvertes;
      if (COMPACT) {
        // les rubriques gardées de la session ; à la première ouverture, celle que le moteur ouvre (une seule, par défaut)
        const secs = [...zone.querySelectorAll(".sec[data-sec]")];
        if (premier && !secs.some((s) => ouv.has(s.dataset.sec))) {
          const k = (secs.find((s) => s.classList.contains("open")) || secs[0] || {}).dataset?.sec;
          if (k) ouv.add(k);
        }
        secs.forEach((s) => s.classList.toggle("open", ouv.has(s.dataset.sec)));
      } else {
        zone.querySelectorAll(".sec[data-sec]").forEach((s) => {
          const cle = s.dataset.sec;
          if (premier) { if (s.classList.contains("open")) ouv.add(cle); }
          else s.classList.toggle("open", ouv.has(cle));
        });
      }
      if (!apercu) premier = false;
      poserVolets(zone);
      zone.scrollTop = defile;
      marquerMeteo();
      reprendreSaisie(zone, saisie);
      if (compact) compact.rafraichir();
    }
    // (« toggle » ne remonte pas : écouté à la capture) un volet qu'on ouvre ou referme à la main est gardé
    $("#donnees").addEventListener("toggle", (ev) => {
      const d = ev.target;
      if (!(d instanceof HTMLDetailsElement) || !d.dataset.volet) return;
      if (d.open) volets.add(d.dataset.volet); else volets.delete(d.dataset.volet);
      memoire.ecrire(`stab.volets.${MODULE}`, [...volets]);
    }, { capture: true, signal });

    // « Dans cette note » : chaque élément s'écrit « nom · dimensions ». Le nom passe dans son propre <span class="nom">
    // pour que ce soit LUI qui se coupe (points de suspension) quand la place manque, jamais les dimensions qui le suivent.
    function separerNoms(zone) {
      zone.querySelectorAll(".ni .t").forEach((t) => {
        if (t.querySelector(":scope > .nom")) return;
        const avant = [];
        for (const n of t.childNodes) { if (n.nodeType === 1 && n.tagName === "SPAN") break; avant.push(n); }
        if (!avant.length) return;
        const nom = document.createElement("span");
        nom.className = "nom";
        avant.forEach((n) => nom.append(n));
        nom.normalize();
        if (nom.lastChild && nom.lastChild.nodeType === 3) nom.lastChild.data = nom.lastChild.data.replace(/\s+$/, "");
        t.prepend(nom);
      });
    }

    // « mm · à l'échelle » : sur un dessin seulement — pas sur un tableau (vue HTML), ni sur une vue que le module dit hors
    // échelle (une courbe, un diagramme : « echelle »: false). L'établi compact ne l'écrit pas : le dessin s'ajuste à la
    // feuille, il n'est à aucune échelle d'impression (les cotes disent les dimensions).
    const echelle = (v) => !COMPACT && v.echelle !== false && estDessin(v);
    // le titre d'une vue : l'EN-TÊTE technique que le module compose (« entete » : la section, ses aciers, ses étriers, la
    // bande des efforts), sinon le titre et la légende
    const titreVue = (v) => (v.entete ? `<div class="stitle et-stitle">${v.entete}</div>`
      : `<div class="stitle"><b>${esc(v.titre)}</b>${v.legende ? `<span class="leg">${esc(v.legende)}</span>` : ""}${echelle(v) ? '<span class="scale">mm · à l\'échelle</span>' : ""}</div>`);

    function afficherVues(vues, phrase) {
      const onglets = $("#vues"), stage = $("#stage");
      bench.classList.toggle("sans-vue", !vues.length);
      if (!vues.length) {
        // un module de consultation peut n'avoir aucune vue pour une note : rien d'anormal, on le dit sans alarmer
        onglets.innerHTML = "";
        stage.innerHTML = '<div class="sansvue">Pas de dessin pour cette note.</div>';
        toile.vue(null);
        return;
      }
      if (!vues.some((v) => v.cle === vueCourante)) vueCourante = vues[0].cle;
      onglets.innerHTML = vues.map((v) => `<button type="button" data-vue="${esc(v.cle)}" class="${v.cle === vueCourante ? "on" : ""}">${esc(v.court)}</button>`).join("");
      image(marquerDebord);
      const date = new Date().toLocaleDateString("fr-BE");
      const cart = `<div class="cart"><div><span>Note</span><b data-cart="nom">${esc(note.nom || "")}</b></div><div><span>Chantier</span><b data-cart="chantier">${esc(note.chantier || "—")}</b></div>`
        + `<div class="ind"><span>Indice</span><b>${esc(note.indice || "A")}</b></div><div class="dt"><span>Date</span><b>${date}</b></div>`
        + `<div class="mark"><i></i><span>Stab Méthodes</span></div></div>`;
      // chaque vue porte SA légende ; la phrase du module (la note en une ligne) se lit sous le titre de la première
      stage.innerHTML = vues.map((v, i) => `<div class="view${v.cle === vueCourante ? " on" : ""}" data-vue="${esc(v.cle)}"><div class="sheet">${titreVue(v)}`
        + `${i === 0 && phrase ? `<div class="sphrase">${phrase}</div>` : ""}`
        + `<div class="draw${estDessin(v) ? " dessin" : " bloc"}">${v.svg}</div>${cart}</div></div>`).join("");
      garderGrec(stage);
      toile.vue(vueCourante);
    }

    // trop de vues pour la largeur : la bande défile, son bord droit s'estompe (on devine la suite)
    function marquerDebord() {
      const o = $("#vues");
      o.classList.toggle("defile", o.scrollWidth > o.clientWidth + 1);
    }
    // (à chaque changement de taille de la bande, transitions comprises : la toile anime sa marge quand un cadre se replie)
    const bandeVues = new ResizeObserver(() => { if (!fini()) marquerDebord(); });
    bandeVues.observe($("#vues"));
    signal.addEventListener("abort", () => bandeVues.disconnect(), { once: true });
    // (téléphone) le message de la page (#et-alerte) se pose sous le bas RÉEL des onglets : le bandeau du démarrage les
    // agrandit — à 118 px fixes, le message les recouvrait (revue S4, ecrans-alerte-couvre-onglets) ; .mtabs est collée
    // sous l'en-tête (sticky) : son bas ne bouge qu'avec sa taille
    const barreOnglets = $(".mtabs");
    if (barreOnglets) {
      const racine = document.documentElement;
      const basOnglets = new ResizeObserver(() => {
        if (!fini()) racine.style.setProperty("--mtabs-bas", `${Math.round(barreOnglets.getBoundingClientRect().bottom)}px`);
      });
      basOnglets.observe(barreOnglets);
      signal.addEventListener("abort", () => basOnglets.disconnect(), { once: true });
      signal.addEventListener("abort", () => racine.style.removeProperty("--mtabs-bas"), { once: true });
    }

    function choisirVue(k) {
      if (!bench.querySelector(`#vues [data-vue="${CSS.escape(k)}"]`)) return;
      vueCourante = k;
      bench.querySelectorAll("[data-vue]").forEach((e) => e.classList.toggle("on", e.dataset.vue === vueCourante));
      toile.vue(vueCourante);
    }
    $("#vues").addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-vue]");
      if (!b) return;
      choisirVue(b.dataset.vue);
      if (compact) compact.vueChoisie(vueCourante);
    }, { signal });

    // -------------------------------------------------------------- le téléphone : onglets, menu
    // (montrerOnglet : l'onglet seul ; onglet : le geste de l'utilisateur — la page remonte en haut, et l'onglet est gardé
    // dans l'entrée d'historique de CET écran : un retour le rend, avec le défilement — revue S4,
    // ecrans-retour-etabli-telephone)
    function montrerOnglet(t) {
      bench.dataset.tab = t;
      document.querySelectorAll(".mtabs [data-onglet]").forEach((b) => {
        b.classList.toggle("on", b.dataset.onglet === t);
        b.setAttribute("aria-pressed", b.dataset.onglet === t ? "true" : "false");
      });
      fermerVolets();
      if (t === "dessin") image(() => toile.vue(vueCourante));
    }
    function onglet(t) {
      montrerOnglet(t);
      if (telephone.matches) scrollTo({ top: 0 });
      if (ONGLETS.includes(t) && typeof ctx.garderEtatEntree === "function") ctx.garderEtatEntree({ onglet: t });
    }
    const ongletEntree = ctx.etatEntree && ONGLETS.includes(ctx.etatEntree.onglet) ? ctx.etatEntree.onglet : null;
    if (ongletEntree && ongletEntree !== bench.dataset.tab) montrerOnglet(ongletEntree);
    document.addEventListener("click", (ev) => {
      const o = ev.target.closest("[data-onglet], [data-vers-onglet]");
      if (o) onglet(o.dataset.onglet || o.dataset.versOnglet);
    }, { signal });

    const menu = $("#mmenu"), menuBouton = $("#menu-m");
    function menuOuvert(on) {
      menu.hidden = !on;
      menuBouton.setAttribute("aria-expanded", on ? "true" : "false");
    }
    menuBouton.addEventListener("click", (ev) => { ev.stopPropagation(); menuOuvert(menu.hidden); }, { signal });
    document.addEventListener("click", (ev) => {
      if (menu.hidden || ev.target.closest("#menu-m")) return;
      // un geste ailleurs referme le menu ; un choix du menu aussi, après avoir agi (une étiquette « Ouvrir » ouvre le
      // choix du fichier APRÈS le clic : le menu ne disparaît qu'ensuite)
      if (!ev.target.closest("#mmenu")) menuOuvert(false);
      else if (ev.target.closest(".mi")) setTimeout(() => { if (!fini()) menuOuvert(false); });
    }, { signal });

    // -------------------------------------------------------------- la saisie
    function lireChamp(el) {
      if (el.type === "checkbox") return el.checked;
      return el.value;
    }

    // les menus et les cases partent à la sélection ; les champs de texte, en quittant le champ (ou Entrée) — et
    // seulement si leur valeur diffère de celle du moteur (defaultValue), même après un rafraîchissement
    function valider(el) {
      if (el.dataset.sorte === "nombre" || el.dataset.sorte === "entier") {
        const t = el.value.replace(/[\s  ]/g, "").replace(",", ".");
        const x = Number(t);
        const ko = t === "" || !Number.isFinite(x) || (el.dataset.min !== undefined && x < +el.dataset.min) || (el.dataset.max !== undefined && x > +el.dataset.max);
        el.closest(".field").classList.toggle("erreur", ko);
        if (ko) { toast(`Valeur refusée : « ${el.value} »`, true); return; }
      }
      envoyer("modifier", { chemin: el.dataset.chemin.split("/"), valeur: lireChamp(el) });
    }
    $("#donnees").addEventListener("change", (ev) => {
      const f = ev.target.closest("input[type=file][data-chemin]");
      if (f) { lireFichier(f); return; }
      const el = ev.target.closest("select[data-chemin], input[type=checkbox][data-chemin]");
      if (el) valider(el);
    }, { signal });
    $("#donnees").addEventListener("focusout", (ev) => {
      const el = ev.target.closest("input[data-chemin]:not([type=checkbox]):not([type=file]), textarea[data-chemin]");
      if (el && el.value !== el.defaultValue) valider(el);
    }, { signal });

    // un fichier choisi (essais de sol, relevés…) part au moteur comme TEXTE : { nom, texte } ; le moteur le lit
    async function lireFichier(f) {
      const fichier = f.files && f.files[0];
      if (!fichier) return;
      if (fichier.size > 2 * 1024 * 1024) { toast("Fichier trop gros (2 Mo au plus)", true); return; }
      const texte = await fichier.text();
      if (fini()) return;
      envoyer("modifier", { chemin: f.dataset.chemin.split("/"), valeur: { nom: fichier.name, texte } });
    }

    $("#donnees").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && ev.target.matches("input[data-chemin]")) ev.target.blur();
    }, { signal });
    // la LIGNE de l'élément ouvert, dans l'en-tête du panneau (colonne compacte) : son NOM est un champ — un clic, et on le
    // renomme ; Entrée valide, Échap rend l'ancien nom — ; la liste des éléments, copier, supprimer passent par agirElement
    if (ELT) {
      ELT.addEventListener("focusout", (ev) => {
        const el = ev.target.closest("input[data-chemin]");
        if (el && el.value !== el.defaultValue) {
          if (!el.value.trim()) { el.value = el.defaultValue; toast("Un nom ne peut pas être vide", true); return; }
          valider(el);
        }
      }, { signal });
      ELT.addEventListener("keydown", (ev) => {
        const el = ev.target.closest("input[data-chemin]");
        if (!el) return;
        if (ev.key === "Enter") el.blur();
        else if (ev.key === "Escape") { el.value = el.defaultValue; el.blur(); }
      }, { signal });
      ELT.addEventListener("click", (ev) => { agirElement(ev); }, { signal });
    }
    // « Dans cette note » : supprimer, ouvrir, ajouter, copier un élément. Le bloc vient du cadre Données ; l'établi compact
    // le déplace dans la liste de gauche (js/compact.js) : les deux cadres répondent par cette même fonction.
    function agirElement(ev) {
      const sup = ev.target.closest("[data-supprimer]");
      if (sup) {
        ev.stopPropagation();
        if (confirm(sup.dataset.confirmer || "Supprimer cet élément de la note ?")) envoyer("supprimer", { index: +sup.dataset.supprimer });
        return true;
      }
      const act = ev.target.closest("[data-activer]");
      if (act) { if (!act.classList.contains("on")) envoyer("activer", { index: +act.dataset.activer }); return true; }
      if (ev.target.closest("[data-ajouter]")) { envoyer("ajouter"); return true; }
      // (établi compact) « Copier » dans le menu ⋯ d'un élément : CET élément, la copie s'insère après lui
      const dup = ev.target.closest("[data-dupliquer-index]");
      if (dup) { envoyer("dupliquer", { index: +dup.dataset.dupliquerIndex }); return true; }
      if (ev.target.closest("[data-dupliquer]")) { envoyer("dupliquer"); return true; }
      return false;
    }
    $("#donnees").addEventListener("click", (ev) => {
      const seg = ev.target.closest("button[data-chemin][data-valeur]");
      if (seg) { if (!seg.classList.contains("on")) envoyer("modifier", { chemin: seg.dataset.chemin.split("/"), valeur: seg.dataset.valeur }); return; }
      const hd = ev.target.closest(".sechd[data-replier]");
      if (hd) {
        const s = hd.parentElement, k = s.dataset.sec;
        s.classList.toggle("open");
        const on = s.classList.contains("open");
        if (on) ouvertes.add(k); else ouvertes.delete(k);
        // (établi compact) un en-tête ouvre ou referme SA rubrique, sans toucher aux autres : on en garde plusieurs ouvertes
        // si on le veut ; ce choix fait à la main survit à la navigation depuis le dessin
        if (COMPACT) { if (on) voulues.add(k); else voulues.delete(k); }
        garderRubriques();
        return;
      }
      // le sélecteur d'armatures (« Proposer ») et « Renommer » d'un élément, posés dans les données
      const pk = ev.target.closest("[data-pk]");
      if (pk) { ev.stopPropagation(); ouvrirSelecteur(pk.dataset.pk, pk); return; }
      const rn = ev.target.closest("[data-renommer]");
      if (rn) {
        ev.stopPropagation();
        // (établi compact) le menu ⋯ d'une poutre qui n'est pas ouverte : elle s'ouvre d'abord, son nom se choisit ensuite
        const i = Number(rn.dataset.renommer);
        if (COMPACT && rn.closest(".et-pt") && Number.isInteger(i) && note.etat && i !== note.etat.actif) envoyer("activer", { index: i }).then((r) => { if (r) renommer(); });
        else renommer();
        return;
      }
      // les fiches de la bibliothèque (bloc « Justifications », api_web/fiches.py) : voir au centre, monter, descendre, retirer
      const vf = ev.target.closest("[data-vue-fiche]");
      if (vf) {
        ev.stopPropagation();
        choisirVue(vf.dataset.vueFiche);
        if (compact) compact.vueChoisie(vueCourante);
        if (telephone.matches) onglet("dessin");
        return;
      }
      const fa = ev.target.closest("[data-fiche-retirer],[data-fiche-monter],[data-fiche-descendre]");
      if (fa) {
        ev.stopPropagation();
        const quoi = fa.hasAttribute("data-fiche-retirer") ? "retirer" : fa.hasAttribute("data-fiche-monter") ? "monter" : "descendre";
        envoyer("fiches", { quoi, index: +fa.getAttribute(`data-fiche-${quoi}`) }).then((r) => { if (r && !r.message && quoi === "retirer") toast("Fiche retirée de la note"); });
        return;
      }
      agirElement(ev);
    }, { signal });

    // l'établi compact : le DESSIN amène à la rubrique d'une pièce. Elle s'ouvre ; les rubriques que le dessin avait ouvertes
    // se replient, celles ouvertes à la main (voulues) restent ouvertes
    function ouvrirRubrique(cle) {
      const zone = $("#donnees");
      zone.querySelectorAll(".sec[data-sec]").forEach((x) => {
        const k = x.dataset.sec, on = k === cle || voulues.has(k);
        x.classList.toggle("open", on);
        if (on) ouvertes.add(k); else ouvertes.delete(k);
      });
      garderRubriques();
      const s = zone.querySelector(`.sec[data-sec="${CSS.escape(cle)}"]`);
      if (s) image(() => s.scrollIntoView({ block: "nearest" }));
    }
    // « Renommer » l'élément ouvert : son champ « Nom », choisi — sur la ligne de l'élément dans l'arborescence (établi
    // compact), dans la tête des RÉSULTATS (colonne de gauche compacte : au téléphone l'onglet Résultat s'ouvre, au PC la
    // colonne repliée se rouvre), dans sa rubrique ailleurs (qui s'ouvre alors)
    function renommer() {
      if (compact) compact.deplierPoutre();
      // (colonne compacte : le nom est dans la ligne de l'élément, en tête du panneau — au téléphone, l'onglet Données)
      if (ELT && $('input[data-chemin="nom"]', ELT)) {
        if (telephone.matches) onglet("donnees");
        bench.classList.remove("sans-g");
        const n = $('input[data-chemin="nom"]', ELT);
        n.focus();
        n.select();
        return;
      }
      const champ = $('#donnees input[data-chemin="nom"]');
      if (!champ) return;
      const rub = champ.closest(".sec[data-sec]");
      if (rub && COMPACT) ouvrirRubrique(rub.dataset.sec); else if (rub) rub.classList.add("open");
      champ.focus();
      champ.select();
    }

    // -------------------------------------------------------------- les volets : calcul détaillé et sélecteur
    function fermerVolets() {
      bench.querySelectorAll(".flyout.on").forEach((f) => f.classList.remove("on"));
      bench.querySelectorAll(".el.sel").forEach((e) => e.classList.remove("sel"));
      bench.classList.remove("fly");
      document.documentElement.classList.remove("volet-ouvert");
    }

    function ouvrir(id, ancre, garder = false) {
      if (!garder) fermerVolets();
      const f = document.getElementById(id);
      if (!f) return;
      f.classList.add("on");
      // au téléphone, le volet monte du bas, par-dessus l'onglet ouvert ; le fond s'assombrit et ne défile plus
      bench.classList.add("fly");
      document.documentElement.classList.toggle("volet-ouvert", telephone.matches);
      if (ancre && ancre.classList.contains("el")) ancre.classList.add("sel");
      if (ancre && !f.classList.contains("calcul") && !telephone.matches) {
        const b = bench.getBoundingClientRect(), a = ancre.getBoundingClientRect();
        f.style.top = Math.max(10, Math.min(a.top - b.top - 12, b.height - f.offsetHeight - 10)) + "px";
      }
      const l = f.querySelector(".pk-list");
      if (l) {
        const r = l.querySelector(".pk-row");
        l.scrollTop = (r ? r.offsetHeight : 34) * (+f.dataset.debut || 0);
      }
    }

    const filtres = {};   // les familles cochées de chaque sélecteur, gardées pendant la session
    async function ouvrirSelecteur(role, ancre) {
      const f = filtres[role] || {};
      const r = await envoyer("selecteur", { role, familles: f.familles || null, froid: f.froid ?? true });
      if (!r || fini()) return;
      const ancien = document.getElementById(`pk-${role}`);
      if (ancien) ancien.remove();
      $("#volets").insertAdjacentHTML("beforeend", r.html);
      garderGrec(document.getElementById(`pk-${role}`));
      ouvrir(`pk-${role}`, ancre);
    }

    $("#resultat").addEventListener("click", (ev) => {
      const pk = ev.target.closest(".prof[data-pk]");
      if (pk) { ev.stopPropagation(); ouvrirSelecteur(pk.dataset.pk, pk.closest(".el") || pk); return; }
      const c = ev.target.closest("[data-calcul]");
      if (c) { ev.stopPropagation(); ouvrir(`ck-${c.dataset.calcul}`, c); }
    }, { signal });

    $("#volets").addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (ev.target.closest("[data-fermer]")) { fermerVolets(); return; }
      const picker = ev.target.closest(".picker");
      const ligne = ev.target.closest(".pk-row[data-choix]");
      if (ligne && picker) {
        const role = picker.dataset.pk;
        fermerVolets();
        if (!ligne.classList.contains("cur")) envoyer("choisir", { role, nom: ligne.dataset.choix }).then((r) => r && toast(`${ligne.dataset.choix} retenu`));
        return;
      }
      const chip = ev.target.closest(".fchip");
      if (chip && picker) {
        const role = picker.dataset.pk;
        const f = filtres[role] || (filtres[role] = {});
        const on = [...picker.querySelectorAll(".fchip[data-famille].on")].map((c) => c.dataset.famille);
        if (chip.dataset.famille) {
          const k = chip.dataset.famille;
          f.familles = on.includes(k) ? on.filter((x) => x !== k) : [...on, k];
          if (!f.familles.length) { toast("Garder au moins une famille", true); f.familles = on; return; }
        } else if (chip.dataset.froid) {
          f.familles = on;
          f.froid = !chip.classList.contains("on");
        }
        ouvrirSelecteur(role, $(`#resultat .prof[data-pk="${CSS.escape(role)}"]`)?.closest(".el"));
      }
    }, { signal });

    $("#volets").addEventListener("mouseover", (ev) => {
      const r = ev.target.closest(".pk-row");
      if (!r) return;
      const p = r.closest(".picker").querySelector(".pk-prop");
      if (p) p.innerHTML = r.dataset.prop || "";
    }, { signal });

    // au toucher, pas de survol : le premier appui sur une ligne du sélecteur montre ses propriétés… et la choisit ; on
    // montre donc les propriétés dès que le doigt se pose
    $("#volets").addEventListener("pointerdown", (ev) => {
      const r = ev.target.closest(".pk-row");
      if (!r || ev.pointerType === "mouse") return;
      const p = r.closest(".picker").querySelector(".pk-prop");
      if (p) p.innerHTML = r.dataset.prop || "";
    }, { signal });

    // une aide ⓘ : à la souris, son texte au survol (title) ; au DOIGT, le survol n'existe pas — un appui l'ouvre dans une
    // bulle près de l'icône (R1 : l'explication se lit sur les trois formats, y compris celles rangées dans ⓘ pour condenser
    // les données) ; un second appui, un appui ailleurs, Échap ou un défilement la ferment. Au doigt, l'icône (13 px) est
    // une cible trop petite : le LIBELLÉ du champ qui la porte l'ouvre aussi (une zone d'appui agrandie autour de l'icône
    // débordait de sa boîte — le harnais « formats » la voyait, à raison, comme un nom coupé)
    const bulle = document.createElement("div");
    bulle.className = "aide-bulle";
    bulle.setAttribute("role", "tooltip");
    bulle.hidden = true;
    document.body.appendChild(bulle);
    let aideOuverte = null;
    let pointeur = "mouse";
    document.addEventListener("pointerdown", (ev) => { pointeur = ev.pointerType || "mouse"; }, { capture: true, signal });
    function fermerAide() { bulle.hidden = true; aideOuverte = null; }
    document.addEventListener("click", (ev) => {
      let a = ev.target.closest(".aide[title]");
      if (!a && pointeur !== "mouse") {
        // (un libellé qui contient son champ le choisit, comme avant : seul un libellé sans contrôle ouvre l'aide)
        const l = ev.target.closest("label");
        if (l && !l.querySelector("input, select, textarea, button")) a = l.querySelector(".aide[title]");
      }
      if (!a) { if (aideOuverte && !ev.target.closest(".aide-bulle")) fermerAide(); return; }
      ev.preventDefault();
      ev.stopPropagation();
      if (aideOuverte === a) { fermerAide(); return; }
      bulle.textContent = a.title;
      bulle.style.left = "0px";
      bulle.style.top = "0px";
      bulle.hidden = false;
      const r = a.getBoundingClientRect(), bw = bulle.offsetWidth, bh = bulle.offsetHeight;
      bulle.style.left = `${Math.round(Math.min(Math.max(8, r.left + r.width / 2 - bw / 2), innerWidth - bw - 8))}px`;
      bulle.style.top = `${Math.round(r.bottom + 6 + bh > innerHeight - 8 ? Math.max(8, r.top - bh - 6) : r.bottom + 6)}px`;
      aideOuverte = a;
    }, { capture: true, signal });
    document.addEventListener("scroll", () => { if (aideOuverte) fermerAide(); }, { capture: true, signal });

    document.addEventListener("click", (ev) => { if (!ev.target.closest(".flyout,[data-pk],[data-calcul]")) fermerVolets(); }, { signal });
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { fermerVolets(); menuOuvert(false); fermerAide(); } }, { signal });
    bench.querySelectorAll("[data-coll]").forEach((b) => b.addEventListener("click", () => {
      bench.classList.toggle(b.dataset.coll === "g" ? "sans-g" : "sans-d");
      image(() => toile.vue(vueCourante));
    }, { signal }));

    // -------------------------------------------------------------- un fichier propre au module (export JSON…)
    // Le module pose dans ses fragments <button data-telecharger="<quoi>"> ; le moteur rend le fichier
    // ({"fichier": {"nom", "type", "texte"}}) et le navigateur l'enregistre sous ce nom, nettoyé (commun.js).
    async function telechargerFichier(quoi) {
      occupe(1);
      // la note PDF : la première se prépare quelques secondes (le paquet de la note se charge), on le dit
      if (quoi === "note_pdf") toast("Préparation de la note de calcul PDF…");
      try {
        // le nom, le chantier, l'indice de la note : la page de garde d'une note PDF (un module qui ne les attend pas les ignore)
        const infos = { nom: note.nom || "", chantier: note.chantier || "", indice: note.indice || "A", date: new Date().toLocaleDateString("fr-BE") };
        let r;
        try {
          r = await client.demander(() => ({ module: MODULE, action: "telecharger", etat: note.etat, quoi, infos }), { note: quoi === "note_pdf" });
        } catch (e) { if (e && e.name === "EcranFerme") return; throw e; }
        if (fini()) return;   // l'écran s'est fermé : le fichier ne part pas (le corps est celui de l'écran suivant)
        const f = r && r.ok ? r.fichier : null;
        if (!f || typeof f.nom !== "string" || (typeof f.texte !== "string" && typeof f.base64 !== "string")) {
          toast(`Téléchargement refusé : ${(r && !r.ok && r.erreur) || "le moteur n'a pas rendu de fichier"}`, true);
          if (r && r.trace) console.error(r.trace);
          return;
        }
        toast(`${telecharger(f.nom, f.base64 ? { base64: f.base64 } : f.texte, f.type || "application/octet-stream")} téléchargé`);
      } finally { occupe(-1); }
    }
    // à la capture : les volets arrêtent la propagation de leurs clics
    bench.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-telecharger]");
      if (!b) return;
      ev.preventDefault();
      ev.stopPropagation();
      telechargerFichier(b.dataset.telecharger);
    }, { capture: true, signal });

    // -------------------------------------------------------------- envoyer l'élément ouvert à un autre module
    // Le module pose <button data-transmettre="<clé du module de destination>"> (« Détailler dans Taux d'armature »
    // d'une poutre) ; le moteur rend la note à créer ({"note": {"module", "nom", "etat"}}, l'état relu par le module de
    // destination). L'écran la crée — même chantier, même dossier — et l'ouvre. La note d'ici n'est pas touchée.
    async function transmettre(vers) {
      occupe(1);
      try {
        let r;
        try {
          r = await client.demander(() => ({ module: MODULE, action: "transmettre", etat: note.etat, vers, date: new Date().toLocaleDateString("fr-BE") }));
        } catch (e) { if (e && e.name === "EcranFerme") return; throw e; }
        if (fini()) return;
        const x = r && r.ok ? r.note : null;
        if (!x || typeof x.module !== "string" || !Object.hasOwn(N.MODULES, x.module) || !x.etat || typeof x.etat !== "object") {
          toast(`Envoi refusé : ${(r && !r.ok && r.erreur) || "le moteur n'a pas rendu de note"}`, true);
          if (r && r.trace) console.error(r.trace);
          return;
        }
        const n = N.nouvelle({ module: x.module, nom: typeof x.nom === "string" ? x.nom : "", chantier: note.chantier || "", dossier: note.dossier || "" });
        n.etat = x.etat;
        if (!N.ecrire(n)) { toast("La note n'a pas pu être créée dans ce navigateur : rien n'est envoyé", true); return; }
        ctx.naviguer(`${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}`);
      } finally { occupe(-1); }
    }
    // à la capture, comme les téléchargements : les volets arrêtent la propagation de leurs clics
    bench.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-transmettre]");
      if (!b || b.disabled || b.getAttribute("aria-disabled") === "true") return;
      ev.preventDefault();
      ev.stopPropagation();
      transmettre(b.dataset.transmettre);
    }, { capture: true, signal });

    // -------------------------------------------------------------- la météo du lieu (Open-Meteo), à la demande
    // Le module pose <button data-meteo="<plan>" data-meteo-chemin="…"> (commun.bouton_meteo) : le PLAN vient du moteur ;
    // js/meteo.js fait les appels vers Open-Meteo et rend les réponses BRUTES, qui partent au module (« modifier » sur le
    // chemin du bouton). Le module les lit, les garde dans la note, et le calcul se fait hors ligne. Une seule récupération à
    // la fois ; pendant ce temps, le bouton dit « Récupération de la météo… ». L'écran qui se ferme l'ANNULE (le signal du
    // montage) : sans message, sans requête au moteur.
    function marquerMeteo() {
      document.querySelectorAll("#donnees [data-meteo]").forEach((b) => {
        const attend = !!meteoEnCours && b.dataset.meteoChemin === meteoEnCours;
        b.classList.toggle("attente", attend);
        if (attend) {
          if (b.dataset.etaitGrise === undefined) b.dataset.etaitGrise = b.disabled ? "1" : "";
          b.disabled = true;
          b.setAttribute("aria-busy", "true");
          const t = b.querySelector("span");
          if (t) { t.dataset.libelle = t.dataset.libelle || t.textContent; t.textContent = "Récupération de la météo…"; }
        }
      });
    }

    async function importerMeteo(b) {
      if (meteoEnCours) return;
      let plan = null;
      try { plan = JSON.parse(b.dataset.meteo); } catch (e) { /* le plan illisible est refusé par recupererMeteo */ }
      const chemin = b.dataset.meteoChemin || "";
      meteoEnCours = chemin;
      marquerMeteo();
      occupe(0);
      try {
        const r = await recupererMeteo(plan, { signal });
        if (fini() || r.annule) return;
        if (!r.ok) { toast(r.erreur, true); return; }
        const rep = await envoyer("modifier", { chemin: chemin.split("/"), valeur: r.valeur });
        if (fini()) return;
        if (rep && !rep.message) toast("Météo récupérée : la note garde les températures, le calcul se fait hors ligne.");
      } catch (e) {
        if (!fini()) toast(`La météo n'a pas pu être récupérée : ${e.message}`, true);
      } finally {
        meteoEnCours = null;
        if (!fini()) {
          // le bouton revient à lui-même (le cadre Données a pu être refait entre-temps : chaque bouton retrouve son libellé)
          document.querySelectorAll("#donnees [data-meteo].attente").forEach((x) => {
            x.classList.remove("attente");
            x.removeAttribute("aria-busy");
            x.disabled = x.dataset.etaitGrise === "1";
            delete x.dataset.etaitGrise;
            const t = x.querySelector("span");
            if (t && t.dataset.libelle) t.textContent = t.dataset.libelle;
          });
          occupe(0);
        }
      }
    }
    // à la capture, comme « télécharger » : rien d'autre ne réagit à ce clic
    bench.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-meteo]");
      if (!b) return;
      ev.preventDefault();
      ev.stopPropagation();
      if (!b.disabled) importerMeteo(b);
    }, { capture: true, signal });

    // -------------------------------------------------------------- la bibliothèque de la note (js/biblio_note.js)
    // (le panneau n'existe que pour un module dont la famille a une bibliothèque : construire.compact ; ses lectures passent
    // par le client de l'écran)
    const bibliotheque = cablerPanneauBibliotheque({ module: MODULE, note: () => note, moteur: client, envoyer, signal,
      avant: () => { fermerVolets(); menuOuvert(false); } });

    // -------------------------------------------------------------- fichier, impression, bibliothèque
    document.addEventListener("click", (ev) => {
      const a = ev.target.closest("[data-action]");
      if (!a) return;
      if (a.dataset.action === "exporter") N.exporter(note);
      else if (a.dataset.action === "imprimer") window.print();
      else if (a.dataset.action === "bibliotheque") bibliotheque.ouvrir(a);
      else if (a.dataset.action === "pdf") {
        if (NOTE_PDF) telechargerFichier("note_pdf");
        else toast("La note PDF arrive bientôt sur le site ; elle reste dans Streamlit.");
      }
    }, { signal });
    $("#ouvrir").addEventListener("change", async (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      try {
        const n = await N.importer(f);
        ctx.naviguer(`${N.pageDe(n.module)}?note=${n.id}`);
      } catch (e) { if (!fini()) toast(`Import impossible : ${e.message}`, true); }
    }, { signal });

    // -------------------------------------------------------------- démarrage
    // L'action du premier calcul se décide AVANT de peindre quoi que ce soit : « nouveau » pour un calcul neuf (sans
    // état : depuis sa carte, « Nouvelle note », une note absente), « calculer » pour une note rouverte.
    const premiere = note.etat ? "calculer" : "nouveau";
    if (note.etat) montrerRouverte();
    else if (ctx.moteur.etat !== "pret") apercu();
    (async () => {
      try {
        await ctx.moteur.pret;
        if (fini()) return;
        const r = await envoyer(premiere);
        if (fini()) return;
        marquer("reponse1_appliquee");
        if (!r) throw Object.assign(new Error((dernierRefus && dernierRefus.erreur) || "le premier calcul a échoué"), { refus: !!dernierRefus && !dernierRefus.mort });
        bandeau.retirer();
        lecture.lever();
        hote.dataset.ecran = "pret";
        ctx.annoncer("prêt");
        // le défilement d'un retour (ou d'une avance) : rendu quand le contenu qui donne sa hauteur à la page est là — au
        // téléphone, la page entière défile, et à l'image suivante du montage elle n'avait encore que son squelette
        // (revue S4, ecrans-retour-etabli-telephone) ; l'application le rend elle-même à 1,5 s si on ne l'a pas fait
        image(() => { if (typeof ctx.rendreDefilement === "function") ctx.rendreDefilement(); });
        // la sonde : deux images plus tard, le navigateur a peint le premier résultat
        if (SONDE) requestAnimationFrame(() => requestAnimationFrame(() => marquer("premier_rendu")));
      } catch (e) {
        if (fini()) return;
        hote.dataset.ecran = "erreur";
        // le moteur a démarré mais refuse CE calcul (une note abîmée, d'un moteur plus ancien) ; sinon le démarrage a raté
        // — une erreur nommée de version se dit seule (commun.texteEchecDemarrage)
        const raison = e && e.refus
          ? `Le calcul n'a pas pu démarrer : le moteur a refusé ${note.etat ? "cette note" : "ce calcul"} (${String(e.message).replace(/[.\s]+$/, "")}).`
          : texteEchecDemarrage(e, ctx.moteur);
        bandeau.erreur(raison);
        // ce qui marche sans le moteur redevient actif (exporter la note telle qu'enregistrée, imprimer), aux trois formats ;
        // pas les champs, ni la note PDF, ni « ranger » (revue S4, ecrans-erreur-export)
        lecture.rendre(ACTIONS_SANS_MOTEUR);
        ctx.annoncer(raison);
      }
    })();

    // l'aperçu : lu avec l'empreinte que le manifeste donne (?v=), peint s'il arrive avant la première réponse
    async function apercu() {
      try {
        const man = await lireManifeste();
        const v = man && man.apercus && man.apercus[MODULE];
        if (!v || fini() || hote.dataset.ecran !== "demarrage") return;
        const rep = await fetch(new URL(`../moteur/apercus/${MODULE}.json?v=${encodeURIComponent(v)}`, import.meta.url));
        if (!rep.ok) return;
        const r = await rep.json();
        if (fini() || hote.dataset.ecran !== "demarrage" || !r || !r.ok) return;
        peindreApercu(r);
        lecture.reposer();
        hote.dataset.ecran = "apercu";
      } catch (e) { /* sans aperçu, l'écran attend la première réponse */ }
    }
  }
}
