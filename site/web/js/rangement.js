// « Enregistrer comme note… » (toute page d'établi depuis le 10/11/2026 — c'était l'établi compact ; construire.bouton_ranger,
// entree_ranger et fenetre_rangement écrivent le bouton, l'entrée du menu ⋯ et la fenêtre).
//
// Le calcul s'ouvre directement depuis sa carte, sans fenêtre « Nouvelle note » : la note est PROVISOIRE (js/etabli.js)
// tant que rien n'est modifié. Ce bouton lui donne un nom, un PROJET et un SOUS-PROJET (les champs « chantier » et
// « dossier » de la note), et l'écrit dans ce navigateur. Une fois écrite, le même bouton la CLASSE (changer son nom, son
// projet, son sous-projet). Les projets et les sous-projets déjà connus — ceux de la page des projets, js/projets.js :
// les mêmes listes partout — sont proposés ; on peut aussi en taper un nouveau.
import * as N from "./notes.js";
import * as P from "./projets.js";
import { toast } from "./commun.js";

// ``signal`` : l'AbortSignal du montage de l'écran (js/app.js) — ses écouteurs tombent à la fermeture
export function cablerRangement({ note, provisoire, apres, signal }) {
  const fen = document.getElementById("et-rangement");
  const boutons = [...document.querySelectorAll("[data-ranger]")];
  if (!fen) return { libeller() {} };
  const $ = (id) => document.getElementById(id);
  const champs = { nom: $("et-rg-nom"), chantier: $("et-rg-chantier"), dossier: $("et-rg-dossier") };
  let retour = null;   // le bouton qui a ouvert la fenêtre : il reprend le focus à la fermeture

  function libeller() {
    const p = provisoire();
    boutons.forEach((b) => {
      const l = b.querySelector(".lbl, .et-mi-ranger");
      if (l) l.textContent = p ? "Enregistrer comme note…" : "Classer la note…";
      b.title = p ? "Nommer cette note, la ranger dans un projet et l'enregistrer dans ce navigateur"
        : "Changer le nom, le projet ou le sous-projet de la note";
    });
    $("et-rg-titre").textContent = p ? "Enregistrer comme note" : "Classer la note";
  }

  // les projets connus (ceux du registre et ceux que les notes nomment, hors archives), dans l'ordre alphabétique ; une
  // mémoire illisible : rien à proposer
  function modele() {
    try { return P.modele(); } catch (e) { return { projets: [] }; }
  }
  function proposer(liste, valeurs) {
    liste.replaceChildren(...valeurs.map((v) => { const o = document.createElement("option"); o.value = v; return o; }));
  }
  // les sous-projets du projet écrit (aucun projet écrit : ceux de tous les projets)
  const proposerDossiers = () => {
    const m = modele(), p = P.trouver(champs.chantier.value, m);
    const sous = (p ? [p] : champs.chantier.value.trim() ? [] : m.projets).flatMap((x) => x.sous.map((s) => s.nom));
    proposer($("et-rg-dossiers"), [...new Set(sous)].sort((a, b) => a.localeCompare(b, "fr")));
  };

  function ouvrir(depuis) {
    retour = depuis || null;
    libeller();
    champs.nom.value = note().nom || "";
    champs.chantier.value = note().chantier || "";
    champs.dossier.value = note().dossier || "";
    proposer($("et-rg-chantiers"), modele().projets.filter((x) => !x.archive).map((x) => x.nom).sort((a, b) => a.localeCompare(b, "fr")));
    proposerDossiers();
    fen.hidden = false;
    champs.nom.focus();
    champs.nom.select();
  }
  function fermer() {
    if (fen.hidden) return;
    fen.hidden = true;
    if (retour && retour.isConnected && !retour.closest("[hidden]")) retour.focus();
  }

  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-ranger]");
    if (b) ouvrir(b);
  }, { signal });
  fen.addEventListener("click", (ev) => {
    if (ev.target === fen || ev.target.closest("[data-fermer-rangement]")) fermer();
  }, { signal });
  fen.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { ev.stopPropagation(); fermer(); } }, { signal });
  champs.chantier.addEventListener("input", proposerDossiers, { signal });
  $("et-rg-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const n = note();
    const etait = provisoire();
    n.nom = champs.nom.value.trim() || N.MODULES[n.module].nom;
    // le projet et le sous-projet s'écrivent comme ils existent déjà (« réalex » rejoint « Réalex »)
    const p = P.trouver(champs.chantier.value);
    n.chantier = p ? p.nom : P.nomPropre(champs.chantier.value);
    const s = p && p.sous.find((x) => x.cle === P.cle(champs.dossier.value));
    n.dossier = s ? s.nom : P.nomPropre(champs.dossier.value);
    const ok = apres();
    fermer();
    // un échec d'écriture est déjà dit par l'indicateur et son message (js/etabli.js) : pas de « enregistrée » ici
    if (!ok) return;
    const ou = [n.chantier, n.dossier].filter(Boolean).join(" › ");
    toast(`${etait ? "Note enregistrée" : "Note classée"}${ou ? ` dans ${ou}` : " (sans projet)"}`);
  }, { signal });
  libeller();
  return { libeller };
}
