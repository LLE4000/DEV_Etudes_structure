// « Enregistrer comme note… » (l'établi compact, pilote « Poutres béton armé » ; construire.compact écrit la fenêtre).
//
// Le calcul s'ouvre directement depuis sa carte, sans fenêtre « Nouvelle note » : la note est PROVISOIRE (js/etabli.js)
// tant que rien n'est modifié. Ce bouton lui donne un nom, un chantier et un dossier, et l'écrit dans ce navigateur. Une
// fois écrite, le même bouton la CLASSE (changer son nom, son chantier, son dossier). Les chantiers et les dossiers déjà
// connus (les notes de ce navigateur) sont proposés ; on peut aussi en taper un nouveau.
import * as N from "./notes.js";
import { toast } from "./commun.js";

export function cablerRangement({ note, provisoire, apres }) {
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
      b.title = p ? "Nommer cette note, la ranger dans un chantier et l'enregistrer dans ce navigateur"
        : "Changer le nom, le chantier ou le dossier de la note";
    });
    $("et-rg-titre").textContent = p ? "Enregistrer comme note" : "Classer la note";
  }

  // les valeurs déjà connues, sans doublon, dans l'ordre alphabétique (les notes abîmées sont ignorées)
  function connues(cle, filtre = () => true) {
    let notes = [];
    try { notes = Object.values(N.toutes() || {}); } catch (e) { /* mémoire illisible : rien à proposer */ }
    const v = new Set(notes.filter((x) => x && typeof x[cle] === "string" && x[cle].trim() && filtre(x)).map((x) => x[cle].trim()));
    return [...v].sort((a, b) => a.localeCompare(b, "fr"));
  }
  function proposer(liste, valeurs) {
    liste.replaceChildren(...valeurs.map((v) => { const o = document.createElement("option"); o.value = v; return o; }));
  }
  const proposerDossiers = () => {
    const c = champs.chantier.value.trim();
    proposer($("et-rg-dossiers"), connues("dossier", (x) => !c || (x.chantier || "").trim() === c));
  };

  function ouvrir(depuis) {
    retour = depuis || null;
    libeller();
    champs.nom.value = note().nom || "";
    champs.chantier.value = note().chantier || "";
    champs.dossier.value = note().dossier || "";
    proposer($("et-rg-chantiers"), connues("chantier"));
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
  });
  fen.addEventListener("click", (ev) => {
    if (ev.target === fen || ev.target.closest("[data-fermer-rangement]")) fermer();
  });
  fen.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { ev.stopPropagation(); fermer(); } });
  champs.chantier.addEventListener("input", proposerDossiers);
  $("et-rg-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const n = note();
    const etait = provisoire();
    n.nom = champs.nom.value.trim() || N.MODULES[n.module].nom;
    n.chantier = champs.chantier.value.trim();
    n.dossier = champs.dossier.value.trim();
    const ok = apres();
    fermer();
    // un échec d'écriture est déjà dit par l'indicateur et son message (js/etabli.js) : pas de « enregistrée » ici
    if (!ok) return;
    const ou = [n.chantier, n.dossier].filter(Boolean).join(" › ");
    toast(`${etait ? "Note enregistrée" : "Note classée"}${ou ? ` dans ${ou}` : " (sans chantier)"}`);
  });
  libeller();
  return { libeller };
}
