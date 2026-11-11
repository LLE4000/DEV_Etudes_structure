// L'accueil : reprendre une note, en créer une (projet, sous-projet, nom, module), voir et créer les PROJETS, lire
// l'ACTIVITÉ du bureau (docs/site/PROJETS_EQUIPE.md § 4.1). Les notes, les projets, l'équipe et le journal viennent de ce
// navigateur ; rien n'est inventé : sans note, l'accueil le dit. La carte d'un module ouvre le calcul lui-même (un lien :
// décision D1, ouvrir un calcul n'écrit rien) ; « Nouvelle note » nomme et range la note d'abord.
//
// Un ÉCRAN MONTABLE (lot S4, REFONTE_ETABLI.md § 3.1) : aucun effet au chargement du fichier ; js/app.js appelle
// ouvrirAccueil(hote, ctx) et garde le { fermer } rendu. Les écouteurs posés sur le document le sont avec le signal du
// montage ; les navigations (« Nouvelle note », l'import d'un .stab) passent par l'application (ctx.naviguer). L'accueil
// n'attend pas le moteur : il est prêt dès qu'il est monté (body[data-ecran] = pret).
import * as N from "./notes.js";
import * as P from "./projets.js";
import * as E from "./equipe.js";
import * as J from "./journal.js";
import * as V from "./vues_bureau.js";
import * as VF from "./verifications.js";
import { BIBLIOTHEQUES } from "./modules.js";
import { marquer } from "./marques.js";
import { cablerNouvelleNote, cablerImport } from "./nouvelle.js";
import { esc, toast, texteTaux, telecharger, cablerBientot } from "./commun.js";

export const PROJETS_VUS = 6;                    // l'accueil montre les six projets les plus récemment actifs
export const ACTIVITE_VUE = 6;                   // et les six dernières lignes du journal
export const RESULTATS_VUS = 8;                  // la recherche montre huit résultats au plus

// sans accents ni capitales : « realex » trouve « Réalex »
const plat = (t) => String(t || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("fr");
const ORDRE = { Module: 0, Bibliothèque: 1, Projet: 2, "Sous-projet": 3, Note: 4 };
const NOMS_FAMILLES = { beton: "Béton", acier: "Acier", autres: "Autres" };

// Ce que la recherche de l'accueil connaît : les modules (ceux qu'on voit), les bibliothèques, les projets et leurs
// sous-projets, les notes de ce navigateur. ``notes`` : la table des notes ; rend [{ sorte, nom, sous, href, module }].
export function indexRecherche(notes) {
  const out = [];
  for (const [k, m] of Object.entries(N.MODULES)) {
    if (!m.masque && !m.vers) out.push({ sorte: "Module", nom: m.nom, sous: NOMS_FAMILLES[m.famille] || "", href: m.page, module: k });
  }
  for (const b of Object.values(BIBLIOTHEQUES)) out.push({ sorte: "Bibliothèque", nom: `Bibliothèque ${b.titre}`, sous: "", href: b.page });
  const md = P.modele(notes);
  for (const p of md.projets) {
    out.push({ sorte: "Projet", nom: p.nom, sous: `${p.nb} note${p.nb > 1 ? "s" : ""}${p.archive ? " · archivé" : ""}`, href: `projets?${new URLSearchParams({ p: p.nom })}` });
    for (const s of p.sous) out.push({ sorte: "Sous-projet", nom: s.nom, sous: p.nom, href: `projets?${new URLSearchParams({ p: p.nom, s: s.nom })}` });
  }
  for (const n of Object.values(notes || {})) {
    if (!n || typeof n !== "object" || !Object.hasOwn(N.MODULES, n.module)) continue;
    const m = N.MODULES[n.module];
    out.push({ sorte: "Note", nom: n.nom || m.nom, sous: [m.nom, n.chantier, n.dossier].filter(Boolean).join(" · "), href: `${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}`,
               module: N.moduleServi(n.module) });
  }
  return out;
}

// Les résultats d'une recherche : chaque mot doit se trouver (dans le nom ou ce qui le situe) ; d'abord ce dont le NOM
// commence par la recherche, puis ce dont le nom la contient, puis le reste ; à rang égal : modules, bibliothèques,
// projets, sous-projets, notes ; puis l'ordre alphabétique. ``max`` résultats au plus.
export function chercher(index, question, max = RESULTATS_VUS) {
  const q = plat(question).trim();
  if (!q) return [];
  const mots = q.split(/\s+/);
  return index.map((x) => {
    const nom = plat(x.nom), tout = `${nom} ${plat(x.sous)} ${plat(x.sorte)}`;
    if (!mots.every((w) => tout.includes(w))) return null;
    return { ...x, rang: nom.startsWith(q) ? 0 : nom.includes(q) ? 1 : 2 };
  }).filter(Boolean).sort((a, b) => (a.rang - b.rang) || (ORDRE[a.sorte] - ORDRE[b.sorte]) || a.nom.localeCompare(b.nom, "fr")).slice(0, max);
}

const $ = (s) => document.querySelector(s);

// La mémoire des notes ABÎMÉE (notes.js : « illisible ») : l'accueil le dit — ses notes ne peuvent pas s'afficher — et
// offre son texte brut, tel quel ; de même pour chaque copie d'une mémoire illisible gardée à part avant une écriture.
const fr = (n) => n.toLocaleString("fr-BE");

// Une note dont la dernière saisie s'ENREGISTRE ENCORE (la modification gardée de l'écran qu'on vient de quitter, C1 ;
// ctx.enCours) : sa carte est NEUTRE — « enregistrement en cours », sans le taux ni la pastille de l'état d'AVANT (comme
// une note rouverte pendant le démarrage, C11) — et l'accueil se redessine quand ctx.apresEnregistrements() se résout. La
// carte montrait l'ancien taux et son verdict comme l'état courant, jusqu'à la visite suivante (revue S4, etat-01 :
// défaut non conservatif). Supprimer une telle note attend son enregistrement (sinon l'écriture tardive la faisait
// revenir, etat-03).
export const TEXTE_ENREGISTREMENT = "enregistrement en cours";

// ``hote`` : le corps de la page ; ``ctx`` : ce que l'application donne (js/app.js) — signal, naviguer, enCours,
// apresEnregistrements, attendreEnregistrement… Rend { fermer }.
export function ouvrirAccueil(hote, ctx) {
  const { signal } = ctx;
  const enCours = (id) => typeof ctx.enCours === "function" && !!ctx.enCours(id);
  const jour = new Date().toLocaleDateString("fr-BE", { weekday: "long", day: "numeric", month: "long" });
  function afficherMemoire(m) {
    const z = $("#et-memoire");
    const bouton = (attr, libelle) => `<button type="button" class="btn sm" ${attr}>${libelle}</button>`;
    const lignes = [];
    if (m.etat === "illisible") {
      lignes.push(`<div class="et-mem-l"><p><b>La mémoire des notes de ce navigateur est illisible</b> (abîmée) : ses notes ne peuvent pas `
        + `s'afficher. Son texte n'est pas effacé : avant toute nouvelle écriture, il est copié à part, tel quel — et si la copie échoue, rien `
        + `n'est écrit.</p>${bouton('data-texte-brut="1"', `Télécharger son texte (${fr(m.brut.length)} caractères)`)}</div>`);
    } else if (m.etat === "indisponible") {
      lignes.push(`<div class="et-mem-l"><p><b>Ce navigateur refuse l'accès à sa mémoire</b> (navigation privée, réglage du site) : les notes `
        + `ne peuvent ni s'afficher ni s'enregistrer. Exportez chaque calcul en fichier .stab.</p></div>`);
    }
    for (const c of N.copies()) {
      lignes.push(`<div class="et-mem-l"><p><b>Une mémoire des notes illisible est gardée à part</b>${c.quand ? ` (copiée ${esc(N.quand(c.quand))})` : ""} : `
        + `les notes qu'elle contenait ne s'affichent pas ici. Son texte, tel quel :</p>`
        + `${bouton(`data-copie="${esc(c.cle)}"`, `Télécharger (${fr(c.taille)} caractères)`)}</div>`);
    }
    z.innerHTML = lignes.join("");
    z.hidden = !lignes.length;
  }
  $("#et-memoire").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-texte-brut], [data-copie]");
    if (!b) return;
    const texte = b.dataset.copie ? N.texteCopie(b.dataset.copie) : N.memoire().brut;
    if (typeof texte !== "string") { toast("Ce texte n'est plus dans ce navigateur.", true); afficher(); return; }
    const date = new Date().toISOString().slice(0, 10);
    telecharger(`memoire-des-notes-illisible-${date}.txt`, texte, "text/plain;charset=utf-8");
  }, { signal });

  function afficher() {
    const m = N.memoire();
    afficherMemoire(m);
    const notes = Object.values(m.notes);
    $("#date").textContent = `${jour.charAt(0).toUpperCase() + jour.slice(1)} · `
      + (m.etat === "illisible" ? "mémoire des notes illisible" : `${notes.length} note${notes.length > 1 ? "s" : ""} dans ce navigateur`);
    const rec = N.recentes(6);
    $("#nb-notes").textContent = rec.length ? "les plus récentes" : "";
    $("#recentes").innerHTML = rec.length ? rec.map((n) => {
      const r = n.resume || {};
      // le taux est déjà arrondi par le moteur : on l'écrit à la française (63,7 et non 63.7) ; « > 999 % » au-delà
      const tx = texteTaux(r.taux);
      // (une saisie s'enregistre encore : ni l'ancien taux, ni sa pastille — la pastille neutre le dit)
      const vol = enCours(n.id);
      const chip = vol ? `<span class="chip c et-enr-chip" title="La dernière saisie de cette note s'enregistre">${TEXTE_ENREGISTREMENT}</span>`
        : tx.texte ? `<span class="chip ${esc(r.niveau)}"${tx.titre ? ` title="${esc(tx.titre)}"` : ""}>${esc(tx.texte)}</span>` : "";
      const lieu = [n.chantier, n.dossier].filter(Boolean).join(" › ") || "Sans projet";
      // la pastille de qui y a travaillé en dernier (rien tant que personne ne s'est présenté)
      const par = n.par || n.auteur ? E.pastille(n.par || n.auteur, "sm") : "";
      return `<a class="card rcard${vol ? " et-enr-cours" : ""}"${vol ? ' aria-busy="true"' : ""} href="${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}"><div class="thumb">${/^\s*<svg/.test(n.vignette || "") ? n.vignette : ""}</div>`
        + `<div class="body"><div class="t" title="${esc(n.nom)}">${esc(n.nom)}</div>`
        + `<div class="m"><span class="ou" title="${esc(N.MODULES[n.module].nom)} · ${esc(lieu)}">${esc(N.MODULES[n.module].nom)} · ${esc(lieu)}</span></div>`
        + `<div class="m">${par}${esc(N.quand(n.modifie))}${chip}<span class="supp" data-supprimer="${esc(n.id)}" title="Supprimer la note">✕</span></div></div></a>`;
    }).join("") : m.etat === "illisible" ? `<div class="vide"><b>Aucune note lisible</b>La mémoire des notes de ce navigateur est illisible (voir plus haut).</div>`
      : `<div class="vide"><b>Aucune note pour l'instant</b>La carte d'un module ouvre un calcul : il devient une note, enregistrée dans ce navigateur, `
        + `à la première saisie (« Nouvelle note » la nomme d'abord).</div>`;
    afficherProjets(m.notes);
    afficherActivite(m.notes);
  }

  // Les PROJETS : les plus récemment actifs (chacun mène à sa page), puis les notes sans projet ; « Tout voir » mène à
  // la page des projets. Un projet se crée d'ici : son nom dans le champ, Entrée.
  function afficherProjets(notes) {
    const md = P.modele(notes);
    const actifs = md.projets.filter((p) => !p.archive);
    const lignes = actifs.slice(0, PROJETS_VUS).map((p) => `<a class="site" href="projets?${new URLSearchParams({ p: p.nom })}">`
      + `<span class="ic">${V.icone("folder")}</span><div><b>${esc(p.nom)}</b><span>${p.sous.length ? V.compter(p.sous.length, "sous-projet", "sous-projets") : "sans sous-projet"}</span></div>`
      + `${V.pile(p.auteurs)}<span class="nb">${V.compter(p.nb, "note")}</span></a>`);
    if (md.sans.length) lignes.push(`<a class="site sans" href="projets?sans=1"><span class="ic">${V.icone("file-plus-2")}</span><div><b>Sans projet</b><span>à ranger</span></div><span class="nb">${V.compter(md.sans.length, "note")}</span></a>`);
    if (actifs.length > PROJETS_VUS) lignes.push(`<a class="bu-plus" href="projets">et ${V.compter(actifs.length - PROJETS_VUS, "autre projet", "autres projets")}</a>`);
    $("#chantiers").innerHTML = lignes.length ? lignes.join("")
      : '<p class="muted bu-vide">Aucun projet pour l\'instant : écrivez le nom d\'un chantier ci-dessus, puis Entrée.</p>';
  }

  // L'ACTIVITÉ : les dernières lignes du journal ; et, tant que personne ne s'est présenté sur ce poste, « Qui êtes-vous ? »
  function afficherActivite(notes) {
    const moi = E.moi();
    $("#moi").innerHTML = E.pastille(moi, "lg");
    $("#bonjour").textContent = moi ? `Bonjour ${V.prenom(moi.nom)}` : "Bonjour";
    const qui = $("#ac-qui");
    if (moi) qui.innerHTML = "";
    else if (!qui.querySelector("form")) { qui.innerHTML = V.formulaireQui(); V.cablerQui(qui, { signal, apres: () => afficher() }); }
    $("#activite").innerHTML = V.journalHTML(J.lire({ max: ACTIVITE_VUE }), { moi, notes,
      vide: "Le journal se remplira avec le travail : une note créée, rangée, exportée…" });
    // la pastille du poste dans la barre suit (on vient peut-être de se présenter) ; les modules vérifiés aussi
    marquer(document);
  }

  // ---------------------------------------------------------------- la recherche : un module, un projet, une note
  // On écrit ; les résultats viennent à mesure ; ↓ ↑ choisissent, Entrée ouvre, Échap referme ; Ctrl K y mène.
  const recherche = { champ: $("#ac-recherche"), liste: $("#ac-resultats"), vus: [], choix: 0 };
  function afficherRecherche() {
    const { champ, liste } = recherche;
    const q = champ.value;
    recherche.vus = q.trim() ? chercher(indexRecherche(N.memoire().notes), q) : [];
    recherche.choix = Math.min(recherche.choix, Math.max(0, recherche.vus.length - 1));
    const verifies = new Set(VF.tous().filter((m) => m.etat === "verifie").map((m) => m.cle));
    const icone = (x) => (x.module ? `<span class="modic ${esc(N.MODULES[x.module].famille)}">${V.iconeModule(x.module)}</span>`
      : `<span class="ic">${V.icone(x.sorte === "Bibliothèque" ? "book-open" : "folder")}</span>`);
    liste.innerHTML = recherche.vus.map((x, i) => `<a class="bu-res${i === recherche.choix ? " on" : ""}" role="option" aria-selected="${i === recherche.choix}" href="${esc(x.href)}" data-rang="${i}">`
      + `${icone(x)}<span class="bu-res-t"><b>${esc(x.nom)}</b><span>${esc([x.sorte, x.sous].filter(Boolean).join(" · "))}</span></span>`
      + `${x.sorte === "Module" && verifies.has(x.module) ? `<span class="ok" title="Module vérifié">${V.icone("badge-check")}</span>` : ""}</a>`).join("")
      || (q.trim() ? `<p class="muted bu-vide">Rien ne porte ce nom : ni module, ni projet, ni note.</p>` : "");
    liste.hidden = !q.trim();
    champ.setAttribute("aria-expanded", String(!liste.hidden));
  }
  const fermerRecherche = () => { recherche.liste.hidden = true; recherche.champ.setAttribute("aria-expanded", "false"); };
  recherche.champ.addEventListener("input", () => { recherche.choix = 0; afficherRecherche(); }, { signal });
  recherche.champ.addEventListener("focus", () => { if (recherche.champ.value.trim()) afficherRecherche(); }, { signal });
  recherche.champ.addEventListener("keydown", (ev) => {
    const n = recherche.vus.length;
    if (ev.key === "ArrowDown" && n) { ev.preventDefault(); recherche.choix = (recherche.choix + 1) % n; afficherRecherche(); }
    else if (ev.key === "ArrowUp" && n) { ev.preventDefault(); recherche.choix = (recherche.choix + n - 1) % n; afficherRecherche(); }
    else if (ev.key === "Escape") { if (recherche.champ.value) { recherche.champ.value = ""; afficherRecherche(); } else recherche.champ.blur(); }
  }, { signal });
  $("#ac-recherche-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const x = recherche.vus[recherche.choix];
    if (x) ctx.naviguer(x.href);
  }, { signal });
  document.addEventListener("click", (ev) => { if (!ev.target.closest(".bu-cherche")) fermerRecherche(); }, { signal });
  document.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && !ev.altKey && !ev.shiftKey && ev.key.toLowerCase() === "k") { ev.preventDefault(); recherche.champ.focus(); recherche.champ.select(); }
  }, { signal });
  V.cablerCreer("ac-projet", { signal, valider: (nom) => {
    const r = P.creer(nom);
    if (!r.ok) { toast(r.erreur, true); return false; }
    toast(r.existe ? `Le projet « ${r.nom} » existe déjà.` : `Projet « ${r.nom} » créé.`);
    afficher();
    return true;
  } });
  afficher();
  // les enregistrements gardés encore en vol (les cartes neutres) : l'accueil se redessine quand ils sont faits — et
  // encore, s'il en reste (un enregistrement qui en suit un autre)
  const suivre = () => {
    if (typeof ctx.apresEnregistrements !== "function" || !N.recentes(6).some((n) => enCours(n.id))) return;
    ctx.apresEnregistrements().then(() => { if (!signal.aborted) { afficher(); suivre(); } });
  };
  suivre();

  $("#recentes").addEventListener("click", (ev) => {
    const s = ev.target.closest("[data-supprimer]");
    if (!s) return;
    ev.preventDefault();
    if (confirm("Supprimer cette note de ce navigateur ? (pensez à l'exporter en .stab si vous voulez la garder)")) {
      const id = s.dataset.supprimer;
      // la dernière saisie de cette note s'enregistre peut-être encore (l'écran qu'on vient de quitter) : on l'attend,
      // puis on supprime — la suppression est le dernier mot (le choix confirmé tient, même si l'accueil est quitté)
      const p = typeof ctx.attendreEnregistrement === "function" ? Promise.resolve(ctx.attendreEnregistrement(id)).catch(() => {}) : Promise.resolve();
      p.then(() => {
        const ok = N.supprimer(id);
        if (signal.aborted) return;
        if (!ok) toast(`Suppression impossible : ${(N.RAISONS[N.bilan().raison] || N.RAISONS.indisponible).long}.`, true);
        afficher();
      });
    }
  }, { signal });

  // « Nouvelle note » (projet, sous-projet, nom, module → l'établi du module) et l'import d'un .stab : js/nouvelle.js
  const nouvelle = cablerNouvelleNote({ signal, naviguer: ctx.naviguer });
  $("#ouvrir-nouvelle").addEventListener("click", () => nouvelle.ouvrir(), { signal });
  cablerImport({ signal, naviguer: ctx.naviguer });
  cablerBientot(signal);

  // prêt dès qu'il est monté : l'accueil n'attend pas le moteur
  hote.dataset.ecran = "pret";
  return { fermer() { nouvelle.fermer(); return { note: null, enregistree: Promise.resolve({ refus: [] }) }; } };
}
