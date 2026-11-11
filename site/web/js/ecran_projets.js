// La page PROJETS (docs/site/PROJETS_EQUIPE.md § 4.2) : l'ARBRE des projets et de leurs sous-projets, les NOTES du projet
// choisi, son ACTIVITÉ. Un projet se crée en écrivant son nom — le chantier — dans le champ de l'arbre, puis Entrée ; un
// sous-projet, de même, dans le projet. Rien d'autre n'est demandé.
//
// Un ÉCRAN MONTABLE (comme js/accueil.js) : aucun effet au chargement ; js/app.js appelle ouvrirProjets(hote, ctx) et
// garde le { fermer } rendu. La sélection vit dans l'adresse (?p=<projet>&s=<sous-projet>, ?sans=1) : un nœud de l'arbre
// est un LIEN (l'application navigue, « retour » revient) ; renommer réécrit l'adresse de l'écran (ctx.adresse). Cet écran
// n'écrit jamais l'historique lui-même. Il n'attend pas le moteur.
import * as N from "./notes.js";
import * as P from "./projets.js";
import * as E from "./equipe.js";
import * as J from "./journal.js";
import * as V from "./vues_bureau.js";
import { cablerNouvelleNote, cablerImport } from "./nouvelle.js";
import { esc, toast, cablerBientot } from "./commun.js";

export const ACTIVITE_PROJET = 40;               // les lignes du journal montrées pour un projet

export function ouvrirProjets(hote, ctx) {
  const { signal } = ctx;
  const $ = (id) => document.getElementById(id);
  const sel = { p: P.cle(ctx.params.get("p")), s: P.cle(ctx.params.get("s")), sans: ctx.params.get("sans") === "1" };
  let filtre = "";
  const adresseDe = (p = "", s = "") => `projets${p ? `?${new URLSearchParams(s ? { p, s } : { p })}` : ""}`;
  const quandLong = (t) => new Date(t).toLocaleDateString("fr-BE", { day: "numeric", month: "long", year: "numeric" });

  // ------------------------------------------------------------------ l'arbre
  function arbre(md, projet) {
    const f = P.cle(filtre);
    const voit = (p) => !f || p.cle.includes(f) || p.sous.some((s) => s.cle.includes(f));
    const noeud = (p) => {
      const on = !!projet && projet.cle === p.cle;
      let h = `<a class="node chantier${on && !sel.s ? " on" : ""}${on ? " ouvert" : ""}" href="${esc(adresseDe(p.nom))}" title="${esc(p.nom)}">${V.icone(on ? "folder-open" : "folder")}`
        + `<span class="bu-nn">${esc(p.nom)}</span><span class="count">${p.nb}</span></a>`;
      if (on || (f && p.sous.some((s) => s.cle.includes(f)))) {
        h += p.sous.map((s) => `<a class="node l1${on && sel.s === s.cle ? " on" : ""}" href="${esc(adresseDe(p.nom, s.nom))}" title="${esc(s.nom)}">`
          + `<span class="bu-nn">${esc(s.nom)}</span><span class="count">${s.notes.length}</span></a>`).join("");
      }
      return h;
    };
    const actifs = md.projets.filter((p) => !p.archive), archives = md.projets.filter((p) => p.archive);
    let h = "";
    // le champ porte un nom que personne ne porte : Entrée (ou ce bouton) crée le projet
    if (f && !md.projets.some((p) => p.cle === f)) {
      h += `<button type="submit" form="pj-champ-form" class="node bu-creer-n">${V.icone("plus")}<span class="bu-nn">Créer « ${esc(P.nomPropre(filtre))} »</span></button>`;
    }
    h += `<a class="node${!projet && !sel.sans ? " on" : ""}" href="projets">${V.icone("layers")}<span class="bu-nn">Tous les projets</span><span class="count">${actifs.length}</span></a>`;
    h += actifs.filter(voit).map(noeud).join("");
    if (md.sans.length && !f) h += `<a class="node${sel.sans ? " on" : ""}" href="projets?sans=1">${V.icone("file-text")}<span class="bu-nn">Sans projet</span><span class="count">${md.sans.length}</span></a>`;
    if (archives.some(voit)) h += `<div class="navlabel">Archives</div>${archives.filter(voit).map(noeud).join("")}`;
    if (f && !md.projets.some(voit)) h += '<p class="hint bu-rien">Aucun projet de ce nom : Entrée le crée.</p>';
    $("pj-noeuds").innerHTML = h;
  }

  // ------------------------------------------------------------------ le menu « ranger » d'une note
  function menuRanger(n, md, projet) {
    const ici = projet ? `<optgroup label="Dans ce projet"><option value="s:">Hors sous-projet</option>${projet.sous.map((s) => `<option value="s:${esc(s.nom)}">${esc(s.nom)}</option>`).join("")}</optgroup>` : "";
    const autres = md.projets.filter((p) => !p.archive && (!projet || p.cle !== projet.cle));
    const ailleurs = autres.length ? `<optgroup label="${projet ? "Dans un autre projet" : "Dans un projet"}">${autres.map((p) => `<option value="p:${esc(p.nom)}">${esc(p.nom)}</option>`).join("")}</optgroup>` : "";
    return `<select class="bu-ranger" data-ranger="${esc(n.id)}" aria-label="Ranger ou supprimer la note ${esc(n.nom || "")}"><option value="">Ranger…</option>${ici}${ailleurs}`
      + `${projet ? '<option value="p:">Sans projet</option>' : ""}<option value="x">Supprimer la note…</option></select>`;
  }

  // ------------------------------------------------------------------ le centre
  function tete(kicker, titre, sous, droite = "") {
    return `<div class="head"><div class="bu-tete"><div class="kicker">${kicker}</div>${titre}<div class="muted bu-sous">${sous}</div></div><span class="spacer"></span>${droite}</div>`;
  }

  function vueTous(md) {
    const actifs = md.projets.filter((p) => !p.archive);
    const t = tete("Le bureau", '<h1 class="h1">Projets</h1>', `${V.compter(actifs.length, "projet")} · ${V.compter(md.nbNotes, "note")}`);
    if (!md.projets.length) {
      return `${t}<div class="vide bu-depart"><b>Aucun projet pour l'instant</b>Écrivez le nom d'un chantier dans le champ « Chercher ou créer un projet », puis Entrée.`
        + `<span>Un projet peut avoir des sous-projets ; les notes de calcul s'y rangent.</span></div>`;
    }
    const carte = (p) => `<a class="card bu-carte${p.archive ? " archive" : ""}" href="${esc(adresseDe(p.nom))}"><div class="bu-carte-h"><span class="ic">${V.icone("folder")}</span><b title="${esc(p.nom)}">${esc(p.nom)}</b></div>`
      + `<div class="muted">${p.sous.length ? `${V.compter(p.sous.length, "sous-projet", "sous-projets")} · ` : ""}${V.compter(p.nb, "note")}</div>`
      + `<div class="bu-carte-f">${V.pile(p.auteurs)}<span class="when">${p.derniere ? esc(J.ilYA(p.derniere)) : "pas encore de note"}</span></div></a>`;
    const archives = md.projets.filter((p) => p.archive);
    return `${t}<div class="bu-cartes">${actifs.map(carte).join("")}</div>`
      + (archives.length ? `<div class="kicker bu-k">Archives</div><div class="bu-cartes">${archives.map(carte).join("")}</div>` : "");
  }

  function vueSans(md) {
    return `<a class="bu-retour" href="projets">${V.icone("chevron-left")}Projets</a>`
      + tete("Notes", '<h1 class="h1">Sans projet</h1>', `${V.compter(md.sans.length, "note")} à ranger : le menu « Ranger… » de chaque note la met dans un projet`)
      + (md.sans.length ? V.tableNotes(md.sans, { ranger: (n) => menuRanger(n, md, null) }) : '<p class="muted bu-vide">Toutes les notes ont un projet.</p>');
  }

  function vueProjet(md, p) {
    const sousVus = sel.s ? p.sous.filter((s) => s.cle === sel.s) : p.sous;
    const faits = [V.compter(p.nb, "note"), p.sous.length ? V.compter(p.sous.length, "sous-projet", "sous-projets") : "", p.derniere ? `dernière note ${J.ilYA(p.derniere)}` : ""].filter(Boolean).join(" · ");
    const menu = `<details class="bu-menu"><summary class="btn icon" title="Archiver ou supprimer le projet" aria-label="Autres gestes sur le projet">${V.icone("ellipsis")}</summary><div class="bu-menu-c">`
      + `<button type="button" data-archiver="${p.archive ? "0" : "1"}">${V.icone("archive")}${p.archive ? "Rouvrir le projet" : "Archiver le projet"}</button>`
      + `<button type="button" data-supprimer-projet="1">${V.icone("trash-2")}Supprimer le projet</button></div></details>`;
    let h = `<a class="bu-retour" href="projets">${V.icone("chevron-left")}Projets</a>`
      + tete(`Projet${p.archive ? " · archivé" : ""}`,
        `<input class="h1 bu-titre" id="pj-nom" value="${esc(p.nom)}" maxlength="${P.NOM_MAX}" aria-label="Nom du projet : le chantier (modifiable)" title="Cliquer pour renommer le projet" spellcheck="false">`,
        esc(faits), `<button type="button" class="btn primary" data-note-ici="">${V.icone("plus")}Note</button>${menu}`);
    if (sel.s) h += `<a class="bu-tout" href="${esc(adresseDe(p.nom))}">${V.icone("chevron-left")}Tout le projet</a>`;
    // les notes rangées dans le projet lui-même, puis chaque sous-projet
    if (!sel.s && p.notes.length) {
      h += `<section class="bu-groupe"><header class="bu-gh"><b class="bu-gt">Notes du projet</b><span class="count">${V.compter(p.notes.length, "note")}</span></header>`
        + `${V.tableNotes(p.notes, { ranger: (n) => menuRanger(n, md, p) })}</section>`;
    }
    h += sousVus.map((s) => `<section class="bu-groupe" data-sous="${esc(s.cle)}"><header class="bu-gh">${V.icone("folder")}`
      + `<input class="bu-sous-nom" value="${esc(s.nom)}" data-sous-nom="${esc(s.nom)}" maxlength="${P.NOM_MAX}" aria-label="Nom du sous-projet (modifiable)" title="Cliquer pour renommer le sous-projet" spellcheck="false">`
      + `<span class="count">${V.compter(s.notes.length, "note")}</span><span class="spacer"></span>`
      + `<button type="button" class="btn sm" data-note-ici="${esc(s.nom)}">${V.icone("plus")}Note</button>`
      + (s.notes.length ? "" : `<button type="button" class="btn icon ghost sm" data-supprimer-sous="${esc(s.nom)}" title="Supprimer ce sous-projet (il est vide)" aria-label="Supprimer le sous-projet ${esc(s.nom)}">${V.icone("trash-2")}</button>`)
      + `</header>${s.notes.length ? V.tableNotes(s.notes, { ranger: (n) => menuRanger(n, md, p) }) : '<p class="muted bu-vide">Aucune note dans ce sous-projet : « + Note » en crée une ici.</p>'}</section>`).join("");
    if (!sel.s) {
      if (!p.nb && !p.sous.length) h += '<p class="muted bu-vide">Ce projet est vide : « + Note » y crée une note de calcul ; un sous-projet (ci-dessous) les regroupe.</p>';
      h += `<form class="bu-creer bu-creer-sous" data-creer-sous="1" autocomplete="off"><div class="field text">${V.icone("plus")}`
        + `<input name="nom" maxlength="${P.NOM_MAX}" placeholder="Nouveau sous-projet : son nom, puis Entrée" aria-label="Nom du nouveau sous-projet" enterkeyhint="done"></div>`
        + `<button type="submit" class="btn sm primary">Créer</button></form>`;
    }
    return h;
  }

  // ------------------------------------------------------------------ le volet : les faits, l'activité
  function volet(md, projet, moi, notes) {
    if (!projet) {
      $("pj-volet").innerHTML = `<div class="h3">Activité du bureau</div><div class="bu-fil">${V.journalHTML(J.lire({ max: ACTIVITE_PROJET }), { moi, notes, jours: true,
        vide: "Le journal se remplira avec le travail : une note créée, rangée, exportée…" })}</div>`;
      return;
    }
    const faits = [["Notes", String(projet.nb)], ["Sous-projets", String(projet.sous.length)]];
    if (projet.cree) faits.push(["Créé le", esc(quandLong(projet.cree))]);
    if (projet.par) faits.push(["Par", `${E.pastille(projet.par, "sm")} ${esc(projet.par.nom)}`]);
    if (projet.auteurs.length) faits.push(["Y ont travaillé", V.pile(projet.auteurs, 6)]);
    $("pj-volet").innerHTML = `<div class="h3">Le projet</div><dl class="facts">${faits.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`
      + `<div class="h3">Activité</div><div class="bu-fil">${V.journalHTML(J.lire({ projet: projet.nom, max: ACTIVITE_PROJET }), { moi, notes, jours: true, lieu: "sous",
        vide: "Rien encore dans ce projet." })}</div>`;
  }

  function afficher() {
    const mem = N.memoire();
    const md = P.modele(mem.notes), moi = E.moi();
    const projet = sel.p ? md.projets.find((p) => p.cle === sel.p) || null : null;
    const sous = projet && sel.s ? projet.sous.find((s) => s.cle === sel.s) || null : null;
    if (projet && sel.s && !sous) sel.s = "";
    arbre(md, projet);
    $("pj-liste").innerHTML = (mem.etat === "illisible" ? '<div class="vide"><b>La mémoire des notes de ce navigateur est illisible</b>Les notes ne peuvent pas s\'afficher (l\'accueil en dit plus).</div>' : "")
      + (sel.sans ? vueSans(md) : projet ? vueProjet(md, projet)
        : (sel.p ? `<div class="vide bu-absent"><b>Ce projet n'existe pas (ou plus) dans ce navigateur</b>Choisissez-en un dans la liste, ou créez-le.</div>` : "") + vueTous(md));
    volet(md, projet, moi, mem.notes);
    const chevron = V.icone("chevron-right");
    $("pj-fil").innerHTML = `${V.icone("folder")}${projet || sel.sans ? '<a href="projets">Projets</a>' : "<b>Projets</b>"}`
      + (sel.sans ? `${chevron}<b>Sans projet</b>` : "")
      + (projet ? `${chevron}${sous ? `<a href="${esc(adresseDe(projet.nom))}">${esc(projet.nom)}</a>${chevron}<b>${esc(sous.nom)}</b>` : `<b>${esc(projet.nom)}</b>`}` : "");
    $("pj").dataset.vue = projet || sel.sans ? "liste" : "arbre";
    ctx.titre(`${projet ? `${projet.nom} · ` : sel.sans ? "Sans projet · " : ""}Projets · Stab Méthodes`);
    return { md, projet };
  }

  // ------------------------------------------------------------------ les gestes
  const dit = (r, bien) => { if (!r.ok) toast(r.erreur, true); else if (bien) toast(bien); return r.ok; };
  const projetCourant = () => (sel.p ? P.trouver(sel.p) : null);
  const nouvelle = cablerNouvelleNote({ signal, naviguer: ctx.naviguer });
  const ouvrirNote = (sous = "") => {
    const p = projetCourant();
    const s = p && sel.s && !sous ? p.sous.find((x) => x.cle === sel.s) : null;
    nouvelle.ouvrir(null, p ? { chantier: p.nom, dossier: sous || (s ? s.nom : "") } : null);
  };
  $("ouvrir-nouvelle").addEventListener("click", () => ouvrirNote(), { signal });
  cablerImport({ signal, naviguer: ctx.naviguer });
  cablerBientot(signal);

  // le champ de l'arbre : il filtre à la frappe ; Entrée ouvre le projet de ce nom, ou le crée
  V.cablerCreer("pj-champ", { signal,
    saisie: (t) => { filtre = t; const mem = N.memoire(); const md = P.modele(mem.notes); arbre(md, sel.p ? md.projets.find((p) => p.cle === sel.p) || null : null); },
    valider: (nom) => {
      const r = P.creer(nom);
      if (!dit(r, r.existe ? "" : `Projet « ${r.nom} » créé.`)) return false;
      ctx.naviguer(adresseDe(r.nom));
      return true;
    } });

  const pj = $("pj");
  pj.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-note-ici], [data-archiver], [data-supprimer-projet], [data-supprimer-sous]");
    // un clic hors d'un menu ouvert le referme
    pj.querySelectorAll("details.bu-menu[open]").forEach((d) => { if (!d.contains(ev.target)) d.open = false; });
    if (!b) return;
    const p = projetCourant();
    if ("noteIci" in b.dataset) { ouvrirNote(b.dataset.noteIci); return; }
    if (!p) return;
    if ("archiver" in b.dataset) {
      if (dit(P.archiver(p.nom, b.dataset.archiver === "1"), b.dataset.archiver === "1" ? `Projet « ${p.nom} » archivé : ses notes restent.` : `Projet « ${p.nom} » rouvert.`)) afficher();
    } else if ("supprimerProjet" in b.dataset) {
      if (p.nb) { toast(`Ce projet a encore ${V.compter(p.nb, "note")} : rangez-les ailleurs ou supprimez-les d'abord (on ne perd jamais une note en supprimant un projet).`, true); return; }
      if (confirm(`Supprimer le projet « ${p.nom} » ? (il est vide : aucune note n'est perdue)`) && dit(P.supprimer(p.nom), `Projet « ${p.nom} » supprimé.`)) ctx.naviguer("projets");
    } else if ("supprimerSous" in b.dataset) {
      if (dit(P.supprimerSous(p.nom, b.dataset.supprimerSous), `Sous-projet « ${b.dataset.supprimerSous} » supprimé.`)) { if (sel.s === P.cle(b.dataset.supprimerSous)) { sel.s = ""; ctx.adresse(adresseDe(p.nom)); } afficher(); }
    }
  }, { signal });

  // renommer : le projet (son titre), un sous-projet (son nom) — on écrit par-dessus, Entrée ou sortir du champ valide
  pj.addEventListener("keydown", (ev) => {
    if (!ev.target.matches || !ev.target.matches("#pj-nom, .bu-sous-nom")) return;
    if (ev.key === "Enter") { ev.preventDefault(); ev.target.blur(); }
    if (ev.key === "Escape") { ev.target.value = ev.target.matches("#pj-nom") ? (projetCourant() || { nom: "" }).nom : ev.target.dataset.sousNom; ev.target.blur(); }
  }, { signal });
  pj.addEventListener("change", (ev) => {
    const t = ev.target;
    const p = projetCourant();
    if (t.matches("#pj-nom") && p) {
      const r = P.renommer(p.nom, t.value);
      if (!r.ok) { toast(r.erreur, true); t.value = p.nom; return; }
      if (r.inchange) { t.value = p.nom; return; }
      const s = sel.s ? p.sous.find((x) => x.cle === sel.s) : null;
      sel.p = P.cle(r.nom);
      ctx.adresse(adresseDe(r.nom, s ? s.nom : ""));
      toast(`Projet renommé « ${r.nom} »${p.nb ? ` : ${V.compter(p.nb, "note suit", "notes suivent")}` : ""}.`);
      afficher();
    } else if (t.matches(".bu-sous-nom") && p) {
      const avant = t.dataset.sousNom;
      const r = P.renommerSous(p.nom, avant, t.value);
      if (!r.ok) { toast(r.erreur, true); t.value = avant; return; }
      if (r.inchange) { t.value = avant; return; }
      if (sel.s === P.cle(avant)) { sel.s = P.cle(r.nom); ctx.adresse(adresseDe(p.nom, r.nom)); }
      toast(`Sous-projet renommé « ${r.nom} ».`);
      afficher();
    } else if (t.matches(".bu-ranger")) {
      const id = t.dataset.ranger, v = t.value;
      t.value = "";
      if (!v) return;
      if (v === "x") {
        const n = N.lire(id);
        if (n && confirm(`Supprimer la note « ${n.nom || "sans nom"} » de ce navigateur ? (pensez à l'exporter en .stab si vous voulez la garder)`)) {
          // sa dernière saisie s'enregistre peut-être encore (l'écran qu'on vient de quitter) : on l'attend, puis on supprime
          const att = typeof ctx.attendreEnregistrement === "function" ? Promise.resolve(ctx.attendreEnregistrement(id)).catch(() => {}) : Promise.resolve();
          att.then(() => {
            const ok = N.supprimer(id);
            if (signal.aborted) return;
            if (!ok) toast(`Suppression impossible : ${(N.RAISONS[N.bilan().raison] || N.RAISONS.indisponible).long}.`, true);
            afficher();
          });
        }
        return;
      }
      const r = v.startsWith("s:") ? P.deplacer(id, p ? p.nom : "", v.slice(2)) : P.deplacer(id, v.slice(2), "");
      const ou = v.startsWith("s:") ? (v.slice(2) || "le projet, hors sous-projet") : (v.slice(2) || "« Sans projet »");
      if (dit(r, `Note rangée dans ${ou}.`)) afficher();
    }
  }, { signal });
  pj.addEventListener("submit", (ev) => {
    const f = ev.target.closest("form[data-creer-sous]");
    if (!f) return;
    ev.preventDefault();
    const p = projetCourant(), champ = f.querySelector('[name="nom"]');
    if (!p || !champ.value.trim()) return;
    const r = P.creerSous(p.nom, champ.value);
    if (dit(r, r.existe ? `Le sous-projet « ${r.nom} » existe déjà.` : `Sous-projet « ${r.nom} » créé.`)) {
      afficher();
      const neuf = pj.querySelector("form[data-creer-sous] input");
      if (neuf) neuf.focus();                     // on en enchaîne plusieurs : le champ garde la main
    }
  }, { signal });

  const { md } = afficher();
  // un bureau sans projet, au PC : le champ attend déjà le nom du premier
  if (!md.projets.length && matchMedia("(min-width: 701px)").matches) $("pj-champ").focus({ preventScroll: true });
  if (typeof ctx.rendreDefilement === "function") ctx.rendreDefilement();
  // prêt dès qu'il est monté : cet écran n'attend pas le moteur
  hote.dataset.ecran = "pret";
  return { fermer() { nouvelle.fermer(); return { note: null, enregistree: Promise.resolve({ refus: [] }) }; } };
}
