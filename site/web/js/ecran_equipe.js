// La page ÉQUIPE (docs/site/PROJETS_EQUIPE.md § 4.3) : qui je suis sur ce poste (mon nom, mon e-mail, mes initiales dans
// leur pastille), les membres du bureau, le journal d'activité entier. Se présenter prend une ligne : le prénom et le
// nom — les initiales sont proposées (LLE pour Louis Lejeune) et se corrigent, de 1 à 5 lettres.
//
// Un ÉCRAN MONTABLE (comme js/accueil.js) : aucun effet au chargement ; js/app.js appelle ouvrirEquipe(hote, ctx). Il
// n'attend pas le moteur et n'écrit jamais l'historique.
import * as N from "./notes.js";
import * as E from "./equipe.js";
import * as J from "./journal.js";
import * as V from "./vues_bureau.js";
import * as VF from "./verifications.js";
import { marquer } from "./marques.js";
import { esc, toast, cablerBientot } from "./commun.js";

export const JOURNAL_VU = 120;                   // les lignes du journal montrées (les plus récentes)

export function ouvrirEquipe(hote, ctx) {
  const { signal } = ctx;
  const $ = (id) => document.getElementById(id);
  let filtre = null;                             // l'identifiant d'un membre : le journal ne montre que ses lignes
  let enEdition = null;                          // l'identifiant du membre dont la ligne est ouverte en saisie
  const teintes = (m) => `<div class="bu-teintes" role="group" aria-label="La teinte de la pastille">${Array.from({ length: E.COULEURS }, (_, c) =>
    `<button type="button" class="avatar sm c${c}${m.couleur === c ? " on" : ""}" data-teinte="${c}" title="Cette teinte" aria-pressed="${m.couleur === c}"></button>`).join("")}</div>`;

  // ------------------------------------------------------------------ moi
  function afficherMoi(eq) {
    const z = $("eq-moi"), moi = eq.moi;
    if (!moi) {
      const autres = eq.membres.length ? `<label class="bu-jesuis"><span>Vous êtes déjà dans la liste ?</span><select id="eq-jesuis" aria-label="Je suis…"><option value="">Je suis…</option>`
        + `${eq.membres.map((m) => `<option value="${esc(m.id)}">${esc(m.nom)} (${esc(m.initiales)})</option>`).join("")}</select></label>` : "";
      z.innerHTML = V.formulaireQui("Qui êtes-vous ?", "Présentez-vous une fois sur ce poste : vos initiales signeront vos notes et vos gestes dans le journal.") + autres;
      V.cablerQui(z, { signal, apres: () => afficher() });
      return;
    }
    z.innerHTML = `<div class="bu-profil">${E.pastille(moi, "xl")}<div class="bu-profil-c"><div class="kicker">Vous, sur ce poste</div>`
      + `<div class="bu-profil-l"><label class="bu-ch bu-ch-nom"><span>Prénom et nom</span><input class="field-in" data-moi="nom" value="${esc(moi.nom)}" maxlength="80" autocomplete="name"></label>`
      + `<label class="bu-ch bu-ch-ini"><span>Initiales</span><input class="field-in" data-moi="initiales" value="${esc(moi.initiales)}" maxlength="5" autocapitalize="characters" spellcheck="false"></label>`
      + `<label class="bu-ch bu-ch-mail"><span>Adresse e-mail</span><input class="field-in" data-moi="email" type="email" value="${esc(moi.email)}" maxlength="120" placeholder="facultatif" autocomplete="email"></label></div>`
      + `<div class="bu-profil-l bu-profil-t"><span class="muted">Teinte</span>${teintes(moi)}<span class="spacer"></span>`
      + `<label class="bu-jesuis court"><span>Ce n'est pas vous ?</span><select id="eq-jesuis" aria-label="Changer de personne sur ce poste">`
      + `${eq.membres.map((m) => `<option value="${esc(m.id)}"${m.id === moi.id ? " selected" : ""}>${esc(m.nom)}</option>`).join("")}<option value="-">Personne (se présenter)</option></select></label></div>`
      + `<p class="hint bu-erreur" id="eq-moi-erreur" role="alert" hidden></p></div></div>`;
  }

  // ------------------------------------------------------------------ les membres
  function afficherMembres(eq) {
    const n = eq.membres.length;
    $("eq-titre").textContent = n ? `Membres · ${n}` : "Membres";
    $("eq-nb").textContent = n ? `${V.compter(n, "membre")} sur ce poste` : "";
    // le rôle : « Administrateur » (il nomme les autres et vérifie les modules) ou « Membre » ; seul un administrateur le change
    const admin = E.estAdmin(eq.moi);
    const role = (m) => (m.role === "admin" ? `<span class="role admin">${V.icone("shield-check")}${esc(E.ROLES.admin)}</span>` : `<span class="role membre">${esc(E.ROLES.membre)}</span>`);
    const ligne = (m) => {
      const moi = eq.moi && eq.moi.id === m.id;
      if (enEdition === m.id) {
        return `<tr data-membre="${esc(m.id)}" class="bu-edit"><td colspan="4"><form class="bu-ligne-f" data-edit="${esc(m.id)}" autocomplete="off">${E.pastille(m)}`
          + `<input class="field-in" name="nom" value="${esc(m.nom)}" maxlength="80" aria-label="Prénom et nom">`
          + `<input class="field-in bu-ini-in" name="initiales" value="${esc(m.initiales)}" maxlength="5" aria-label="Initiales" autocapitalize="characters" spellcheck="false">`
          + `<input class="field-in" name="email" type="email" value="${esc(m.email)}" maxlength="120" placeholder="adresse e-mail" aria-label="Adresse e-mail">`
          + (admin ? `<select class="field-in bu-role-in" name="role" aria-label="Rôle">${Object.entries(E.ROLES).map(([k, nom]) => `<option value="${k}"${m.role === k ? " selected" : ""}>${esc(nom)}</option>`).join("")}</select>` : "")
          + `<button type="submit" class="btn sm primary">${V.icone("check")}Enregistrer</button><button type="button" class="btn sm" data-annuler="1">Annuler</button></form></td></tr>`;
      }
      return `<tr data-membre="${esc(m.id)}"><td><div class="who2">${E.pastille(m)}<div><b>${esc(m.nom)}${moi ? ' <span class="tag pub">vous</span>' : ""}`
        + `${m.role === "admin" ? `<span class="bu-role-m" title="${esc(E.ROLES.admin)}">${V.icone("shield-check")}</span>` : ""}</b>`
        + `<span>${m.email ? esc(m.email) : "pas d'adresse e-mail"}</span></div></div></td><td class="bu-mr">${role(m)}</td>`
        + `<td class="bu-mi"><span class="bu-ini-l">${esc(m.initiales)}</span></td>`
        + `<td class="n"><span class="bu-actions">`
        + `<button type="button" class="btn icon ghost sm${filtre === m.id ? " on" : ""}" data-voir="${esc(m.id)}" title="Voir son activité dans le journal" aria-label="Voir l'activité de ${esc(m.nom)}" aria-pressed="${filtre === m.id}">${V.icone("clock")}</button>`
        + `<button type="button" class="btn icon ghost sm" data-modifier="${esc(m.id)}" title="Modifier la fiche" aria-label="Modifier la fiche de ${esc(m.nom)}">${V.icone("pencil-line")}</button>`
        + `<button type="button" class="btn icon ghost sm" data-retirer="${esc(m.id)}" title="Retirer de l'équipe" aria-label="Retirer ${esc(m.nom)} de l'équipe">${V.icone("trash-2")}</button></span></td></tr>`;
    };
    $("eq-membres").innerHTML = n ? `<div class="tscroll"><table class="table bu-membres"><thead><tr><th>Personne</th><th>Rôle</th><th>Initiales</th><th class="n"></th></tr></thead><tbody>${eq.membres.map(ligne).join("")}</tbody></table></div>`
      : '<p class="muted bu-vide">Personne encore : présentez-vous ci-dessus, puis ajoutez vos collègues.</p>';
  }

  // ------------------------------------------------------------------ le journal
  function afficherJournal(eq) {
    const qui = filtre ? eq.membres.find((m) => m.id === filtre) : null;
    if (filtre && !qui) filtre = null;
    $("eq-filtre").innerHTML = qui ? `<button type="button" class="btn sm soft" data-voir="${esc(qui.id)}" title="Montrer tout le journal">${E.pastille(qui, "sm")}${esc(V.prenom(qui.nom))}${V.icone("x")}</button>` : "";
    $("eq-journal").innerHTML = V.journalHTML(J.lire({ membre: qui ? qui.id : null, max: JOURNAL_VU }), { moi: eq.moi, notes: N.toutes(), jours: true,
      vide: qui ? `${V.prenom(qui.nom)} n'a encore rien fait sur ce poste.` : "Le journal se remplira avec le travail : une note créée, rangée, exportée, un projet créé…" });
  }

  // ------------------------------------------------------------------ la vérification des modules
  // Chaque module de calcul : vérifié (par qui, quand), à vérifier, ou à REVÉRIFIER (son calcul a changé depuis). Un
  // administrateur le déclare vérifié — ou retire sa vérification ; les autres lisent.
  function afficherModules(eq) {
    const admin = E.estAdmin(eq.moi);
    const tous = VF.tous();
    const faits = tous.filter((m) => m.etat === "verifie").length;
    $("eq-verif-nb").textContent = `${faits} vérifié${faits > 1 ? "s" : ""} sur ${tous.length}`;
    const ligne = (m) => {
      const d = VF.dire(m.cle, m);
      // (vérifié : l'écusson ARGENTÉ, un administrateur ; DORÉ, deux administrateurs différents — leurs deux noms)
      const etat = m.etat === "verifie" ? `<span class="valid ok" data-niveau="${esc(m.niveau)}">${V.icone("badge-check")}${esc(d.court)}</span>`
        : `<span class="valid non${m.etat === "change" ? " change" : ""}">${V.icone("shield-check")}${esc(d.court)}</span>`;
      const signes = m.etat === "verifie" ? m.pars : m.par ? [m.par] : [];
      const noms = signes.map((p) => esc(p.nom));
      const qui = m.etat === "non" ? "" : `${signes.map((p) => E.pastille(p, "sm")).join("")}<span class="muted" title="${esc(d.long)}">`
        + `${noms.length > 1 ? `${noms.slice(0, -1).join(", ")} et ${noms[noms.length - 1]}` : noms[0]} · ${esc(d.qui.split(" · ")[1])}`
        + `${m.etat === "change" ? " · le calcul a changé depuis" : ""}</span>`;
      const retirer = `<button type="button" class="btn sm ghost" data-retirer-verif="${esc(m.cle)}" aria-label="Retirer la vérification du module ${esc(m.nom)}">Retirer</button>`;
      const poser = (mot, dire) => `<button type="button" class="btn sm" data-verifier="${esc(m.cle)}" aria-label="${dire} le module ${esc(m.nom)}">${V.icone("badge-check")}${mot}</button>`;
      // un administrateur : « Vérifier » ; « Vérifier aussi » s'il n'a pas encore signé une marque argentée (elle deviendra dorée) ;
      // « Retirer » dès qu'elle est posée
      const geste = !admin ? "" : m.etat !== "verifie" ? poser(m.etat === "change" ? "Revérifier" : "Vérifier", "Déclarer vérifié")
        : m.niveau === "argent" && !VF.aSigne(m, eq.moi) ? poser("Vérifier aussi", "Vérifier à votre tour") + " " + retirer : retirer;
      return `<tr data-module-verif="${esc(m.cle)}" data-etat="${m.etat}"><td class="bu-vm"><a class="mod bu-lien" href="${esc(N.pageDe(m.cle))}"><span class="modic ${esc(m.famille)}">${V.iconeModule(m.cle)}</span><b>${esc(m.nom)}</b></a></td>`
        + `<td class="bu-ve">${etat}</td><td class="bu-vq"><span class="bu-vqui">${qui}</span></td><td class="n bu-vg">${geste}</td></tr>`;
    };
    $("eq-modules").innerHTML = `<div class="tscroll"><table class="table bu-verif"><thead><tr><th>Module</th><th>État</th><th>Par · le</th><th class="n"></th></tr></thead>`
      + `<tbody>${tous.map(ligne).join("")}</tbody></table></div>`
      + `<p class="hint bu-verif-note">${admin ? "Vous êtes administrateur : « Vérifier » dit que vous avez relu la méthode du module. "
        : eq.moi ? "Seul un administrateur déclare un module vérifié. " : "Présentez-vous pour savoir si vous pouvez vérifier un module. "}`
      + `La marque est argentée ; elle devient dorée quand un deuxième administrateur vérifie le module à son tour. `
      + `Elle tombe d'elle-même si le calcul du module change.</p>`;
  }

  function afficher() {
    const eq = E.equipe();
    afficherMoi(eq);
    afficherMembres(eq);
    afficherModules(eq);
    afficherJournal(eq);
    marquer(document);                          // la pastille du poste, les modules vérifiés du volet
    return eq;
  }

  // ------------------------------------------------------------------ les gestes
  const page = hote.querySelector(".bu-team");
  // ma fiche : chaque champ s'enregistre quand on en sort (ou à Entrée) ; un refus dit pourquoi et rend l'ancienne valeur
  page.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && ev.target.matches && ev.target.matches("[data-moi]")) { ev.preventDefault(); ev.target.blur(); }
  }, { signal });
  page.addEventListener("change", (ev) => {
    const t = ev.target, moi = E.moi();
    if (t.matches("[data-moi]") && moi) {
      const r = E.modifier(moi.id, { [t.dataset.moi]: t.value });
      const err = $("eq-moi-erreur");
      if (!r.ok) { err.textContent = r.erreur; err.hidden = false; t.value = moi[t.dataset.moi]; t.focus(); return; }
      toast("Votre fiche est enregistrée.");
      afficher();
    } else if (t.matches("#eq-jesuis")) {
      E.choisirMoi(t.value && t.value !== "-" ? t.value : null);
      afficher();
    }
  }, { signal });
  page.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-teinte], [data-voir], [data-modifier], [data-retirer], [data-annuler], [data-verifier], [data-retirer-verif]");
    if (!b) return;
    const eq = E.equipe();
    if ("verifier" in b.dataset || "retirerVerif" in b.dataset) {
      const cle = b.dataset.verifier || b.dataset.retirerVerif, nom = (N.MODULES[cle] || {}).nom || cle;
      const poser = "verifier" in b.dataset, aussi = poser && VF.etat(cle).etat === "verifie";
      if (!confirm(aussi ? `Vérifier AUSSI le module « ${nom} » ?\n\nVous dites en avoir relu la méthode, à votre tour. Avec deux administrateurs, la marque devient dorée et porte vos deux noms ; elle tombera d'elle-même si son calcul change.`
        : poser ? `Déclarer le module « ${nom} » VÉRIFIÉ ?\n\nVous dites en avoir relu la méthode. Votre nom et la date s'afficheront sur l'accueil et dans le module ; la marque tombera d'elle-même si son calcul change.`
        : `Retirer la vérification du module « ${nom} » ? Il redeviendra « à vérifier ».`)) return;
      const r = poser ? VF.verifier(cle) : VF.retirer(cle);
      if (!r.ok) { toast(r.erreur, true); return; }
      toast(!poser ? `Le module « ${nom} » est de nouveau à vérifier.` : r.niveau === "or" ? `Module « ${nom} » vérifié deux fois : la marque est dorée.` : `Module « ${nom} » vérifié.`);
      afficher();
      return;
    }
    if ("teinte" in b.dataset && eq.moi) { E.modifier(eq.moi.id, { couleur: Number(b.dataset.teinte) }); afficher(); return; }
    if ("voir" in b.dataset) { filtre = filtre === b.dataset.voir ? null : b.dataset.voir; afficherMembres(eq); afficherJournal(eq); return; }
    if ("annuler" in b.dataset) { enEdition = null; afficherMembres(eq); return; }
    if ("modifier" in b.dataset) {
      enEdition = b.dataset.modifier;
      afficherMembres(eq);
      const c = page.querySelector(`form[data-edit="${CSS.escape(enEdition)}"] [name="nom"]`);
      if (c) { c.focus(); c.select(); }
      return;
    }
    if ("retirer" in b.dataset) {
      const m = eq.membres.find((x) => x.id === b.dataset.retirer);
      if (!m) return;
      const moi = eq.moi && eq.moi.id === m.id;
      if (!confirm(`Retirer ${m.nom} de l'équipe de ce poste ? Ses notes et ses lignes du journal gardent ses initiales.${moi ? "\n\nC'est vous : il faudra vous présenter à nouveau." : ""}`)) return;
      if (!E.retirer(m.id)) { toast("L'équipe n'a pas pu être enregistrée dans ce navigateur.", true); return; }
      if (!moi) J.noter("membre.retire", { membre: m.nom });
      toast(`${m.nom} ne fait plus partie de l'équipe de ce poste.`);
      afficher();
    }
  }, { signal });
  page.addEventListener("submit", (ev) => {
    const f = ev.target.closest("form[data-edit]");
    if (!f) return;
    ev.preventDefault();
    const c = (n) => f.querySelector(`[name="${n}"]`).value;
    const r = E.modifier(f.dataset.edit, { nom: c("nom"), initiales: c("initiales"), email: c("email") });
    if (!r.ok) { toast(r.erreur, true); const x = f.querySelector(`[name="${r.champ}"]`); if (x) x.focus(); return; }
    // le rôle (le menu n'existe que pour un administrateur) : un refus — le dernier administrateur — se dit, la fiche reste
    const choisi = f.querySelector('[name="role"]');
    if (choisi && choisi.value !== r.membre.role) {
      const n = E.nommer(f.dataset.edit, choisi.value);
      if (!n.ok) { toast(n.erreur, true); afficher(); return; }
      J.noter(choisi.value === "admin" ? "membre.admin" : "membre.membre", { membre: r.membre.nom });
    }
    enEdition = null;
    toast(`La fiche de ${r.membre.nom} est enregistrée.`);
    afficher();
  }, { signal });

  // ajouter un collègue : son nom ; ses initiales sont proposées, son adresse est facultative
  const fiche = V.cablerFiche({ nom: $("eq-nom"), initiales: $("eq-ini"), email: $("eq-email"), erreur: $("eq-erreur"), signal,
                                prises: () => E.equipe().membres.map((m) => m.initiales) });
  $("eq-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const r = E.ajouter(fiche.lire());
    if (!r.ok) { fiche.erreur(r.erreur, r.champ); return; }
    fiche.erreur();
    fiche.vider();
    J.noter("membre.ajoute", { membre: r.membre.nom });
    toast(`${r.membre.nom} (${r.membre.initiales}) est dans l'équipe.`);
    afficher();
    $("eq-nom").focus();
  }, { signal });
  cablerBientot(signal);

  afficher();
  ctx.titre("Équipe · Stab Méthodes");
  if (typeof ctx.rendreDefilement === "function") ctx.rendreDefilement();
  // prêt dès qu'il est monté : cet écran n'attend pas le moteur
  hote.dataset.ecran = "pret";
  return { fermer() { return { note: null, enregistree: Promise.resolve({ refus: [] }) }; } };
}
