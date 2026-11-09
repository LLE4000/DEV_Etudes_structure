// L'accueil : reprendre une note, en créer une (chantier, dossier, nom, module), voir les chantiers. Les notes viennent
// de ce navigateur (notes.js) ; rien n'est inventé : sans note, l'accueil le dit.
import * as N from "./notes.js";
import { cablerRail, esc, toast, texteTaux, telecharger } from "./commun.js";

cablerRail();
const $ = (s) => document.querySelector(s);
const jour = new Date().toLocaleDateString("fr-BE", { weekday: "long", day: "numeric", month: "long" });

// La mémoire des notes ABÎMÉE (notes.js : « illisible ») : l'accueil le dit — ses notes ne peuvent pas s'afficher — et
// offre son texte brut, tel quel ; de même pour chaque copie d'une mémoire illisible gardée à part avant une écriture.
const fr = (n) => n.toLocaleString("fr-BE");
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
});

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
    const chip = tx.texte ? `<span class="chip ${esc(r.niveau)}"${tx.titre ? ` title="${esc(tx.titre)}"` : ""}>${esc(tx.texte)}</span>` : "";
    const lieu = [n.chantier, n.dossier].filter(Boolean).join(" › ") || "Sans chantier";
    return `<a class="card rcard" href="${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}"><div class="thumb">${/^\s*<svg/.test(n.vignette || "") ? n.vignette : ""}</div>`
      + `<div class="body"><div class="t" title="${esc(n.nom)}">${esc(n.nom)}</div>`
      + `<div class="m"><span class="ou" title="${esc(N.MODULES[n.module].nom)} · ${esc(lieu)}">${esc(N.MODULES[n.module].nom)} · ${esc(lieu)}</span></div>`
      + `<div class="m">${esc(N.quand(n.modifie))}${chip}<span class="supp" data-supprimer="${esc(n.id)}" title="Supprimer la note">✕</span></div></div></a>`;
  }).join("") : m.etat === "illisible" ? `<div class="vide"><b>Aucune note lisible</b>La mémoire des notes de ce navigateur est illisible (voir plus haut).</div>`
    : `<div class="vide"><b>Aucune note pour l'instant</b>« Nouvelle note » en crée une ; elle s'enregistre dans ce navigateur.</div>`;
  // les chantiers : leurs dossiers et leurs notes
  const ch = {};
  notes.forEach((n) => {
    const c = n.chantier || "Sans chantier";
    (ch[c] = ch[c] || { dossiers: new Set(), notes: 0 }).notes++;
    if (n.dossier) ch[c].dossiers.add(n.dossier);
  });
  const noms = Object.keys(ch).sort((a, b) => a.localeCompare(b, "fr"));
  $("#chantiers").innerHTML = noms.length ? noms.map((c) => `<div class="chantier"><div class="t"><b>${esc(c)}</b>`
    + `<span>${ch[c].dossiers.size} dossier${ch[c].dossiers.size > 1 ? "s" : ""}</span></div><span class="nb">${ch[c].notes} note${ch[c].notes > 1 ? "s" : ""}</span></div>`).join("")
    : '<p class="muted">Les chantiers apparaissent avec les notes.</p>';
  $("#nn-chantiers").innerHTML = noms.filter((c) => c !== "Sans chantier").map((c) => `<option value="${esc(c)}">`).join("");
}
afficher();

$("#recentes").addEventListener("click", (ev) => {
  const s = ev.target.closest("[data-supprimer]");
  if (!s) return;
  ev.preventDefault();
  if (confirm("Supprimer cette note de ce navigateur ? (pensez à l'exporter en .stab si vous voulez la garder)")) {
    if (!N.supprimer(s.dataset.supprimer)) toast(`Suppression impossible : ${(N.RAISONS[N.bilan().raison] || N.RAISONS.indisponible).long}.`, true);
    afficher();
  }
});

// « Nouvelle note » : chantier, dossier, nom, module → l'établi du module
const modal = $("#nouvelle");
let module = null;
function ouvrirNouvelle(cle = null) {
  module = cle;
  modal.querySelectorAll(".mopt").forEach((o) => o.classList.toggle("on", o.dataset.module === cle));
  $("#nn-creer").disabled = !module;
  modal.hidden = false;
  $("#nn-chantier").focus();
}
$("#ouvrir-nouvelle").addEventListener("click", () => ouvrirNouvelle());
modal.addEventListener("click", (ev) => {
  if (ev.target === modal || ev.target.closest("[data-fermer-modal]")) { modal.hidden = true; return; }
  const o = ev.target.closest(".mopt");
  if (o) { module = o.dataset.module; modal.querySelectorAll(".mopt").forEach((x) => x.classList.toggle("on", x === o)); $("#nn-creer").disabled = false; }
});
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") modal.hidden = true; });
$("#nn-creer").addEventListener("click", () => {
  const n = N.nouvelle({ module, nom: $("#nn-nom").value.trim(), chantier: $("#nn-chantier").value.trim(), dossier: $("#nn-dossier").value.trim() });
  // la note non enregistrée (stockage plein, mémoire illisible) : l'établi la recrée de l'adresse, et y dit l'échec
  const ou = N.ecrire(n) ? `?note=${encodeURIComponent(n.id)}`
    : `?${new URLSearchParams({ nom: $("#nn-nom").value.trim(), chantier: n.chantier, dossier: n.dossier })}`;
  location.href = `${N.pageDe(module)}${ou}`;
});
// une tuile de module ouvre la fenêtre avec ce module déjà choisi
document.querySelectorAll(".tile[data-nouvelle-module]").forEach((t) => t.addEventListener("click", (ev) => { ev.preventDefault(); ouvrirNouvelle(t.dataset.nouvelleModule); }));

$("#importer").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  try {
    const n = await N.importer(f);
    location.href = `${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}`;
  } catch (e) { alert(`Import impossible : ${e.message}`); }
});

// les entrées « bientôt » de la barre (Projets, Équipe, Paramètres) : elles viendront avec les comptes du bureau
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-bientot]");
  if (b) { ev.preventDefault(); toast("Bientôt : avec les comptes du bureau."); }
});
