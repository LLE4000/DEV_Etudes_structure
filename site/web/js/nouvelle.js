// La fenêtre « Nouvelle note » et l'import d'un fichier .stab : les mêmes à l'accueil et sur la page des projets
// (construire.fenetre_nouvelle_note écrit la fenêtre). Le projet, le sous-projet, le nom, puis le module → l'établi du
// module. Les projets et les sous-projets déjà connus sont proposés (js/projets.js) ; on peut en taper un nouveau.
//
// ``naviguer`` : la navigation de l'application (ctx.naviguer) ; ``signal`` : celui du montage de l'écran.
import * as N from "./notes.js";
import * as P from "./projets.js";

export function cablerNouvelleNote({ signal, naviguer }) {
  const modal = document.getElementById("nouvelle");
  const $ = (id) => document.getElementById(id);
  if (!modal) return { ouvrir() {}, fermer() {} };
  let module = null;
  const proposer = (liste, valeurs) => {
    liste.replaceChildren(...valeurs.map((v) => { const o = document.createElement("option"); o.value = v; return o; }));
  };
  const proposerSous = () => {
    const p = P.trouver($("nn-chantier").value);
    proposer($("nn-dossiers"), p ? p.sous.map((s) => s.nom) : []);
  };
  // ``cle`` : un module déjà choisi ; ``ou`` : { chantier, dossier } à préremplir (on crée depuis un projet)
  function ouvrir(cle = null, ou = null) {
    module = cle;
    modal.querySelectorAll(".mopt").forEach((o) => o.classList.toggle("on", o.dataset.module === cle));
    $("nn-creer").disabled = !module;
    if (ou) { $("nn-chantier").value = ou.chantier || ""; $("nn-dossier").value = ou.dossier || ""; }
    let m = { projets: [] };
    try { m = P.modele(); } catch (e) { /* mémoire illisible : rien à proposer */ }
    proposer($("nn-chantiers"), m.projets.filter((p) => !p.archive).map((p) => p.nom));
    proposerSous();
    modal.hidden = false;
    // on crée depuis un projet : il est déjà écrit, le nom de la note attend
    (ou && ou.chantier ? $("nn-nom") : $("nn-chantier")).focus();
  }
  const fermer = () => { modal.hidden = true; };
  $("nn-chantier").addEventListener("input", proposerSous, { signal });
  modal.addEventListener("click", (ev) => {
    if (ev.target === modal || ev.target.closest("[data-fermer-modal]")) { fermer(); return; }
    const o = ev.target.closest(".mopt");
    if (o) { module = o.dataset.module; modal.querySelectorAll(".mopt").forEach((x) => x.classList.toggle("on", x === o)); $("nn-creer").disabled = false; }
  }, { signal });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") fermer(); }, { signal });
  $("nn-creer").addEventListener("click", () => {
    // le projet et le sous-projet s'écrivent comme ils existent déjà (« réalex » rejoint « Réalex »)
    const p = P.trouver($("nn-chantier").value);
    const chantier = p ? p.nom : P.nomPropre($("nn-chantier").value);
    const s = p && p.sous.find((x) => x.cle === P.cle($("nn-dossier").value));
    const dossier = chantier ? (s ? s.nom : P.nomPropre($("nn-dossier").value)) : "";
    const n = N.nouvelle({ module, nom: $("nn-nom").value.trim(), chantier, dossier });
    // la note non enregistrée (stockage plein, mémoire illisible) : l'établi la recrée de l'adresse, et y dit l'échec
    const ou = N.ecrire(n) ? `?note=${encodeURIComponent(n.id)}`
      : `?${new URLSearchParams({ nom: $("nn-nom").value.trim(), chantier: n.chantier, dossier: n.dossier })}`;
    naviguer(`${N.pageDe(module)}${ou}`);
  }, { signal });
  return { ouvrir, fermer };
}

// L'import d'un fichier .stab (ou d'un enregistrement Streamlit) : la note s'enregistre, puis s'ouvre.
export function cablerImport({ signal, naviguer }) {
  const champ = document.getElementById("importer");
  if (!champ) return;
  champ.addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    try {
      const n = await N.importer(f);
      naviguer(`${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}`);
    } catch (e) { if (!signal.aborted) alert(`Import impossible : ${e.message}`); }
  }, { signal });
}
