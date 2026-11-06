// Ce que toutes les pages partagent : le thème (clair, sombre), le volet des modules, les messages éphémères.
export function cablerRail() {
  const racine = document.documentElement;
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest("[data-theme-toggle]");
    if (t) {
      const sombre = racine.dataset.theme === "dark" || (!racine.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      racine.dataset.theme = sombre ? "light" : "dark";
      try { localStorage.setItem("stab.theme", racine.dataset.theme); } catch (e) { /* navigation privée */ }
      return;
    }
    const m = ev.target.closest("[data-modules]");
    if (m) { document.body.classList.toggle("show-modules"); return; }
    if (!ev.target.closest(".modpanel")) document.body.classList.remove("show-modules");
  });
}

let minuterie = null;
export function toast(texte, erreur = false) {
  const t = document.getElementById("toast");
  if (!t) { if (erreur) alert(texte); return; }
  t.textContent = texte;
  t.classList.toggle("erreur", !!erreur);
  t.hidden = false;
  clearTimeout(minuterie);
  minuterie = setTimeout(() => { t.hidden = true; }, erreur ? 7000 : 2600);
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
