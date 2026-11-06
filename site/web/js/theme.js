// Le thème choisi (clair ou sombre), posé AVANT l'affichage pour éviter un éclair de la mauvaise couleur. Un fichier
// et non un script en ligne : la politique de sécurité du site (_headers) n'autorise aucun script écrit dans la page.
try {
  const t = localStorage.getItem("stab.theme");
  if (t) document.documentElement.dataset.theme = t;
} catch (e) { /* stockage bloqué : le thème du système s'applique */ }
