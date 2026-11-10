// La SONDE du démarrage : une MESURE, jamais active par défaut. « ?sonde=1 » dans l'adresse d'un établi date chaque
// étape du démarrage — marques performance.mark « stab:<étape> », sur la page (js/etabli.js, js/moteur.js) et dans le
// travailleur (js/travailleur.js, qui reçoit « sonde=1 » dans sa propre adresse) — et sépare, dans le travailleur,
// l'import du module de son premier calcul, sans rien ajouter au chemin du premier rendu.
//
// Sans « ?sonde=1 », rien ne change : aucune marque, aucun message de plus entre la page et le travailleur, aucun
// travail Python de plus, la même adresse du travailleur, le même écran. C'est contrôlé dans le vrai navigateur
// (site/web/verifier_site.js, filtre « sonde ») et ce module est la SEULE porte : performance.mark ne s'écrit nulle part
// ailleurs dans js/ (tests/test_site_web.py). Les chiffres versés : docs/site/mesures/.
//
// L'adresse est lue au chargement du module, avant que l'établi ne la remplace par « ?note=… ». Dans l'application
// (js/app.js : les écrans se succèdent sans recharger le document), elle vaut pour toute la session du document.
//
// La sonde PAR OUVERTURE (lot S4, REFONTE_ETABLI.md) : un document sert plusieurs écrans, et le moteur reste démarré.
// Les marques gardent leurs noms (« stab:requete1_envoyee »…) mais valent pour l'ouverture COURANTE : js/app.js appelle
// nouvelleOuverture(n) avant de monter un écran, ce qui efface les marques « stab: » de la page ; les messages
// « sonde » du travailleur, eux, s'accumulent dans globalThis.__sondes (js/moteur.js), qui garde l'historique.
export const SONDE = new URL(self.location.href).searchParams.get("sonde") === "1";
export const PREFIXE = "stab:";

// une marque, si la sonde est active
export function marquer(nom) {
  if (SONDE) performance.mark(PREFIXE + nom);
}

// la première fois seulement (la première requête, la première réponse…)
export function marquerUneFois(nom) {
  if (SONDE && !performance.getEntriesByName(PREFIXE + nom, "mark").length) performance.mark(PREFIXE + nom);
}

// la fin d'une promesse, marquée — sans la sonde, la MÊME promesse est rendue (pas même un « then » de plus)
export function finir(nom, promesse) {
  return SONDE ? promesse.then((x) => { marquer(nom); return x; }) : promesse;
}

// les marques posées ici, en temps ABSOLU (timeOrigin + startTime) : la page et le travailleur n'ont pas la même origine
export function marquesAbsolues() {
  const o = performance.timeOrigin;
  return performance.getEntriesByType("mark").filter((m) => m.name.startsWith(PREFIXE))
    .map((m) => [m.name.slice(PREFIXE.length), +(o + m.startTime).toFixed(1)]);
}

// une nouvelle ouverture d'écran (js/app.js) : les marques « stab: » de la page sont effacées — celles de l'ouverture
// suivante reprennent les mêmes noms ; globalThis.__sondes (l'historique des messages du travailleur) est gardé
export function nouvelleOuverture(n) {
  if (!SONDE) return;
  for (const m of performance.getEntriesByType("mark")) if (m.name.startsWith(PREFIXE)) performance.clearMarks(m.name);
  globalThis.__ouverture = n;
}

// le travailleur (js/travailleur.js) : les marques d'une REQUÊTE (« requete1_debut »…), effacées avant la première
// requête de chaque nouvelle clé — elles gardent leurs noms et valent pour cette clé-là
export function effacerMarques(noms) {
  if (SONDE) for (const nom of noms) performance.clearMarks(PREFIXE + nom);
}
