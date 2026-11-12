# Les plans PDF faits dans le navigateur par le moteur de RBAR — mesure du 12/11/2026 (soir)

Mesure datée (elle ne se réécrit pas ; une nouvelle mesure fait un nouveau fichier). Suite de
`2026-11-12_plans_pdf_par_rbar_plans.md` : RBAR a livré le même soir la version navigateur de son moteur.

## Conditions

- **Le moteur** : RBAR 0.4.38 pour le navigateur (WebAssembly), archive `rbar-plans-web-0.4.38.zip` transmise par le
  propriétaire, empreinte SHA-256 `77c2da23…f5e5bf` (celle que RBAR a annoncée) ; déposé par
  `site/web/outils/vendre_rbar_plans.py` : 36 fichiers, 15,14 Mio, le plus gros 6,12 Mio (`_framework/Rbar.Core.wasm`) ;
  chaque fichier a l'empreinte de la notice du paquet et celle que le chargeur de .NET vérifie.
- **Le site** : construit au commit qui livre ce lot, servi par `site/web/outils/serveur_local.py` (les en-têtes et la
  politique de sécurité de Cloudflare reproduits) ; Chromium (Playwright), sans limitation de réseau.
- **Le parcours** (harnais `site/web/verifier_site.js`, filtre `exporter`) : une note neuve des fondations de grues
  (chantier « Greenwood », nom « Fondation grue tour GT1 ») ; menu « Exporter » ouvert (le moteur de RBAR se charge
  alors) ; clic sur « Plan de coffrage (PDF) », puis sur « Plan d'armatures (PDF) » ; durée du clic au fichier reçu.
  Trois formats d'écran — PC 1 440 × 900, tablette portrait 820 × 1 180, téléphone 390 × 844 (tactile) —, tous sur le
  processeur du même poste de contrôle.

## Résultats

| Format | Premier plan (coffrage) | Second plan (armatures) |
|---|---|---|
| PC | 6 021 ms | 115 ms |
| Tablette portrait | 5 506 ms | 97 ms |
| Téléphone | 6 106 ms | 110 ms |

- Le premier plan comprend : le fichier « rbar.ouvrage » écrit par le moteur Python, la fin du chargement du moteur de
  RBAR (commencé à l'ouverture du menu), le calcul des deux feuilles et leurs PDF. Le second sort du même calcul.
- Deux vrais PDF (en-tête `%PDF-1.`), plus de 20 Ko chacun, aux noms « F1 - plan de coffrage.pdf » et « F1 - plan
  d'armatures.pdf » ; le message à l'écran dit le fichier et la durée ; aucune erreur JavaScript, aucun refus.
- En tablette portrait, le menu « Exporter » (huit lignes) ne tient pas entier dans la colonne courte des résultats : il
  s'ouvre vers le haut et défile (le contrôle atteint chaque ligne d'un bord à l'autre).

## Ce que cette mesure ne dit pas

- Les trois formats ont tourné sur le processeur du poste de contrôle : ce n'est **pas** la durée sur une vraie tablette
  ou un vrai téléphone, plus lents. Le message à l'écran donne la durée mesurée sur l'appareil même : c'est là qu'on la
  lira.
- La mémoire prise par le moteur n'est pas mesurée (le navigateur ne la donne pas à la page sans isolement).
