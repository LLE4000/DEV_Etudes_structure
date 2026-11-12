# « Plans PDF » en un seul fichier — le moteur de RBAR 0.4.40 dans le navigateur — mesure du 12/11/2026 (soir)

Mesure datée (elle ne se réécrit pas ; une nouvelle mesure fait un nouveau fichier). Suite de
`2026-11-12_plans_pdf_dans_le_navigateur.md` (RBAR 0.4.38, une ligne par plan).

## Conditions

- **Le moteur** : RBAR 0.4.40 pour le navigateur, archive `rbar-plans-web-0.4.40.zip` transmise par le propriétaire avec
  son empreinte (`rbar-plans-web-0.4.40.zip.sha256` : `0d7bc3f5…faebb51`, vérifiée) ; déposé par
  `site/web/outils/vendre_rbar_plans.py` à la place de 0.4.38 : 38 fichiers, 15,21 Mio (dont `licences/LICENSE.TXT` et
  `licences/THIRD-PARTY-NOTICES.TXT` du runtime .NET).
- **L'appel** : `moteur.plans(ouvrage, { nom, unSeulPdf: true })`, dans le travailleur des plans.
- **Le site** : servi par `site/web/outils/serveur_local.py` (en-têtes et politique de sécurité de Cloudflare reproduits) ;
  Chromium (Playwright). Le parcours : harnais `site/web/verifier_site.js`, filtre `exporter` — une note neuve des
  fondations de grues (chantier « Greenwood », nom « Fondation grue tour GT1 », valeurs du classeur BASE), menu
  « Exporter » ouvert, « Plans PDF » touché deux fois ; trois formats d'écran, sur le processeur du même poste.

## Résultats

| Format | Premier « Plans PDF » | Second (même fichier) |
|---|---|---|
| PC 1 440 × 900 | 5 875 ms (dessin 5,0 s) | 285 ms |
| Tablette portrait 820 × 1 180 | 6 101 ms | 285 ms |
| Téléphone 390 × 844 (tactile) | 5 877 ms | 301 ms |

- **Un PDF de 93 731 octets, cinq pages A0** (1 189 × 841 mm) : le compte rendu de RBAR annonce cinq feuilles — 01
  plan de coffrage 1/2 (5 vues), 02 plan de coffrage 2/2 (1 vue), 03 plan d'armatures 1/3 (5 vues), 04 plan
  d'armatures 2/3 (2 vues), 05 plan d'armatures 3/3 (2 vues) ; le PDF a cinq objets « /Type /Page » (« /Count 5 »).
  Sans l'option, les cinq PDF séparés pèsent 51 106 + 39 232 + 55 371 + 44 518 + 51 608 = 241 835 octets (les polices
  une fois au lieu de cinq).
- Message final : « F1 - plans.pdf téléchargé : 5 feuilles (coffrage 2, armatures 3) en 5,8 s — dessin 5,0 s » ; au
  second clic : « … en 0,1 s ».
- Pendant l'attente, le message tenu a dit « écriture du fichier de l'ouvrage » (≈ 2 s : le moteur Python écrit le
  fichier « rbar.ouvrage ») puis « dessin des feuilles par RBAR » ; « chargement du moteur » ne s'est pas montré : sur ce
  poste, le moteur de RBAR (préchauffé à l'ouverture du menu) est prêt avant que le fichier le soit.
- En tablette portrait, le menu « Exporter » (sept lignes) tient entier dans la colonne des résultats.

## Ce que cette mesure ne dit pas

- La durée sur une vraie tablette ou un vrai téléphone (le message final la donnera, mesurée sur l'appareil).
- La mémoire prise par le moteur (le navigateur ne la donne pas à la page sans isolement).
