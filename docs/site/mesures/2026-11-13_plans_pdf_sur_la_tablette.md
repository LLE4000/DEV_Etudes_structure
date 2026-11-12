# Les plans PDF sur la tablette du propriétaire — mesure du 13/11/2026 (vers 0 h 30)

Mesure datée (elle ne se réécrit pas ; une nouvelle mesure fait un nouveau fichier). Suite de
`2026-11-12_plans_pdf_un_seul_fichier.md` (le poste de contrôle) : la première mesure sur un vrai appareil.

## Conditions

- **L'appareil** : la tablette du propriétaire — Samsung, Android, **Chrome**, en paysage (≈ 1 280 × 654 de page) ; Wi-Fi.
- **Le site** : en ligne (Cloudflare), RBAR 0.4.40 (« Plans PDF », un seul fichier) ; correction du menu « Exporter »
  comprise (d0b37ce).
- **La lecture** : le message final du site, rapporté par le propriétaire (de vive voix, et une capture d'écran pour le
  dernier) ; la durée est celle du clic au fichier reçu, mesurée sur la tablette même.

## Résultats

| Essai | Durée dite à l'écran |
|---|---|
| Premier plan (le moteur de RBAR téléchargé et démarré) | 12 s |
| Plan suivant, page rechargée (moteur gardé par le navigateur) | 2 s |
| Après une mise à jour du site | 2 s |
| Même fichier, second clic | 0,2 s |
| Après un changement de béton (C40/50), capture d'écran | « 5 feuilles (coffrage 2, armatures 3) en 8,8 s — dessin 8,7 s » |

- Le dessin des cinq feuilles prend 8,7 s sur la tablette, contre 5,0 à 5,2 s au poste de contrôle (× 1,7).
- RBAR attendait 30 à 60 s pour un premier plan sur une tablette (processeur ralenti × 4 à × 6, en émulation) : la
  tablette du propriétaire est bien plus rapide que cette estimation.

## Ce que cette mesure ne dit pas

- Les quatre premières durées sont rapportées de vive voix : leur décimale n'est pas sûre.
- La part du téléchargement dans les 12 s du premier plan (le message ne la sépare pas quand le moteur se charge
  pendant l'écriture du fichier).
- La mémoire prise par le moteur.
