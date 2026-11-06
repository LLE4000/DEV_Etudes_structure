# Maquettes du site « Stab Méthodes »

Les maquettes servent à **valider l'apparence** du futur site (stab-methodes.be) avant de le construire.
Rien n'y est inventé : les dessins et les chiffres sont produits par le **moteur Python du dépôt**
(le même que l'écran Streamlit), extraits le 06/11/2026.

| Page | Ce qu'elle montre |
|---|---|
| `pages/accueil.html` | le tableau de bord : recherche, rapports récents avec leur taux, menus Béton / Acier / Autres |
| `pages/projets.html` | chantier › dossiers › rapports, chaque rapport avec ses indices (A, B, C…) |
| `pages/garde-corps.html` | l'établi d'un module : la toile du dessin sur toute la largeur, les données et le résultat dans deux cadres flottants qu'on replie |
| `pages/equipe.html` | l'équipe : inviter une adresse, les rôles (administrateur, membre), la validation des modules |

Ce que le bureau a demandé, et les réponses retenues : `docs/site/CAHIER_DES_CHARGES.md`.

Chaque page a ses trois formats. Au téléphone : un en-tête et des onglets en bas (Accueil, Projets,
Récents, Modules) ; l'accueil montre une famille de modules à la fois ; les projets montrent les dossiers
en pastilles et les rapports en cartes, et la fiche d'un rapport (ses indices) monte du bas ; le module
a trois onglets (Données, Dessin, Résultat) et le sélecteur de profilés monte du bas. Le téléphone
fonctionne, sans plus d'effort : le bureau travaille sur PC.

Options d'adresse : `?theme=dark` (thème sombre), `?vue=coupe` (une autre vue du dessin),
`?profils=1` (le sélecteur de profilés du montant), `?calcul=montant` (« Voir le calcul »), `?replie=1` (les deux
cadres repliés, le dessin sur toute la largeur), `?tab=donnees|dessin|resultat` (l'onglet du téléphone),
`?fiche=1` (la fiche du rapport, au téléphone). Les pages se lient entre elles (rapport récent → module,
retour → projets, « Ouvrir » → module), et le bouton soleil / lune bascule le thème.

**L'aperçu cliquable** réunit les quatre pages en une seule (`apercu.py`), à ouvrir au téléphone comme au
PC ; son thème suit celui de l'appareil.

## Les fichiers

- `extraire_donnees.py` : appelle le moteur et écrit `donnees.json` (dessins SVG, taux, éléments, profilés triés par
  masse pour chaque élément, les formules de « Voir le calcul », la première page de la vraie note pour l'indice old).
  Le code couleur (rouge, vert foncé, vert pâle) vient du statut du moteur : aucun seuil n'est recopié.
- `charte.css` : la charte (couleurs, typographie, composants). C'est elle que le vrai site reprendra.
- `construire.py` : assemble les pages à partir de la charte et de `donnees.json`.
- `captures.js` : vingt et une captures PC 1440, tablette 1280 × 800 et téléphone 390, avec contrôle de ce que
  l'œil voit : aucune erreur, aucun débordement horizontal, dessin visible, aucun nom de profilé coupé,
  l'élément attendu de chaque vue à l'écran, et rien du format voisin (le choix de famille du téléphone
  sur le PC, la barre latérale du PC au téléphone).
- `apercu.py` : l'aperçu cliquable en une page ; `parcours_apercu.js` le parcourt au téléphone (clair et
  sombre) puis au PC, et mesure chaque pas.
- `icones/` : icônes Lucide (licence ISC, `icones/LICENSE`).

## Régénérer

```bash
python3 site/maquettes/extraire_donnees.py
python3 site/maquettes/construire.py
node site/maquettes/captures.js <dossier_des_captures>
python3 site/maquettes/apercu.py <apercu.html>
node site/maquettes/parcours_apercu.js <apercu.html> <dossier_des_captures>
```

La police est Inter quand elle est installée sur la machine ; le vrai site l'empaquettera
(aucune ressource externe).
