# Maquettes du site « Stab Méthodes »

Les maquettes servent à **valider l'apparence** du futur site (stab-methodes.be) avant de le construire.
Rien n'y est inventé : les dessins et les chiffres sont produits par le **moteur Python du dépôt**
(le même que l'écran Streamlit), extraits le 06/11/2026.

| Page | Ce qu'elle montre |
|---|---|
| `pages/accueil.html` | le tableau de bord : recherche, rapports récents avec leur taux, menus Béton / Acier / Autres |
| `pages/projets.html` | chantier › dossiers › rapports, chaque rapport avec ses indices (A, B, C…) |
| `pages/garde-corps.html` | l'établi d'un module : données à gauche, dessin au centre, résultat à droite |

Options d'adresse : `?theme=dark` (thème sombre), `?vue=coupe` (une autre vue du dessin),
`?profils=1` (la feuille des profilés), `?tab=donnees|dessin|resultat` (l'onglet du téléphone).

## Les fichiers

- `extraire_donnees.py` : appelle le moteur et écrit `donnees.json` (dessins SVG, taux, éléments, tableau des profilés).
  Le code couleur (rouge, vert foncé, vert pâle) vient du statut du moteur : aucun seuil n'est recopié.
- `charte.css` : la charte (couleurs, typographie, composants). C'est elle que le vrai site reprendra.
- `construire.py` : assemble les pages à partir de la charte et de `donnees.json`.
- `captures.js` : captures PC 1440, tablette 1280 × 800 et téléphone 390, avec contrôle
  (aucune erreur, aucun débordement horizontal, dessin visible).
- `icones/` : icônes Lucide (licence ISC, `icones/LICENSE`).

## Régénérer

```bash
python3 site/maquettes/extraire_donnees.py
python3 site/maquettes/construire.py
node site/maquettes/captures.js <dossier_des_captures>
```

La police est Inter quand elle est installée sur la machine ; le vrai site l'empaquettera
(aucune ressource externe).
