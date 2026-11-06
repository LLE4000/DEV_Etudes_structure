# Maquettes du site « Stab Méthodes »

Les maquettes servent à **valider l'apparence** du futur site (stab-methodes.be) avant de le construire.
Rien n'y est inventé : les dessins et les chiffres sont produits par le **moteur Python du dépôt**
(le même que l'écran Streamlit), extraits le 06/11/2026.

| Page | Ce qu'elle montre |
|---|---|
| `pages/accueil.html` | le tableau de bord : recherche, rapports récents avec leur taux, menus Béton / Acier / Autres |
| `pages/projets.html` | chantier › dossiers › rapports, chaque rapport avec ses indices (A, B, C…) |
| `pages/garde-corps.html` | l'établi d'un module : la toile du dessin sur toute la largeur, les données et le résultat dans deux cadres flottants qu'on replie |
| `pages/poutre.html` | l'établi des **poutres béton armé**, bâti avec les MÊMES composants : sections et sollicitations, coupes à l'échelle (section A, section B, les deux), résultat section par section, fenêtre de calcul par section, choix des armatures |
| `pages/equipe.html` | l'équipe : inviter une adresse, les rôles (administrateur, membre), la validation des modules |
| `pages/parametres.html` | les valeurs du bureau, puis celles de chaque module ; une note peut en changer une pour elle seule |

Ce que le bureau a demandé, et les réponses retenues : `docs/site/CAHIER_DES_CHARGES.md`.

Chaque page a ses trois formats. Sur PC, une barre de cinq icônes (Accueil, Projets, Modules, Équipe, Paramètres),
chacune nommée au survol ; « Modules » ouvre le volet des modules par famille. Au téléphone : les cinq mêmes entrées en
bas ; l'accueil montre une famille de modules à la fois ; les projets montrent les dossiers en pastilles et les
rapports en cartes, et la fiche d'un rapport (ses indices) monte du bas ; un module a trois onglets (Données, Dessin,
Résultat), et le sélecteur comme la fenêtre de calcul montent du bas, sur toute la largeur. Le téléphone fonctionne,
sans plus d'effort : le bureau travaille sur PC.

Options d'adresse : `?theme=dark` (thème sombre), `?vue=coupe` ou `?vue=AB` (une autre vue du dessin),
`?profils=1` (le sélecteur de la pièce par défaut : le montant, les armatures inférieures de la section A),
`?calcul=montant` ou `?calcul=1` (la fenêtre de calcul), `?replie=1` (les deux cadres repliés, le dessin sur toute la
largeur), `?tab=donnees|dessin|resultat` (l'onglet du téléphone), `?fiche=1` (la fiche du rapport, au téléphone),
`?modules=1` (le volet des modules), `?nouvelle=1` (la fenêtre « Nouvelle note »). Les pages se lient entre elles
(rapport récent → module, retour → projets, « Ouvrir » → module, « Créer la note » → poutre), et le bouton soleil /
lune bascule le thème.

**L'aperçu cliquable** réunit les six pages en une seule (`apercu.py`), à ouvrir au téléphone comme au PC ; son thème
suit celui de l'appareil.

**D'où viennent les chiffres de la poutre.** Du module poutre de Streamlit (`modules/poutre.py`), piloté sans écran :
son calcul est mêlé à son écran, et deux grandeurs (τ et s_th) n'existent que dans le code d'écran — l'extracteur les
recalcule et les compare aux états du module (une divergence arrête l'extraction). Avant le vrai site, ce calcul doit
devenir un moteur nommé et testé, comme celui du garde-corps.

## Les fichiers

- `extraire_donnees.py` : appelle le moteur et écrit `donnees.json` (dessins SVG, taux, éléments, profilés triés par
  masse pour chaque élément, les formules de « Voir le calcul », la première page de la vraie note pour l'indice old).
  Le code couleur (rouge, vert foncé, vert pâle) vient du statut du moteur : aucun seuil n'est recopié.
- `charte.css` : la charte (couleurs, typographie, composants). C'est elle que le vrai site reprendra.
- `construire.py` : les composants communs (barre, volet des modules, lignes de résultat, fenêtre de calcul,
  sélecteur, établi…) et les pages accueil, projets, garde-corps, équipe ; `page_poutre.py` et `page_parametres.py` :
  les deux autres pages, bâties sur ces composants ; `dessin_poutre.py` : la coupe d'une poutre à l'échelle (la hauteur
  utile d se cote depuis la face comprimée).
- `captures.js` : trente-trois captures PC 1440, tablette 1280 × 800 et téléphone 390, avec contrôle de ce que
  l'œil voit : aucune erreur, aucun débordement horizontal, dessin visible, aucun nom ni libellé coupé, aucun contenu
  rogné par sa boîte, l'élément attendu de chaque vue à l'écran, une feuille du téléphone sur toute la largeur et son
  tableau entier, et rien du format voisin (le choix de famille du téléphone sur le PC, la barre du PC au téléphone).
- `tests/test_site_maquettes.py` (à la racine) : ce qu'une capture ne prouve pas — les valeurs du bureau affichées sans
  arrondi (γ_M2 = 1,25), les lignes de la poutre aux taux du module, la cote d, les cinq entrées identiques PC et
  téléphone, l'étiquette « modifié ».
- `apercu.py` : l'aperçu cliquable en une page ; `parcours_apercu.js` le parcourt au téléphone (clair et
  sombre) puis au PC, et mesure chaque pas.
- `icones/` : icônes Lucide (licence ISC, `icones/LICENSE`).

## Régénérer

```bash
python3.12 site/maquettes/extraire_donnees.py   # le moteur s'écrit en Python 3.12
python3 site/maquettes/construire.py            # le générateur se lit aussi en 3.11
node site/maquettes/captures.js <dossier_des_captures>
python3 site/maquettes/apercu.py <apercu.html>
node site/maquettes/parcours_apercu.js <apercu.html> <dossier_des_captures>
```

La police est Inter quand elle est installée sur la machine ; le vrai site l'empaquettera
(aucune ressource externe).
