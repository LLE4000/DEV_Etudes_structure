# Modèles de dimensionnement par module — réflexion (DAL7, 13/11/2026)

*Demande du propriétaire (13/11/2026, § 9 de la révision des dalles) : « prévoir des templates de dimensionnement par
module (espacements, géométrie par défaut, taux visé ~100 %, préférences) — réflexion seulement ; pour le moment,
limiter aux modules Dalle et Prédalle ; ne pas commencer la refonte des poutres ». Ce document est une PROPOSITION :
rien n'est construit tant que le propriétaire ne l'a pas tranchée (§ 6).*

## 1. Ce qu'un modèle est — et ce qu'il n'est pas

Un **modèle** est un jeu de **valeurs de départ et de préférences** d'un module : ce qu'une note neuve reçoit, et ce
que le sélecteur d'armatures propose. Il sert à **encoder plus vite** et à **dimensionner comme le bureau le fait**.

Il n'est **jamais** une règle de calcul : un modèle ne change ni une formule, ni un coefficient normatif, ni un seuil
de vérification. Le moteur reste la source unique du calcul ; le modèle ne fait que choisir des DONNÉES (une épaisseur,
un treillis, un espacement) que le moteur vérifie comme toutes les autres. Un modèle mal réglé donne au pire une
proposition moins bonne, jamais un « conforme » faux.

## 2. Ce qu'il contiendrait pour les dalles et prédalles

| Rubrique | Exemples | Où c'est aujourd'hui |
|---|---|---|
| Géométrie par défaut | b = 100 cm, h = 20 cm, c = 3 cm, direction principale Y ; prédalle h_p = 6 cm | `beton/dalle/entree.py` (DEFAUTS), `beton/predalle/entree.py` |
| Matériaux | C30/37, B500 | idem |
| Méthode | γs = 1,5, jeu du premier lit = 0 | idem (communs à la note) |
| Armatures proposées | treillis du stock du bureau (liste), diamètres admis (Ø8 à Ø16), **espacements autorisés** (10, 15, 20, 25 cm) | la liste du sélecteur : `MO.candidats_base` (Ø8 à Ø25, 100 à 250 mm, les dix treillis) |
| Objectif | taux visé : la plus petite section dont le taux est ≤ 100 % (aujourd'hui) — ou une bande, p. ex. 85 à 100 % | l'étoile ★ du sélecteur (« la plus petite qui tient ») |
| Préférences | treillis avant barres ; le même treillis en bas et en haut ; renforts en barres Ø/e plutôt qu'en n barres | — |

Le lot DAL4 pose la PREMIÈRE pièce : la liste des **espacements autorisés**, dans les options de la note, avec ses
valeurs par défaut. Le reste attend la décision.

## 3. Où il vivrait

- **Par bureau**, et non par personne : deux ingénieurs du même bureau dimensionnent pareil. Le site a déjà le compte du
  bureau (pages Projets et Équipe) : le modèle serait une fiche de plus, lisible par tous, modifiable par un
  administrateur — comme la marque « Module vérifié ».
- **Hors ligne** : le site calcule sans réseau (le moteur est dans la page). Le modèle lu en dernier serait gardé dans
  le navigateur ; sans lui, les valeurs d'usine du module (celles d'aujourd'hui).
- **Une note garde ce qu'elle a reçu.** Le modèle ne s'applique qu'à la CRÉATION (note neuve, dalle ou section
  ajoutée) et au sélecteur ; une note existante ne bouge jamais quand le modèle change. Sinon, rouvrir une ancienne note
  pourrait changer son résultat sans que personne n'y ait touché. La note PDF pourrait nommer le modèle de départ
  (« Modèle du bureau, version du 13/11/2026 ») — à décider.

## 4. Comment il serait vérifié

- Le moteur ne lit pas le modèle : les tests du calcul ne changent pas.
- Un test lie chaque valeur du modèle d'usine aux valeurs par défaut du moteur (une seule source : le modèle d'usine
  EST la table du moteur, lue, pas recopiée).
- Le sélecteur : sous un modèle qui n'admet que 15 et 20 cm, aucune proposition à 10 ou 25 cm, et l'étoile est la plus
  petite section qui tient PARMI les admises ; une note sans modèle propose comme aujourd'hui.
- Le harnais : une note neuve créée sous un modèle porte ses valeurs, aux trois formats.

## 5. L'ordre proposé

1. **DAL4** : les espacements autorisés dans les options de la note (dalle et prédalle). Et, décidé le 13/11/2026 : les
   PAS des flèches de chaque champ de dimension (`docs/site/CHARTE_ERGONOMIE.md`, E9) font partie du modèle — le
   propriétaire les donne (« une hauteur de poutre, plutôt du 5 cm ; un enrobage, du 1 cm »).
2. Le modèle d'usine rendu visible : une page « Modèle du bureau » qui montre, module par module, les valeurs de départ
   — en lecture seule d'abord.
3. Le modèle modifiable par un administrateur (dalle et prédalle seulement), appliqué aux notes NEUVES.
4. Les autres modules, un par un, quand le propriétaire le demande (la poutre en dernier : sa refonte n'est pas
   commencée, à sa demande).

## 6. Questions au propriétaire

1. Un modèle **par bureau** (proposé) ou par ingénieur ?
2. Le **taux visé** : « la plus petite qui tient » (≤ 100 %, aujourd'hui) suffit-il, ou faut-il une marge (p. ex. viser
   ≤ 95 %) ?
3. Les **treillis du stock** : la liste actuelle (dix treillis, de 5/5/150/150 à 12/12/100/100) est-elle celle du
   bureau ?
4. La note PDF doit-elle **nommer le modèle** de départ ?
