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

## 7. Ce qui est construit — le GABARIT DU BUREAU (lot 6, 14/11/2026)

*Décision du propriétaire du 13/11/2026 (au soir) : « dans les options du site, les options par défaut de chaque module
— que le responsable fixe ; chaque note peut changer une valeur au cas par cas (« pour cette dalle-là, pas de jeu ») » ;
son exemple : « on prend la longueur de recouvrement et on l'arrondit tout le temps au 5 cm près ou à 10 cm près ». Les
questions du § 6 restent ouvertes, sauf la première : le gabarit est celui du POSTE aujourd'hui (comme l'équipe), celui
du BUREAU avec les comptes.*

- **La page « Paramètres »** (le menu : l'icône ⚙ du rail, le dernier onglet du téléphone ; elle disait « bientôt ») :
  « Le site » (le thème : Clair · Sombre · Comme l'appareil), puis « Gabarit du bureau » — une carte par module qui en a
  un (poutre ; dalles et prédalles), sa rubrique **Options RENDUE PAR LE MOTEUR** (l'action « gabarit » du routeur,
  `api_web/gabarit.py`) : le même composant que dans la note — mêmes champs, mêmes bornes, mêmes refus (R1, R2). Une
  valeur du bureau (un écart à l'usine) est **encadrée**, l'usine au survol ; « Valeurs d'usine » efface le module.
- **Ce qu'il contient** — les options de la rubrique Options, rien d'autre : *poutre* — jeu de pose, jeu entre lits,
  arrondi, γs, d_g, Ø et s_max des armatures de peau, l'option « Taux d'armature » et ses réglages ; *dalles et
  prédalles* — γs, jeu de pose, arrondi, **le pas d'arrondi de la longueur des barres de couture** (1, 5 ou 10 cm : une
  option nouvelle, l'exemple du propriétaire), les espacements admis. Les clés : `GABARIT` de chaque module.
- **Qui le change** : un **administrateur** (`js/equipe.js`) ; un membre le LIT (contrôles éteints) ; personne présenté :
  la page renvoie à l'Équipe. Chaque changement laisse une ligne au journal (« a changé le gabarit du bureau : Dalles
  béton armé », une par visite ; « a rendu aux valeurs d'usine … »).
- **Ce qui est gardé** : les ÉCARTS à l'usine que le moteur a rendus (`js/gabarit.js`, `stab.gabarit.v1` : `{ modules: {
  <module>: { <option>: valeur } }, par, t }`) — ni valeur d'usine, ni borne, ni libellé. Une valeur égale à l'usine
  n'est pas gardée : elle SUIT l'usine (si le moteur change un défaut, le gabarit ne le fige pas).
- **Ce qui le reçoit** : une note **NEUVE** — la requête « nouveau » de `js/etabli.js` porte le gabarit du module ; le
  moteur écrit chaque valeur par la saisie du module (`modifier` : le contrôle de la rubrique Options) et DIT un refus
  (« … — gabarit du bureau : la valeur d'usine est gardée »). L'aperçu d'un calcul neuf (calculé aux valeurs d'usine) n'est
  pas montré quand un gabarit s'applique. Une note **existante** ne bouge jamais (« calculer », « modifier » ignorent le
  gabarit) ; dans sa rubrique Options, chaque valeur se change au cas par cas.
- **Le moteur ne lit pas le gabarit pour calculer** : il choisit des données, vérifiées comme toutes les autres (§ 1).
- **Les contrôles** : `tests/test_gabarit.py` (l'usine LUE des tables du moteur, la note neuve et son calcul, les refus,
  la note existante intacte, l'action « gabarit », 5 mutations) ; `tests/test_site_bureau.py` (le registre joué dans
  node : écarts, signature, mémoire abîmée copiée, stockage plein ; la page et le menu ; 5 mutations) ; le harnais
  (`site/web/verifier_site.js`, filtre « bureau ») aux quatre formats — la page mesurée (rien ne dépasse, 40 px au doigt,
  le contour d'une valeur du bureau), un administrateur change, une note neuve reçoit (couture « 2 × 60 cm »), une note
  existante garde 1 cm, un membre lit, « Valeurs d'usine » efface.

**Pas encore fait** (une décision du propriétaire d'abord) : des réglages distincts « coulé en place » / « préfabriqué »
(un jeu de pose propre aux barres de la peau d'une prédalle — le propriétaire a dit « on peut avoir le choix », sans
valeur différente : 1 cm pour les deux) ; le jeu entre deux couches d'une dalle (0 aujourd'hui, fixé le 13/11/2026) ;
la géométrie de départ (enrobage, h_p) et le taux visé (§ 6, question 2) ; la liste des treillis du stock (question 3) ;
la note PDF qui nomme le gabarit de départ (question 4).
