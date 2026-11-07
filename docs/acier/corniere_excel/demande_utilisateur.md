# Demande de l'ingénieur (07/11/2026), mot pour mot

AMÉLIORATION DU MODULE CORNIÈRES — STAB MÉTHODES

Je te transmets mon fichier Excel « REALEX Dimensionnement corniereS E10A#0 02-04-26.xlsx ».

Il s'agit d'une note de calcul que j'ai utilisée professionnellement, dont la méthode a été validée par mon maître de stage.

Objectif : reprendre cette méthode comme référence pour améliorer fortement le module Cornières de Stab Méthodes, sans perdre les principes mécaniques ni les calculs existants.

Je ne te demande pas de remplacer mes calculs par un modèle entièrement différent. Je veux comprendre, conserver, vérifier et étendre la méthode.

1. Analyse préalable de l'Excel

Avant toute modification du code, analyse intégralement le classeur :

- Toutes les cellules de saisie et les formules.
- Les paramètres géométriques de la cornière.
- Les différents jeux et excentrements.
- Le calcul des moments dans les ailes horizontale et verticale.
- Les quatre sections de vérification.
- Les contraintes de flexion, de cisaillement et équivalentes.
- Les efforts de traction et de cisaillement dans les ancrages.
- Les feuilles annexes et le catalogue des cornières.

Reconstitue les schémas mécaniques afin d'expliquer le fonctionnement réel de la cornière et l'origine de chaque bras de levier.

Important : ne modifie pas silencieusement les formules historiques. S'il existe une approximation, une incohérence ou un problème de généralisation, explique-le et propose une correction justifiée.

La validation historique de l'Excel ne signifie pas que toutes ses hypothèses sont applicables à toutes les configurations.

2. Conserver les quatre sections de calcul

L'Excel vérifie quatre sections :

1. Aile horizontale — épaisseur courante.
2. Aile horizontale — zone du congé.
3. Aile verticale — épaisseur courante.
4. Aile verticale — zone du congé.

Il calcule les moments, les modules de résistance, les contraintes et les taux d'utilisation.

Les formules utilisent notamment les épaisseurs "e" et "e+r", ainsi que les bras de levier "Xe", "Xe+r", "Ye", "Ye+r" et "YH".

Je veux conserver ces quatre sections comme points de référence.

Pour contrôler ta reprise, voici le cas de l'Excel :

| Donnée | Valeur |
|---|---|
| Cornière | L120×120×15 |
| Rayon | 13 mm |
| Acier | S355 |
| Longueur reprise | 300 mm |
| Réaction ELS | 42,76 kN |
| X | 75 mm |
| Y | 85 mm |
| Jeu horizontal | 0 mm |
| Jeu vertical | 0 mm |
| Coefficient sur la contrainte admissible | 1,50 |

Résultats historiques :

- M1 = 2,00972 kN·m.
- M2 = 2,56560 kN·m.
- M3 = 2,35870 kN·m.
- M4 = 2,89665 kN·m.
- Taux maximum = 88,86 %, section 3.

La nouvelle version doit reproduire ces résultats dans un mode « méthode Excel historique », avant d'étendre le calcul.

3. Gestion des jeux et des excentrements

Je veux conserver les différents jeux, mais les représenter graphiquement et les rendre parfaitement compréhensibles.

Prévoir notamment :

- Le jeu entre la position de référence de la charge et son application réelle.
- Le jeu éventuel entre la cornière et son support.
- Les décalages verticaux.
- La position réelle de chaque ancrage et de la résultante de compression.

Je souhaite pouvoir définir un jeu par défaut de 20 mm, modifiable pour chaque calcul.

Attention : l'Excel contient plusieurs paramètres distincts, notamment "Xj" (C14), "Yj" (C17) et le jeu utilisé pour les ancrages (C76). Ils ne doivent pas être fusionnés sans comprendre leur signification.

Une distance de montage ne doit pas être confondue avec un bras de levier d'ancrage.

Toutes les dimensions doivent être visibles sur le dessin et leur modification doit mettre à jour les efforts.

4. Tracer les diagrammes de moments continus

C'est une amélioration essentielle.

Actuellement, l'Excel vérifie seulement quatre sections.

Je veux obtenir un diagramme continu des moments fléchissants dans chacune des deux ailes, avec :

- La position réelle de la charge.
- Les variations des efforts.
- La position des quatre sections historiques.
- Les valeurs des moments aux sections remarquables.
- Les moments maximaux et leurs positions.
- Les zones de congés clairement identifiées.

Il faut aussi calculer les propriétés résistantes adaptées aux sections réellement étudiées.

Ne pas confondre le diagramme de moment et la variation de résistance de la section.

Le moment dépend des efforts et de l'équilibre mécanique. La contrainte dépend également de la géométrie locale.

Dans le congé, l'approximation de l'Excel utilisant "e+r" devra être analysée : déterminer si elle est conservatrice, admissible ou à remplacer par une méthode plus représentative.

Je veux aussi pouvoir afficher séparément le diagramme des contraintes et, si pertinent, le taux d'utilisation en fonction de la position.

Il faut que les résultats soient mécaniquement cohérents et pas simplement des courbes interpolées entre les quatre points de l'Excel.

5. Cornière dans les deux orientations

Actuellement, la note représente une cornière avec une orientation particulière.

Il faut prévoir :

Configuration A — aile horizontale orientée vers le bas. Reprendre le principe exact de l'Excel et ses quatre vérifications.

Configuration B — aile horizontale orientée vers le haut. Inverser la cornière et reconstruire correctement le modèle mécanique.

Vérifier notamment : les nouvelles positions de charge ; les bras de levier ; les zones en traction et en compression ; les contacts possibles avec le support ; les efforts dans les ancrages ; les moments dans les deux ailes.

Il ne suffit pas de retourner graphiquement la cornière : les hypothèses mécaniques doivent rester valables.

Prévoir également les cornières à ailes inégales et leur orientation.

6. Longueur de cornière et ancrages multiples

C'est le deuxième grand développement.

Aujourd'hui, l'Excel calcule principalement une section de cornière en considérant une longueur reprise.

Mais imaginons : une cornière de 2,00 m ; trois ancrages répartis sur cette longueur ; une ou plusieurs charges appliquées à différentes positions ; une charge éventuellement répartie sur une partie de la cornière.

Je veux pouvoir définir librement : la longueur totale ; le nombre d'ancrages ; leurs positions et entraxes ; les distances aux extrémités ; les positions longitudinales des charges ; la largeur réelle d'application des charges ; les dimensions des ailes et leur épaisseur.

Il faudra alors distinguer :

A. Le comportement local de la section transversale de la cornière. C'est le principe de mon Excel : les moments dans les ailes et les efforts transmis aux ancrages.

B. Le comportement longitudinal sur la longueur totale de la cornière. Les charges ne sont pas nécessairement réparties uniformément entre les ancrages. Il faut déterminer la flexion longitudinale éventuelle, les efforts dans chaque ancrage, les effets d'excentricité et, si nécessaire, la torsion.

Pour une cornière longue, je ne veux pas qu'on suppose automatiquement que chaque ancrage reprend la charge totale divisée par leur nombre.

Analyse la méthode mécanique pertinente et les conditions nécessaires pour obtenir une répartition fiable. Si les rigidités des ancrages, du support ou de la cornière sont indispensables, il faut en tenir compte ou annoncer clairement les hypothèses simplificatrices.

Si nécessaire, prévoir un diagramme longitudinal distinct du diagramme local des ailes.

7. Calculs ELS et ancrages ELU

Mon Excel utilise volontairement une méthode de vérification aux contraintes admissibles avec des sollicitations ELS non majorées. Il compare notamment la contrainte équivalente à : "σadm = fy / 1,50".

Je souhaite conserver cette méthode historique pour la cornière et afficher explicitement son coefficient de sécurité.

Attention : ce n'est pas une justification Eurocode 3 aux états limites à confondre avec les coefficients partiels réglementaires.

Pour les ancrages, je travaille avec Hilti PROFIS Engineering, et leurs sollicitations sont vérifiées à l'ELU.

Je veux donc : les efforts ELS dans la cornière ; les moments et contraintes de la méthode historique ; les efforts ELS de chaque ancrage ; les efforts ELU de chaque ancrage destinés à Hilti ; les composantes de traction et de cisaillement clairement séparées ; les positions exactes des ancrages et les hypothèses de répartition.

Les charges ELU devront être introduites ou calculées à partir des combinaisons réellement définies. Ne pas appliquer automatiquement un coefficient global arbitraire aux résultats ELS.

Le module doit pouvoir fournir les efforts à introduire dans Hilti, mais ne doit pas prétendre vérifier intégralement les chevilles si les résistances du support, les distances aux bords et les autres paramètres nécessaires ne sont pas connus.

8. Dessins techniques à réaliser

Présentation identique aux autres modules Stab Méthodes, dessins propres, cotés et à l'échelle lorsque c'est possible.

Coupe transversale : géométrie de la cornière ; mur ou support ; épaisseurs et congé ; charge et excentrements ; jeux ; position de l'ancrage ; résultante de compression ; bras de levier ; quatre sections de vérification.

Élévation : longueur totale de la cornière ; positions de tous les ancrages ; entraxes ; distances aux extrémités ; position et largeur d'application des charges ; répartition des charges entre les ancrages.

Diagrammes : moments locaux dans les ailes ; contraintes et taux d'utilisation ; moments longitudinaux si nécessaire ; efforts repris par chaque ancrage.

Tout doit se mettre à jour automatiquement avec les paramètres.

9. Catalogue des cornières et futures cornières PRS

Réutiliser le catalogue de profilés du site, éviter une deuxième base indépendante. Prévoir : cornières à ailes égales ; à ailes inégales ; différentes épaisseurs ; nuances d'acier ; orientations ; géométrie personnalisée. Plus tard : cornières PRS de deux plats soudés perpendiculairement (épaisseurs indépendantes, soudures, propriétés réelles) — ne pas développer tout de suite, prévoir l'architecture.

10. Présentation de la note de calcul

Note professionnelle A4 paysage, homogène avec les autres modules. Organisation : 1 identification du chantier et de la cornière ; 2 hypothèses et méthode ; 3 géométrie et coupes cotées ; 4 charges ELS et ELU ; 5 schéma mécanique et réactions ; 6 diagrammes de moments ; 7 vérifications des ailes et des sections critiques ; 8 efforts par ancrage ; 9 synthèse des taux ; 10 conclusion et limites. Toutes les formules consultables depuis l'interface (fenêtre de calcul détaillé) : formule symbolique, valeurs substituées, résultat, admissible, taux à droite. Compréhensible et vérifiable par un autre ingénieur sans lire le code.

11. Méthode de développement par étapes : 1 analyse de l'Excel ; 2 reproduction exacte du cas historique ; 3 dessins, deux orientations, diagrammes continus, quatre sections ; 4 généralisation (longueurs, charges, ancrages, répartition justifiée) ; 5 note finale.

12. Ce que j'attends maintenant : avant de modifier le module, une analyse technique critique et détaillée du classeur. Expliquer :
1. Le cheminement des efforts dans la cornière.
2. La signification physique de chaque bras de levier et de chaque jeu.
3. Pourquoi les quatre moments sont différents.
4. Comment obtenir des diagrammes continus exacts à partir des hypothèses du modèle.
5. Comment traiter une cornière de 2 m avec plusieurs ancrages et des charges localisées.
6. Ce qui change lorsqu'on retourne la cornière.
7. Les hypothèses de l'Excel qui peuvent être conservées et celles qui doivent être vérifiées ou corrigées.
8. Les améliorations proposées au module existant.

Ne pas commencer par réécrire tout le logiciel. Comprendre parfaitement l'Excel, contrôler les équations, proposer un modèle mécanique cohérent. Conserver l'esprit et la méthode de la note historique, avec un module plus complet, rigoureux et professionnel. « Le modèle de la cornière, on peut faire un truc bien. À la limite, il peut y avoir une sorte de maillage où on voit la déformation de la cornière, on voit les contraintes dans la cornière. Si c'est facile à faire, on peut faire afficher des petites couleurs pour faire les contraintes, les déformations. Il faut vraiment que ce soit une belle note de calcul. »
