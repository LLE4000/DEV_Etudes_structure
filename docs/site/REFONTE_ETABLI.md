<!-- En-tête du versement (lot 0) ; sous la ligne de séparation, la spécification telle que le jury l'a rendue, sans retouche. -->

# Refonte de l'établi du site « Stab Méthodes »

- **Date** : 07/11/2026.
- **Statut** : **spécification retenue, en cours de réalisation.** Le lot 0 (référence figée, sonde du démarrage,
  mesures « avant ») est versé avec ce document ; les lots suivants suivent l'ordre de B.4
  (0 → 0 bis → S1 → S2 → S3 → S4 → M1 à M5 → C → V), **resserré par l'ingénieur le 07/11/2026** : S2 et S3 d'abord
  sur la poutre seule, au PC (voir « Le pilote » ci-dessous).
- **Au 08/11/2026** : faits, les lots 0, 0 bis (et 0 ter) et M1 (poutre, dalle, prédalle) — S2 et S3 pour ces trois
  modules — sous la forme de l'établi de la poutre. **Révision du 08/11/2026, au pied de ce document** : l'écran de la
  poutre devient la référence de tous les modules, la présentation se refait module par module dans les lots M, le
  regel des photos est gardé (`garde`, `--accepter`), S1 est réduit, D7 revue ; ordre M5 → M4 → M2 → M3, puis C (avec
  l'accord de l'ingénieur) et V.
- **Ce qui suit** est la copie fidèle de la spécification finale : l'avis du jury sur les trois propositions
  (« atelier », « note », « doigt »), puis la spécification en lots. Elle n'est pas retouchée ; une correction s'y
  portera en révision datée, au pied de ce document. Les chemins `/tmp/…` et `/home/user/…` qu'elle cite désignent
  les copies de travail de l'enquête, hors du dépôt.
- **Les décisions D1 à D9** s'écriront dans `docs/site/CAHIER_DES_CHARGES.md` par les lots qui les réalisent (B.1).
- **Livré au lot 0** :
  - le gel des calculs et des valeurs affichées des 18 modules : `tests/test_reference_refonte.py` et
    `tests/reference_refonte/` (239 contrôles ; `mutations.py` en est la preuve, 7/7) ;
  - la sonde du démarrage, fermée par défaut : `site/web/js/sonde.js` (`?sonde=1` dans l'adresse d'un établi) ;
  - les mesures « avant » : `docs/site/mesures/2026-11-07_demarrage_avant.md` et
    `docs/site/mesures/2026-11-07_couverture_avant.json`.

## Les deux demandes de l'ingénieur

Citées mot pour mot telles que le jury les reprend (A.1) ; « … » marque une coupe faite par le jury.

1. **L'établi et le rangement des calculs**, en six priorités :
   - **P1** « Séparer ouvrir un calcul et créer une note… on pourrait ensuite choisir de l'enregistrer comme note » ;
   - **P2** « Vraie structure aux chantiers et dossiers… endroit évident pour créer et parcourir » ;
   - **P3** « Panneaux compacts, données visibles ; liste à gauche, dessin au centre, propriétés ET résultats à
     droite ; réductibles ; chaque élément dans une liste » ;
   - **P4** « Noms visibles, ou entrées clairement désactivées » ;
   - **P5** « Familles Béton, Acier, Autres dans cette fenêtre » ;
   - **P6** « États vides, où les calculs sont sauvegardés, comment les retrouver ».
2. **Le démarrage** : « mesurer ; charger le module en arrière-plan et le réutiliser ; pas de préchargement
   bloquant ».

Et pour l'une comme pour l'autre : « Valider avec des cas de référence et des cas limites par module ».

## Le pilote « Poutres béton armé » (07/11/2026)

**Décision de l'ingénieur** (07/11/2026), qui resserre l'ordre de B.4 : *un seul module d'abord* — « commencer
uniquement par la poutre béton et voir ce que ça donne ; si c'est bon, on fait tous les autres comme ça » — et *le
PC d'abord* (la tablette vérifiée, le téléphone seulement « pas cassé »). Les lots S2 et S3 sont donc réalisés sur la
poutre seule ; les 17 autres modules gardent l'établi d'avant, à l'octet (tests/test_site_web.py le vérifie page par
page), jusqu'à son avis.

**Ce que fait le pilote** :
- la carte de l'accueil ouvre le calcul lui-même, sans fenêtre « Nouvelle note » ; la note est **provisoire** : rien
  n'est écrit tant que l'utilisateur ne modifie rien (ouvrir pour regarder ne remplit plus la liste des notes). La
  première saisie, un renommage ou « Enregistrer comme note… » l'écrivent, et l'adresse prend alors `?note=` ;
- « Enregistrer comme note… » (barre du haut, menu ⋯ du téléphone) donne le nom, le chantier et le dossier (les
  chantiers et dossiers déjà connus sont proposés) ; une fois la note écrite, le même bouton la « classe » ;
- à gauche, la **liste** : la note, ses éléments (le bloc « Dans cette note » du moteur, déplacé tel quel), puis les
  étapes de l'élément ouvert ; les étapes secondaires (la « Méthode ») sous « Options », repliées ;
- au centre, le dessin : un clic sur une pièce (le béton, les barres, les efforts) choisit son étape ; un glisser
  déplace la vue sans rien choisir ; l'étape choisie est surlignée ;
- à droite, les **propriétés** de l'étape choisie (le cadre Données du moteur, filtré : rien n'est copié) au-dessus du
  **résultat** ; chacun se replie ; les deux panneaux se replient en bande ;
- aucun calcul ne change : les 239 cas de référence restent identiques, calcul et affichage.

**Comment l'étendre à un autre module** : l'ajouter à `ETABLI_COMPACT` (`site/web/construire.py`) ; marquer ses
étapes secondaires (`commun.section(…, secondaire=True)`) ; poser sur les pièces de ses dessins
`<g data-sel="<clé d'étape>">` (pour la poutre : `dessin_poutre._cible`, les clés `pt`, `so`, `arm<i>`). Le contrat
est vérifié par `tests/test_api_web_poutre.py` §12 (chaque pièce désigne une étape qui existe) et dans le navigateur
par le filtre « pilote » de `site/web/verifier_site.js` (chaque champ du cadre Données s'atteint par une étape de la
liste ; rien n'est écrit avant la première saisie ; le clic sur le dessin ; le rangement ; les replis).

**Les fichiers** : `site/web/js/compact.js` (la liste, le choix d'étape, le dessin cliquable),
`site/web/js/rangement.js` (« Enregistrer comme note… »), `site/web/css/etabli.css` (la disposition, scopée par
`body[data-etabli="compact"]`), et dans `site/web/js/etabli.js` la note provisoire.

## La finalisation du module « Poutres béton armé » (08/11/2026)

**Demande de l'ingénieur** (« STAB MÉTHODES — finalisation du module poutres béton armé ») : trois colonnes — à gauche
TOUTES les données, au centre un dessin technique interactif et éditable, à droite UNIQUEMENT les résultats — ; le
ferraillage détaillé (lits multiples positionnés, cadres multiples, épingles, latérales) ; des résultats compacts avec
une fenêtre de détail comme le garde-corps ; des états honnêtes (jamais « Conforme » sans effort) ; la note PDF validée
de Streamlit, dans le site ; le `.stab` sans `.json`. Elle **remplace** la disposition du pilote ci-dessus (liste à
gauche, propriétés et résultat à droite).

**Ce qui est fait** :
- **Données** (colonne de gauche, largeur réglable) : « Poutres de la note » (ouvrir, renommer, copier, supprimer,
  ajouter), puis quatre rubriques en **accordéon** (une seule ouverte) : 1 Poutre, 2 Sections et sollicitations (copier,
  retirer, ajouter une section), 3 Armatures de la section courante (lits : n, Ø, répartition, position ; latérales ;
  cadres et épingles : type, Ø, barres entourées ; pas ; « Proposer » ouvre le sélecteur, qui ne propose que des
  dispositions qui se posent), 4 Options (méthode, γs, jeux, granulat, latérales automatiques). « Annuler » et
  « Rétablir » (Ctrl+Z, Ctrl+Maj+Z) dans l'en-tête.
- **Dessin** : la coupe cotée (`api_web/coupe_ba.py`), fond clair, juste sous la barre ; chaque lit, cadre, groupe de
  latérales est une pièce (`data-sel`) dont la ligne des données porte la même clé (`data-cible`) : un clic d'un côté
  choisit l'autre (la rubrique s'ouvre, la ligne se déplie, la pièce s'éclaire) ; un glisser déplace la vue.
- **Résultats** : le verdict, puis par section son état et, pour chaque vérification, DEUX lignes (nom, pièce, taux ;
  valeur comparée à sa limite — le nécessaire face au posé pour les aciers) ; la ligne ouvre SA fenêtre (hypothèses,
  ligne du tableau, formules avec les valeurs, critère, référence) ; « Dispositions constructives » dit son état en mots,
  sans pourcentage ; sans V, « non effectuée ».
- **Note PDF** dans le navigateur, **.stab** et **import Streamlit** : `docs/beton/POUTRE_FERRAILLAGE.md` § 4.

**Les contrôles** : `tests/test_api_web_poutre.py` (§ 12 à 14 : la correspondance dessin ↔ données, les états, le
ferraillage détaillé), `tests/test_note_poutre.py` (la parité de la note), `tests/test_poutre_moteur.py` (la parité du
calcul, inchangée), et le filtre « pilote » de `site/web/verifier_site.js` (accordéon, chaque champ atteignable, clic sur
le dessin dans les deux sens, annuler et rétablir, chaque fenêtre de détail, la note PDF téléchargée, le nom du .stab,
l'import Streamlit, le redimensionnement).

## La deuxième passe : ergonomie et dessin (08/11/2026, suite)

**Demande de l'ingénieur** (« STAB MÉTHODES — MODULE POUTRES BA — deuxième passe d'amélioration ergonomique et
graphique », 28 remarques) : une interface moins chargée, à la finition d'un logiciel d'ingénierie — à gauche la
navigation et les données, au centre un dessin technique de grande qualité, à droite les résultats et vérifications —
sans perdre une fonction ni la fiabilité du calcul. Priorité : simplifier la gauche, redessiner les étriers,
synchroniser les résultats par section. Signalé de vive voix : deux étriers sur les barres 1 → 3 et 2 → 4 donnaient,
dans la note PDF, un étrier hors de la poutre. (Les trois captures annotées annoncées ne sont pas parvenues : la passe
s'appuie sur les 28 remarques écrites.)

**Ce qui est fait, et pourquoi** :
- **Gauche, une arborescence** : chaque poutre « nom · b × h cm » (ni taux, ni pastille : le résultat est à droite) ;
  un clic ouvre une poutre (l'autre se replie), un clic sur la poutre ouverte la replie ; ses actions (renommer, copier
  CETTE poutre, supprimer) dans un menu « ⋯ » discret. Sous la poutre ouverte, ses quatre rubriques : **une ouverte par
  défaut**, un en-tête ouvre ou referme SA rubrique sans toucher aux autres (plusieurs ouvertes si on le veut). Le dessin
  amène à la rubrique d'une pièce : celles qu'il avait ouvertes se replient, celles ouvertes à la main restent.
- **Armatures compactes** : « Inférieures · A_s requis 1 212 mm² », puis une ligne par lit « L1 | 4 | Ø20 | 1 257 mm² »
  dont le détail (répartition en trois modes nommés et dessinés — Uniforme, Superposée, Centrée —, position,
  espacements libres contrôlés et leur état) ne s'ouvre que sur le lit choisi ; une ligne par cadre « C1 · Ø10 · 2 brins
  · barres 1–3 », son détail (type, Ø, les barres entourées : un clic par bout, « Toute la largeur ») sur le cadre
  choisi. Une pastille seulement sur une ligne repliée qui porte un défaut. Libellés courts (« y_G imposé »), le reste
  dans l'info-bulle.
- **Sections** : un bouton rond choisit la section courante (rubrique 2), un sélecteur la change (rubrique 3) ; la
  section choisie à gauche est celle du dessin et la seule vérifiée à droite ; « Toutes » au centre montre toutes les
  coupes et tous les blocs.
- **Dessin** : un en-tête technique (« SECTION A · 25 × 50 cm · C30/37 · B500 », les aciers, les étriers) et une bande
  compacte des trois efforts, sans couleur d'alerte ; plus de « mm · à l'échelle » ; le zoom (icône « ajuster » au lieu
  de « 100 % », qui laissait croire à une échelle d'impression) n'apparaît qu'au survol de la toile (au doigt, après
  un appui). Le béton en contour net sur un semis discret de granulats ; chaque cadre à sa place réelle (x0, x1 du
  moteur), cintré, fermé à 135° en haut à gauche ; acier en double trait. Un cadre choisi (sur le dessin, son étiquette
  ou sa ligne) passe au-dessus des autres et montre deux **poignées** : on les glisse, elles s'aimantent aux barres du
  lit 1 inférieur (jamais entre deux barres, jamais un cadre sur une seule barre).
- **Droite** : le nom de la poutre, puis un **bloc par section** — en-tête « SECTION A · Non conforme » coloré par son
  état (vert pâle conforme, ambre à vérifier, rouge non conforme, gris données incomplètes), repliable, « tout le
  calcul » à droite ; le grand indicateur global est retiré. Chaque vérification sur deux lignes (le nom et ce qu'elle
  nomme — barres posées, ce que demande la bande de τ, les brins —, le taux ; la valeur et sa limite en clair, une jauge
  fine) ; un clic ouvre la fenêtre de détail. Les **dispositions** sont une liste de contrôle rangée par état : non
  conformes (avec ce qu'il faut corriger), à examiner (« Marquer examinée… », justification facultative gardée dans la
  note), examinées et justifiées par l'ingénieur (« Annuler »), conformes automatiquement (une ligne chacune). Un clic
  sur une observation montre sa pièce au dessin et dans les données. Une non-conformité n'a jamais de bouton.
- **Note PDF** : la coupe pose barres et cadres aux positions du moteur (le défaut signalé est corrigé), chaque
  observation a sa ligne (et sa justification) ; `docs/beton/POUTRE_FERRAILLAGE.md` § 4.

**Les contrôles** : `tests/test_api_web_poutre.py` § 15 à 19 (les barres entourées et leurs refus, le ramenage d'un
cadre quand une barre disparaît, la justification et son empreinte, une non-conformité qui ne se justifie pas, la liste
rangée par état, chaque observation qui désigne une pièce, l'en-tête des vues, les poignées cachées, le menu ⋯),
`tests/test_poutre_dispositions.py` (les états), `tests/test_note_poutre.py` (les coupes de la note mesurées), et le
filtre « pilote » de `site/web/verifier_site.js` (l'arborescence sans taux, l'accordéon à plusieurs rubriques, la
navigation depuis le dessin qui garde les rubriques ouvertes à la main, les poignées visibles sur le seul cadre choisi
et glissées jusqu'à une barre, les barres choisies dans les données, la synchronisation section ↔ dessin ↔ résultats,
« Toutes », la justification enregistrée, le zoom discret, le menu ⋯).

## Un seul établi pour les trois modules béton armé (08/11/2026, suite)

**Demande de l'ingénieur** : les modules « Poutres », « Dalles » et « Prédalles » béton armé se ressemblent ; leur
interface doit être la même, « les fonctions mises au même endroit ». Les notes PDF des dalles et prédalles restent
comme elles sont pour l'instant (dans Streamlit) ; dans la note de la poutre, « Armature de peau » dépassait du canevas.

**Ce qui est fait** :
- **Une source pour les trois écrans** : `api_web/etabli_compact.py` porte les briques communes — l'arborescence
  (`arbre` : « nom · dimensions », le menu « ⋯ » renommer / copier CET élément / supprimer, jamais le dernier), l'en-tête
  technique d'une vue (`entete` : trois lignes et la bande des efforts, sans couleur d'alerte), la tête des résultats
  et le bloc d'une section (`bloc` : en-tête coloré par l'état, deux lignes par vérification, « tout le calcul »), la
  carte d'une section et le choix de la section courante. La poutre les utilise (son rendu est resté identique à
  l'octet, prouvé avant le changement suivant) ; `api_web/plaque_compact.py` les assemble pour les dalles et les
  prédalles, dont les écrans Streamlit sont la même famille.
- **Dalles et prédalles, aux places de la poutre** : à gauche l'arborescence, sous l'élément ouvert quatre rubriques
  (1 l'élément, 2 « Sections et sollicitations » — une carte par section, ses quatre moments et V —, 3 « Armatures » de
  la section courante — les quatre nappes, chaque couche sur une ligne qui se déplie : axe, pose, section au mètre,
  « Retirer » —, 4 « Options », repliée) ; au centre les coupes et le plan de chaque section et « Toutes », chacune sous
  son en-tête technique (dont la dernière ligne est la légende du dessin : barres coupées en points, X horizontales…),
  chaque couche cliquable (`data-sel` = `data-cible` : `s<i>.<face>-<j>`) ; à droite un bloc par section. Les efforts
  ne sont plus écrits en rouge sur le dessin : la bande de l'en-tête les donne une fois. Un axe ou un centre de
  gravité saisi garde son « ↺ » (revenir à l'automatique ; l'infobulle dit ce qu'il serait), comme l'écran d'avant :
  la comparaison aux photos figées l'avait vu disparaître, il est rétabli.
- **L'état d'une section** (le même ordre pour les trois modules) : *non conforme* d'abord — une non-conformité
  trouvée sans aucun effort (acier excessif, d utile ≤ 0, barres trop serrées) n'est jamais cachée —, puis *données
  incomplètes* (aucun effort), puis à vérifier / conforme. L'ordre inverse, en place jusque-là dans le bloc de la
  poutre, montrait gris un cas rouge : défaut non conservatif, corrigé et figé par un test de chaque module.
- **La note PDF** : seule la poutre la fait dans le site (`construire.NOTE_PDF`) ; ailleurs les boutons restent grisés
  « bientôt ». *(Révision du 09/11/2026 : la dalle, le garde-corps, les soudures, les charges sur plancher et les
  assemblages la font aussi, sur toutes leurs pages, compactes ou non — docs/site/MODULES_SITE.md § 4 quinquies.)* Dans la note de la poutre, les étiquettes à droite de la coupe se réduisent si elles manquent de place
  (jamais sous 70 % du corps) et la colonne d'étiquettes s'élargit : rien ne sort du cadre (`ndc_pdf/section.py`).

**Les contrôles** : `tests/test_api_web_dalle.py` § 13 et `tests/test_api_web_predalle.py` § 12 (l'arborescence, les
rubriques, le menu, la copie par rang, chaque couche dessinée et saisie sous la même clé, la section courante, l'en-tête,
la tête des résultats), § 3 et § 10 (non conforme sans effort), `tests/test_api_web_poutre.py` § 13 (le même cas pour la
poutre), `tests/test_note_poutre.py` (aucun texte hors du cadre, sept proportions de coupe), `tests/test_site_web.py`
(les trois pages compactes, la note PDF de la seule poutre), `tests/test_reference_refonte.py` (regelé après la preuve
que le calcul est identique : 18 cas sur 18, moteur, état et nombres des fenêtres de calcul), et dans le navigateur
les filtres « formats », « module » et « plaques » de `site/web/verifier_site.js` (l'arborescence, le clic sur une
couche du dessin, « ↺ », la section courante qui garde le genre de la vue, le menu ⋯).

## Les données condensées, le nom sur sa ligne, les poignées au doigt (08/11/2026, troisième passe)

**Demande de l'ingénieur** : « Poutre A » et ses trois petits points dans la colonne de gauche — pourquoi ne pas
modifier le nom là, directement (on gagne trois lignes) ? Béton « C30/37 » et acier « B500 » en menus déroulants sur
une même ligne, la largeur et la hauteur sur une même ligne, l'enrobage dessous : condenser au maximum (sur un écran de
PC, la rubrique prenait beaucoup de lignes pour dire peu). Et sur la tablette, les poignées des étriers se voient mais
ne bougent pas.

**Ce qui est fait** :
- **Le nom sur sa ligne** : l'élément ouvert de l'arborescence porte son nom en champ (`etabli_compact.arbre`, chemin
  « nom », la largeur de son texte, l'allure du titre ; un cadre au survol et au focus) ; le chevron seul replie ou
  déplie. Les rubriques n'ont plus de champ « Nom » ; « Renommer » (menu ⋯) pose le curseur dans ce champ, le nom
  choisi. Un élément verrouillé (dalle, prédalle) lit son nom sans champ, et son « Renommer » est grisé — il n'aurait
  aucun champ où poser le curseur.
- **Deux champs par ligne** : `commun.ligne2` (libellé court, champ, libellé court, champ). Poutre : béton et acier
  (deux menus), b et h, puis c — six lignes deviennent trois. Dalle : béton et acier, b et h, c et la direction
  principale — huit lignes (avec un paragraphe d'aide) deviennent trois. Prédalle : béton et acier, b et h, h_p et c,
  la direction principale et le coulé en place — dix deviennent quatre. Les options aussi, deux par ligne.
- **Aucune explication perdue** : les paragraphes d'aide vont dans l'icône ⓘ de leur champ. Au doigt, le survol qui
  montre une infobulle n'existe pas : un appui sur une ⓘ ouvre son texte dans une bulle, un second appui ou un appui
  ailleurs la ferme (`js/etabli.js`, toutes les pages ; R1 : ce qui se lit à la souris se lit au doigt). L'icône
  (13 px) étant une petite cible, au doigt le LIBELLÉ du champ ouvre aussi sa bulle. (Une première version agrandissait
  la zone d'appui de l'icône par un pseudo-élément invisible : il débordait de sa boîte, et le harnais « formats » l'a
  vu au téléphone comme un nom coupé, sur quatorze modules — il avait raison, la mesure ne doit voir déborder rien.)
- **Les poignées au doigt** : un appui qui bouge, le navigateur le prenait pour faire défiler la page et annulait le
  glisser (`pointercancel` : la poignée revenait à sa place). Un appui qui commence sur une poignée du cadre choisi lui
  est désormais refusé (`touchstart` non passif, sur la toile) ; ailleurs, le doigt fait défiler comme avant.
- Au téléphone, le chevron et le champ du nom font 40 px (cibles au doigt), le nom s'écrit en 16 px.

**Les contrôles** : `tests/test_api_web_poutre.py`, `test_api_web_dalle.py`, `test_api_web_predalle.py` (le nom en champ
sur la ligne de l'élément ouvert et nulle part ailleurs, le nombre de lignes de la rubrique 1, l'acier en menu, le
verrou qui grise « Renommer ») ; `tests/test_reference_refonte.py` vert SANS regel (239 sur 239 : le calcul et ce qui
s'affiche des fenêtres de calcul sont identiques) ; dans le navigateur, les filtres « pilote » (« Renommer » pose le
curseur, Entrée écrit le nom dans la note, une saisie directe aussi), « plaques », « formats » (le nom long se coupe,
les dimensions restent entières ; cibles de 36 px et champs en 16 px au téléphone) et « doigt » (tablette tactile : la
poignée glissée jusqu'à la 3ᵉ barre sans annulation, la bulle d'une ⓘ ouverte et fermée d'un appui) de
`site/web/verifier_site.js` ; le contrôle « aucun nom coupé » de « formats » mesure aussi les libellés à deux champs
par ligne (`.row2`). Chaque contrôle nouveau a été vu rouge sans sa correction.

---

# Refonte de l'établi Stab Méthodes : avis du jury sur les trois propositions, puis spécification finale en lots

Pour chaque proposition, l'avis porte sur la demande de l'ingénieur et ses six priorités, sur R1, R3 et l'accès aux données sans le dessin, sur la compacité vue sur les captures, sur l'effort et le risque, et sur la faisabilité module par module. Je n'ai rien changé dans `/home/user/integ` (e3a0ca0) ni dans `/home/user/chantier_golden`. Je n'ai fait aucun commit ni aucun push. Mes scripts et sorties sont dans `/tmp/claude-0/-home-user/092afadd-796a-54d4-b802-037ae1cad731/scratchpad/refonte/` (`jury_golden.txt`, `jury_notes/essai.mjs`).

---

## A. JURY

### A.0 Ce que le jury a vérifié lui-même (VÉRIFIÉ)

- **Le gel des calculs couvre bien les 18 modules.** `/usr/bin/python3.12 tests/test_reference_refonte.py` dans `/home/user/chantier_golden` (commit db5fb75) donne **239 OK, 0 échec, en 49,0 s**.
  - Ses clés sont les 18 de `registre.cles()` : age, armatures, assemblages, bois, catalogue, corniere, dalle, donnees_beton, enrobage, flambement, garde_corps, plancher, poutre, predalle, profil, sol, soudures, taux.
  - Cas de référence + cas limites par module : age 5+6, armatures 4+5, assemblages 27+12, bois 4+5, catalogue 4+5, corniere 4+4, dalle 4+5, donnees_beton 5+4, enrobage 4+5, flambement 4+6, garde_corps 4+6, plancher 4+5, poutre 4+8, predalle 4+5, profil 4+5, sol 5+7, soudures 4+5, taux 4+5. Le minimum de 3+3 est tenu partout.
  - Les 7 mutations sont VÉRIFIÉES par l'enquêteur ; le jury ne les a pas rejouées.
- **Un défaut non conservatif nouveau, plus grave que l'échec muet.** Si `stab.notes.v1` est abîmé (même d'un seul octet), la première écriture efface toutes les notes.
  - Cause : `toutes()` (`site/web/js/notes.js` L11-13) rend `{}` quand le JSON ne se lit pas, puis `ecrire()` (L22-27) réécrit la clé avec la seule note courante.
  - Mesure (node, localStorage simulé) : 30 notes, JSON tronqué d'un octet, puis une écriture → **1 note** reste, 195 caractères au lieu de 1 910. Le texte d'origine est perdu.
- **L'échec d'enregistrement muet est confirmé.** `charte.css` L678 (`.saved .txt{display:none}`), dans le bloc `@media (max-width:1360px)` qui commence L673. `etabli.js` L42-48 ne change que ce texte.
- **Une aide de la poutre affiche du HTML brut.** Sur les 18 réponses `nouveau`, un seul `title` contient du HTML échappé : celui de la poutre (aide de γs, `poutre.py:186`).
- **`.elgrp` n'est pas une brique de `commun.py`**, qui n'a que `ligne_res` en L234. Chaque module écrit ses groupes de résultat : rattacher les résultats aux nœuds (N3) se fait donc module par module.
- **Le travailleur sait déjà servir n'importe quel module.** Chaque requête porte `module`, et seul le préchargement de numpy dépend de la clé (`travailleur.js` L40-44). Le vrai coût de l'« app unique » est ailleurs : `etabli.js` fait 508 lignes de code exécuté au chargement, avec `MODULE = document.body.dataset.module` en L13.
- **Taille des réponses `nouveau`** (CPython) : de 15 ko (donnees_beton) à 278 ko (garde_corps), environ 1,6 Mo pour les 17 modules hors age. Premier `nouveau` du garde-corps : 1 255 ms (coût des imports).
- **Familles** (`site/maquettes/construire.py` L65-73) : Béton 9, Acier 6, Autres 3, soit les 18 modules.
- **Points de rupture actuels** : téléphone ≤ 700 px, tablette 701 à 1199 px (avec `orientation`), PC ≥ 1200 px. `charte.css` doit rester identique à sa copie (`test_site_web.py` L101).
- **Pas de `docs/DECISIONS.md`** dans ce dépôt : les décisions s'écrivent dans `docs/site/CAHIER_DES_CHARGES.md`.
- **Captures regardées** :
  - atelier : `pc_etabli`, `tel_etabli_donnees`, `tablette_portrait_etabli`, `pc_chantiers` ;
  - note : `etabli_pc`, `etabli_tel_plan`, `etabli_tablette_portrait`, `accueil_pc` ;
  - doigt : `etabli_pc`, `etabli_tel_panneau`, `etabli_tablette_portrait`, `accueil_tel`.

### A.1 Les priorités, une par une

✓ = tenu ; ≈ = tenu en partie ; ✗ = manqué.

| Exigence (mot pour mot) | atelier | note | doigt |
|---|---|---|---|
| P1 « Séparer ouvrir un calcul et créer une note… on pourrait ensuite choisir de l'enregistrer comme note » | ✓ tuile = lien ; bouton « Enregistrer comme note… » | ✓ idem, bouton principal | ≈ tuile = lien, mais le bouton s'appelle « Ranger » : le mot de l'ingénieur disparaît |
| P2 « Vraie structure aux chantiers et dossiers… endroit évident pour créer et parcourir » | ✓ page Chantiers, stab.v2, migration (capture `pc_chantiers` nette) | ✓ idem + bloc Chantiers de l'accueil cliquable | ✓ idem |
| P3 « Panneaux compacts, données visibles ; liste à gauche, dessin au centre, propriétés ET résultats à droite ; réductibles ; chaque élément dans une liste » | ✓ Propriétés et Résultat empilés et visibles ensemble ; l'élément montre toutes ses étapes. Mais au PC le Résultat est à l'étroit (≈ 5 lignes visibles) | ≈ la fiche ne montre que le nœud choisi ; au téléphone, l'onglet Plan laisse ≈ 30 % de vide et le dessin est caché pendant la saisie | ≈ une étape à la fois (moins de données visibles d'un coup) ; tous les résultats derrière un nœud. **Meilleur téléphone** : dessin, panneau et barre au pouce ensemble |
| P4 « Noms visibles, ou entrées clairement désactivées » | ✓ libellés ; Équipe grisée « BIENTÔT » | ✓ idem | ✓ libellés ; Équipe retirée (admissible, mais l'entrée disparaît) |
| P5 « Familles Béton, Acier, Autres dans cette fenêtre » | ✓ une seule source, aussi dans « Nouveau calcul ici » | ✓ | ✓ |
| P6 « États vides, où les calculs sont sauvegardés, comment les retrouver » | ✓ + échec visible | ✓ + échec visible | ✓ + échec visible |
| Démarrage : « mesurer ; charger le module en arrière-plan et le réutiliser ; pas de préchargement bloquant » | ≈ l'app unique n'arrive qu'au dernier lot, comme une option | ✓ app unique dès l'accueil | ≈ app unique « ensuite » |
| « Valider avec des cas de référence et des cas limites par module » | ✓ (gel) | ✓ (gel + couverture prototypée 29/29) | ✓ (gel) |

### A.2 Notes sur 10 (la fidélité compte double dans la note globale)

| Critère | atelier | note | doigt |
|---|---|---|---|
| Fidélité à la demande et aux 6 priorités | 8 | 8 | 7 |
| R1 (trois formats, composant partagé) | 8 | 8 | 9 |
| R3, et R2 au passage | 7 (`arbre` JSON = 2ᵉ représentation dans chaque réponse) | 9 (DOM unique filtré ; le moteur compte les valeurs modifiées) | 8 |
| Toutes les données accessibles sans le dessin | 9 | 9 | 8 |
| Compacité réelle (captures) | 8 | 7 | 7 (lignes de liste de 36-40 px sur deux lignes, liste longue au PC) |
| Effort et risque sur les calculs | 8 | 9 (garde la délégation et la préservation du brouillon d'`etabli.js`) | 8 |
| Faisabilité par module | 7 (prouvé sur la poutre seulement) | 7 (poutre seulement) | 9 (prouvé sur les assemblages : 22 cotes réelles touchables) |
| **Note globale** | **7,9** | **8,1** | **7,9** |

**Verdict.** Aucune proposition ne gagne seule ; la spécification finale en prend le meilleur :
- **la disposition de « atelier »** : Propriétés et Résultat empilés à droite, l'élément montre toutes ses étapes ;
- **le contrat de « note »** : un seul DOM filtré par la CSS, la liste dérivée du DOM, la garde des options modifiées, deux verdicts nommés ;
- **le téléphone et la densité de « doigt »** : dessin gardé visible avec une feuille, barre d'étapes au pouce, densité qui suit le pointeur, et ses pièges CSS (préfixe des classes, `!important` limité à `.sel`).

---

## B. SPÉCIFICATION FINALE

### B.1 Décisions

Elles s'appliquent par défaut. Elles s'écrivent dans `docs/site/CAHIER_DES_CHARGES.md`, aux §1 bis, §5 et §8 réécrits, en datant chaque changement.

- **D1. Ouvrir un calcul n'écrit rien.**
  - À la première saisie, la note est écrite « Non classée ».
  - Le bouton « Enregistrer comme note… » range et nomme ; il devient « Ranger… » une fois la note classée.
  - « Jeter ce calcul » est offert tant que la note n'est pas classée.
  - Raison : le cahier §7, « Ne jamais perdre une note ».
- **D2. « Chantiers » remplace « Projets »** : dans la navigation, l'écran et l'adresse `chantiers`.
- **D3. Navigation.** Équipe reste visible, grisée, avec « bientôt » écrit, `aria-disabled="true"` et `tabindex="-1"`. Paramètres devient actif, en page locale.
- **D4. Panneaux accrochés** (fin des cadres flottants du §8). Le résultat garde ses deux lignes, resserrées.
- **D5. Étapes secondaires** (révisables par l'ingénieur sans toucher l'écran) :
  - age `me` ; armatures `cr`, `ta` ; assemblages `co`, `ca` ; bois `els`, `ap` ; catalogue `co`, `ex` ; corniere `cr` ; dalle `me` ; donnees_beton `so` ; enrobage `gu` ; flambement `me` ;
  - garde_corps `me` seulement : `tr`, `rp` et `fx` sont essentielles ;
  - plancher `hy`, `af` ; `bi` passe sous le nœud Note ;
  - poutre `me` ; predalle `me` ;
  - profil `co` (`in`, les instabilités, reste une étape) ;
  - sol `op`, `me` ; soudures `me` ; taux `qt`.
  - **Règle : `secondaire` ≠ `ouvert=False`.**
- **D6. Onglets du téléphone** : on garde « Données · Dessin · Résultat ».
- **D7. Contrat de la liste.**
  - La liste est **dérivée du DOM** ; aucune clé `arbre` n'est ajoutée à la réponse.
  - Les identifiants sont des préfixes de chemin, fondés sur le rang et alignés sur `data-chemin`.
  - **Aucun champ n'est ajouté à `etat`** : pas d'uid. Le gel fige l'état, et les `.stab` existants doivent rester lisibles.
- **D8. Hors périmètre** : instantané mémoire Pyodide (API privée), bytecode `.pyc`, SharedWorker. Le bouton « Repérer » est inclus.
- **D9. Étiquettes et identification.**
  - L'étiquette figée « Indice A · Brouillon » (`construire.py` L271) devient « Indice <indice réel> ».
  - L'identification des soudures passe sous le nœud Note.
  - Celle du garde-corps reste non affichée : l'afficher ajouterait une fonction.

### B.2 Règles de chantier (tous les lots)

1. **Calculs.**
   - `/usr/bin/python3.12 tests/test_reference_refonte.py` donne **239 OK** à chaque lot.
   - **Jamais de `--geler`** dans un commit de refonte : `git diff <base> -- tests/reference_refonte/*.json` doit être vide.
   - Aucun nouveau champ dans `etat`.
2. **Balisage relu par des modules : interdit d'y toucher.**
   - `<table class="vtab">` : `catalogue.py` L835, `donnees_beton.py` L399, et `sol.py`, qui lève une erreur s'il manque.
   - `<div class="l2">` : `catalogue.py` L882, `donnees_beton.py` L380.
   - `<span class="chip c">` et « à compléter » : `plancher.py` L1367-1374.
3. **Briques communes.** Un attribut n'est écrit que s'il est passé, **après** les attributs existants. Sans paramètre, le HTML reste identique à l'octet.
4. **SVG.**
   - Les attributs vont sur un `<g>` qui **enveloppe**, jamais sur une primitive.
   - Quatre motifs de test finissent par `/>` et rougiraient sinon : `test_api_web_enrobage.py` L292 et L306, `test_api_web_taux.py` L171, `test_api_web_age.py` L395.
5. **Routeur.** Aucun second rendu dans `routeur.traiter` : le gel espionne `C.message`, et un rendu de plus ajouterait des avis.
6. **CSS.**
   - Les nouvelles classes portent le préfixe `et-`, dans `css/etabli.css`.
   - Règle `[hidden]{display:none!important}` : `.btn` imposait `display`, défaut trouvé par deux maquettes.
   - `!important` seulement sur `.sel` et `.sel-dans` : le `<style>` du kit a une spécificité d'identifiant.
   - `charte.css` n'est jamais modifié ; on le surcharge dans `site.css` ou `etabli.css`.
7. **R1 et R5.**
   - Chaque lot livre les trois formats dans le même commit.
   - Rien d'externe ; `_headers` (CSP) inchangé.
8. **Harnais.**
   - `site/web/verifier_site.js` est mis à jour dans le même commit que les sélecteurs.
   - Chaque rouge est classé dans le commit : le harnais **suit** le changement ou a **trouvé** un défaut.
9. **Navigateur.**
   - Serveur détaché, lancé hors du verrou, par `setsid nohup python3 site/web/outils/serveur_local.py <port 8840-8899>`.
   - Navigateur sous `flock /tmp/verrou_navigateur`.
   - On n'arrête un processus que par son PID, et seulement les siens.
   - Tablette testée **avec `hasTouch`**.
10. **Code.**
    - Noms et commentaires en français.
    - Chaque correction est figée par un test qui rougit si on la retire ; la mutation est nommée dans le commit.

### B.3 L'établi retenu (référence commune)

**Dispositions** (rail = barre de navigation latérale)

| | PC 1440×900 (≥ 1200, pointeur fin) | Tablette paysage 1180×820 | Tablette portrait 820×1180 | Téléphone 390×844 (≤ 700) |
|---|---|---|---|---|
| Navigation | rail de 72 px à libellés | rail de 64 px à libellés | rail de 64 px | dans l'établi : ← et menu ⋯ (mêmes entrées) |
| Barre du haut | ≤ 48 px : module · fil chantier › dossier › nom (bouton) · indice · état d'enregistrement en texte · « Note · <gouvernant> <taux> » · actions avec libellés | ≤ 48 px : actions en icônes avec `aria-label`, état en court « ✓ 17:08 » | ≤ 48 px | en-tête de 56 px : nom, puis en 2ᵉ ligne point d'état · fil ; pastille de la note ; ⋯. Onglets de 44 px |
| Liste | 260 px (200–400, réglable) | 240 px | 270 px, en bas à gauche | onglet Données |
| Toile | **≥ 740 px** de large | **≥ 540 px** | en haut sur toute la largeur, **≥ 470 px** de haut | onglet Dessin ; feuille ouverte : toile ≥ 300 px de haut |
| Panneau | 360 px (300–520) : **Propriétés** en haut, **Résultat** en bas, poignée de partage (défaut 60/40) | 320 px, empilé | en bas à droite, onglets Propriétés / Résultat (même composant) | Propriétés : remplace la liste (« ‹ Liste ») ou feuille ≤ 45 svh sur le dessin. Résultat : onglet |
| Réduit | liste et panneau en bandes de 40 px (pastilles numérotées touchables, η vertical) ; Propriétés ou Résultat réduits à leur en-tête de 32 px | idem | liste réduite à 60 px ; toile à 176 px | — |

- **Densité.**
  - Pointeur fin : ligne de liste ≤ 28 px ; champ de 30 px ; pas d'une ligne à l'autre ≤ 36 px ; en-tête d'étape ≤ 30 px ; marge du panneau 12 px.
  - `pointer: coarse` : toute cible (ligne, bouton, segment) ≥ 44 px. Les segments sont mesurés aujourd'hui à 36-38 px.
  - Téléphone : champs en police de 16 px.
- **Persistance.**
  - Largeurs, partage et réductions : `localStorage` `stab.etabli.v1`, sous try/catch.
  - Choix et ouvertures : `sessionStorage` `stab.choix.<idNote>`, purgé après `ajouter` ou `supprimer`.

**La liste** (`js/liste.js`), dérivée du DOM rendu par le moteur :

1. **Racine Note** : nom, fil, indice, `resume`. Elle porte aussi les nœuds `.sec[data-parent="note"]`.
2. **Éléments** : `.ni[data-activer]`, avec titre, détail et taux. Le menu ⋯ est **toujours visible au doigt**. « + Ajouter » et « Copier » viennent de `[data-ajouter]` et `[data-dupliquer]`. Un élément inactif envoie `activer`.
3. **Sous l'élément actif** :
   - les étapes `.sec[data-sec]` dans l'ordre (numéro, `.tt`, `.sum`, `data-taux`) ;
   - les objets `.obj[data-sel]`, imbriqués selon `data-parent` ou leur ancêtre, sur **trois niveaux au plus** sous la note ;
   - « Options · N », replié, avec les étapes `data-secondaire`. Si `data-modifie > 0`, la ligne affiche « Options · n modifiée(s) » en accent ;
   - le nœud **Résultat** : les objets nés du calcul (`#resultat [data-sel][data-nom]`).
4. **Clavier** : `role="tree"`, focus itinérant (↑↓←→, Entrée, Échap).
5. **Taux d'une étape** : `data-taux` posé par le moteur, sinon le plus grand des `.pct` des lignes `#resultat .el[data-noeud~=clé]`. L'écran choisit parmi des taux nommés, il ne recalcule rien.

**Le panneau** (`js/panneau.js`) : **un seul DOM** (`#donnees`, `#resultat`), filtré par la CSS, jamais copié.

- Il est **déplacé** entre ses hôtes (inspecteur, page Données ou feuille au téléphone), jamais recopié.
- **Élément choisi** : toutes ses étapes non secondaires, puis « Options » replié. C'est l'équivalent du cadre Données d'aujourd'hui, et c'est la vue d'ouverture.
- **Étape ou objet choisi** : lui seul, avec « ‹ <élément> · tout voir ».
- **Pied** : « ‹ Précédent · k/n · Suivant › », dans l'ordre de la liste.
- **Champ désigné par le dessin** : encadré, et mis au focus seulement à la souris ou au clavier, jamais au doigt.
- **Résultat** :
  - le `#resultat` complet de l'élément ;
  - en-tête « Résultat · <élément> · <statut_html> » ;
  - les lignes `.el[data-noeud~=X]` du nœud choisi prennent `.et-lie` (pas `.sel` : `.el.sel` sert déjà d'ancre aux sélecteurs, `etabli.js` L302 et L315) ;
  - interrupteur « Liés seulement » ; badge « N avis ».
  - La barre du haut porte le verdict de la **note**, nommé « Note · … ». Les deux verdicts ne se confondent plus.
- **Fenêtre de calcul** : posée sur la liste et la toile. **Sélecteur** (`data-pk`) : ancré à gauche du panneau, à hauteur de sa ligne. Au téléphone, l'un et l'autre en feuille plein écran.

**La sélection par le dessin** (`js/selection.js`) : on reprend telle quelle la spécification de l'enquêteur « sélection » (§1 et §2).

- **Contrat avec le moteur** : `data-sel` et/ou `data-champ` sur un `<g>` qui enveloppe le dessin.
- **Geste** : on lit `pointerdown` puis `pointerup`, **jamais `click`**.
  - Seuil au-delà duquel l'appui devient un glisser : 5 px à la souris, 10 px au doigt.
  - Rayon de prise : 6 px à la souris, 12 px au doigt. « Aussi ici » si un second candidat est à moins de 4 px.
  - Pas de bascule, pas de `preventDefault`, pas de capture du pointeur.
- **Après une réponse** : la chaîne de repli `[id, ancêtres, étape]` reprend la sélection ; `rafraichir()` est appelée à la fin d'`appliquer`.
- **Classes du dessin** : `sel` sur la cible, `sel-dans` sur ses descendants.
- **Vue** : une pastille « Voir sur : <vue> » apparaît quand la cible est dans une autre vue ; la vue ne bascule jamais d'elle-même.
- **Clavier** : un seul arrêt de tabulation par dessin.
- **Toile** (`toile.js`) : garde du double appui en L123 ; nouvelle fonction `montrer(rect)`.
- **Téléphone** : un appui sur le dessin ouvre la feuille (≤ 45 svh), sans changer d'onglet.
- **Plein écran** : la feuille passe au-dessus (z-index > 45) ; ouvrir un `data-pk` quitte d'abord le plein écran.
- **« Repérer »** dans la barre du zoom montre toutes les cibles.

**L'enregistrement** : le texte reste visible aux trois formats ; la classe `.et-enr` n'a aucune règle de masquage.

| État | Barre du haut | Téléphone, 2ᵉ ligne |
|---|---|---|
| Neuf, rien saisi | « ○ Pas encore enregistré · le sera à la première saisie » | point gris |
| Non classée | « ● Enregistré · Non classée · 17:08 » + bouton principal « Enregistrer comme note… » | point vert |
| Classée | « ● Enregistré · 17:08 » ; le fil ouvre « Ranger… » | « ✓ 17:08 · Réalex › Fondations » |
| Échec | « ⚠ Non enregistré : stockage plein » en rouge, **persistant**, + message unique avec « Exporter .stab » | point rouge et texte rouge |

**Le démarrage vu par l'utilisateur.**

- Le voile plein écran disparaît : `construire.py` L311-313, `etabli.js` L497-508, `site.css` L107.
- Un bandeau posé sur la toile : « Le moteur de calcul démarre · aperçu en lecture seule », avec un chronomètre.
- Note existante : sa vignette et son `resume`, avec la liste en squelette — **jamais les valeurs par défaut**.
- Calcul neuf : l'aperçu `moteur/apercus/<clé>.json`, rendu inerte (`inert`).
- Erreur : bandeau rouge avec « Recharger ».

### B.4 Lots de réalisation (dans l'ordre)

Ordre : **0 → 0 bis → S1 → S2 → S3 → S4 → M1 à M5 → C → V**.
- M1 à M5 peuvent démarrer après S2 ; M3 dépend de M2 (le kit des schémas).
- C peut démarrer après S3.
- Fusions séquentielles, à cause de `verifier_site.js`.
- Efforts : SUPPOSÉS, environ 50 j au total.

---

#### LOT 0 — Référence figée et mesures « avant » (≈ 0,5 j)

- **Objet.** Apporter le gel et mesurer l'existant avant tout changement.
- **Fichiers.**
  - Reprendre le commit db5fb75 de `/home/user/chantier_golden` (`tests/test_reference_refonte.py`, `tests/reference_refonte/{cas.py, mutations.py, <18 clés>.json}`) sur la base e3a0ca0.
  - Porter l'instrumentation de `/home/user/chantier_mesure` (`site/web/js/travailleur.js`, `moteur.js`, `etabli.js` : marques `performance.mark` et option `?sonde=1`), **sans changer aucun comportement**.
- **Ce qui manque au gel**, et qui est couvert ailleurs :
  - Pyodide → S4 (parité des aperçus) ;
  - stockage JavaScript → lots 0 bis et C ;
  - titres et résumés du cadre Données, non figés exprès → chaque lot de module liste dans son commit les libellés d'étape changés.
- **Mesures versées** (une mesure ne se réécrit jamais) :
  1. `docs/site/mesures/2026-11-07_demarrage_avant.md`.
     - Médianes de 3 tirages, profil Chromium sur disque : à froid, 2ᵉ visite, et passage d'un module à l'autre, pour poutre, assemblages, plancher et age, au PC et au téléphone émulé, plus la 2ᵉ visite des 18 modules.
     - Décomposition : wasm, `loadPyodide`, numpy, déballage, routeur, import du module, premier calcul, peinture.
  2. `docs/site/mesures/2026-11-07_couverture_avant.json`, par module et par format : l'ensemble des `data-chemin` présents dans `#donnees`.
- **Tests qui doivent rougir.** `tests/reference_refonte/mutations.py` rejoué : 7/7 conformes (f_yd ×1,001 → 12 cas rouges ; avertissement retiré → rouge ; balisage seul → vert).
- **Acceptation.** `/usr/bin/python3.12 lancer_tests.py` VERT, avec « 239 OK » pour la référence ; les deux mesures sont versées.

#### LOT 0 bis — Défauts non conservatifs (passe devant tout, ≈ 0,5 j)

1. **Échec d'enregistrement muet.**
   - Nouvel indicateur `.et-enr`, visible à 1 440, 1 280, 820 et 390 px.
   - En échec : texte rouge persistant et message avec « Exporter .stab ». Au téléphone, il est porté par la 2ᵉ ligne de l'en-tête.
   - Fichiers : `site/web/css/site.css`, `js/etabli.js` L42-48, `construire.py` (barre et `.mtop`).
2. **Mémoire des notes abîmée, écrasée en silence** (VÉRIFIÉ par le jury).
   - `toutes()` doit distinguer « absente » de « illisible ».
   - Si illisible : copier le texte brut une fois dans `stab.notes.v1.illisible.<horodatage>`.
   - Écrire de nouveau **seulement si cette copie a réussi** ; sinon refuser l'écriture (`ecrire` rend `false`) et afficher « Non enregistré : la mémoire des notes est illisible ».
   - L'accueil le signale. Fichier : `js/notes.js` L11-27.
- **Tests qui doivent rougir sans la correction.**
  - `verifier_site.js`, filtre `enregistrement` : `Storage.prototype.setItem` lève une exception → le texte d'échec doit être visible (`checkVisibility({visibilityProperty:true, opacityProperty:true})`, boîte > 0, couleur d'erreur) aux 4 largeurs. Mutation : remettre `display:none` sous 1 360 px → rouge.
  - `tests/test_site_web.py`, essai joué par node comme celui de `texteTaux` (L159-179) : 30 notes, JSON abîmé, puis `ecrire` → le texte d'origine survit (clé d'origine ou copie) et `ecrire` rend `false` si la copie échoue. Mutation : rétablir `return {}` → rouge.
- **Acceptation.** Les deux contrôles sont verts, les mutations rouges, la référence à 239 OK.

#### LOT S1 — Socle : contrat moteur → écran (Python, sans changement visible, ≈ 2,5 j)

**Fichiers.** `api_web/commun.py`, `api_web/routeur.py`, les 18 `api_web/<clé>.py` (seulement `secondaire=True` selon D5), `tests/test_api_web_modules.py`, `docs/site/MODULES_SITE.md` §3-4.

**Briques** :
- `section(n, titre, resume, corps, ouvert=True, cle="", *, secondaire=False, parent="", taux=None, niveau="")` → ` data-secondaire="1" data-parent="…" data-taux="…" data-niveau="…"`, placés après `data-sec` (L407-411).
- `objet(sel, nom, corps, *, resume="", parent="", indice="", taux=None, niveau="", retirer="", copier="")` :
  - sortie `<div class="secard obj" data-sel="…/" data-nom="…" data-resume="…">`, avec l'en-tête `.sh2` (la CSS existante reste valable) ;
  - `data-sel` est un préfixe de chemin terminé par `/` ;
  - noms échappés par `esc`.
- `repli(cle, titre, resume, corps)` → `<details class="repli" data-repli><summary>titre <span class="rs">résumé</span></summary>…</details>`. Le résumé est composé par le moteur. `<details>` est déjà prouvé neutre par la mutation d1 du gel.
- `liste_elements(items, actif, *, titre, ajouter, copier)` : produit le balisage `.ni` d'aujourd'hui, plus `data-nom`, `data-taux` et `data-niveau`. La migration des 18 copies se fait dans les lots M.
- `ligne_res(…, noeud="", sel="", nom="")` (L234) : ` data-noeud="a b"` (liste séparée par des espaces) ; ` data-sel`/`data-nom` pour un objet qui n'existe qu'au Résultat.
- `cible(contenu, sel="", champ="", zones=(), anneau=False)` et `zone(x, y, w, h)` : `<g>` enveloppant, avec `<rect class="zone" … fill="transparent"/>` en dernier enfant. Sans `sel` ni `champ`, elle rend `contenu` tel quel.
- **Garde des options : `marquer_modifies(donnees, etat, neuf)`**, appelée par le routeur juste après `mod.resultat(etat)`, avec `neuf = mod.nouvel_etat()`. **Aucun rendu de plus.**
  - Pour chaque `.sec[data-secondaire]`, elle compte les `data-chemin` dont la valeur dans `etat` diffère de celle dans `neuf` (chemin découpé sur `/`, les segments numériques étant des indices).
  - Elle écrit ` data-modifie="n"` sur la section et ` data-modifie="1"` sur le champ.
  - Un chemin absent de `neuf` n'est pas compté ; il est listé dans une table nommée `CHEMINS_SANS_DEFAUT`, justifiée.

**Tests qui doivent rougir** (`tests/test_api_web_modules.py`) :
- **K1** : chaque brique appelée sans les nouveaux paramètres égale une **copie nommée** de l'ancienne sortie. Mutation : un attribut vide écrit par défaut → rouge.
- **K2** : chaque `data-chemin` des Données est dans exactement une `.sec`.
- **K3** : l'ensemble des `.sec[data-secondaire]` de chaque module est égal à la table `SECONDAIRES` (D5), et chaque module garde au moins une étape non secondaire. Mutations : retirer un drapeau → rouge ; mettre `tr` en secondaire → rouge.
- **K4** : sur `nouvel_etat`, tous les `data-modifie` valent 0. Après le `modifier` de la table `MODIFS = {clé: (chemin, valeur)}`, la section concernée vaut au moins 1. Mutation : `marquer_modifies` rend le HTML inchangé → rouge.
- **K5** : aucune primitive SVG ne porte `data-sel` ou `data-champ` (motif `<(rect|circle|path|polygon|polyline|line|text|ellipse)\b[^>]*\bdata-(sel|champ)=`).
- **K6** : piège avec un guillemet et `<` dans un nom (nom de note, nom d'appui du plancher) : tous les attributs restent `[^"<>]*`.

**Acceptation.**
- Référence à 239 OK.
- Pour les 18 réponses `nouveau`, `donnees` privé des attributs `data-(secondaire|modifie)` est identique à celui produit sur e3a0ca0.
- Coût médian de `marquer_modifies` ≤ 1 ms par requête en CPython (mesuré, valeur reportée dans le commit).

#### LOT S2 — Socle : l'établi (liste, panneau, sélection) sur les 18 modules (≈ 5 j)

**Fichiers.**
- Nouveaux : `site/web/js/liste.js`, `js/panneau.js`, `js/selection.js`, `css/etabli.css`.
- Modifiés :
  - `js/etabli.js` (affichage, `appliquer` → `selection.rafraichir()`, Échap en L391) ;
  - `js/toile.js` (L96-102 et L123, `montrer(rect)`) ;
  - `construire.py` (gabarit L268-317 : barre, colonnes, onglets du téléphone) ;
  - `css/site.css` ;
  - `tests/test_site_web.py` (listes de fichiers L112-118 ; test des préfixes CSS) ;
  - `verifier_site.js` (filtres `etabli[:<clé>]` et `selection[:<clé>]`) ;
  - `docs/site/CAHIER_DES_CHARGES.md` §8.
- **Poutre pilote** : `api_web/dessin_poutre.py` enveloppe b, h, c, les barres inférieures et supérieures et l'étrier par `C.cible(champ=…)`.

**Changements.**
- Tout B.3, en N0 pour les 18 modules : liste tirée des `.sec` existants, Options grâce à S1.
- Correction des deux verdicts (barre du haut « Note · … », en-tête « Résultat · <élément> · … »).
- Fil d'Ariane non coupé quand la place existe : aujourd'hui, « Sans chantier » obtient 77 px pour 85 nécessaires, alors que 128 px sont libres à 1 440.

**Tests qui doivent rougir** (navigateur, 3 formats, tablette avec `hasTouch`, sur la note neuve et sur l'état final de deux cas de `tests/reference_refonte/<clé>.json` semés comme notes) :
- **E1 couverture** : choisir chaque nœud de la liste, puis faire l'union des `[data-chemin]` **visibles** du panneau (ou de la feuille au téléphone). Elle doit égaler tous les `data-chemin` de `#donnees`, hors boutons `+` et `-`.
  - Mutations : Options vidé par la CSS → rouge ; `liste.js` qui saute les étapes secondaires → rouge.
- **E2** : la même chose avec la liste réduite, par les pastilles.
- **E3** : chaque `.el` est visible dans le Résultat ; chaque `data-pk` et chaque `data-calcul` ouvre son volet, visible.
- **E4 géométrie.**
  - Largeurs de la toile : ≥ 740 px à 1 440 × 900 ; ≥ 1 280 px quand les deux panneaux sont réduits ; ≥ 540 px à 1 180 × 820.
  - Tablette portrait 820 × 1 180 : toile ≥ 470 px de haut.
  - Téléphone : feuille ≤ 0,45 × la hauteur de la fenêtre, cible au-dessus de la feuille.
  - `scrollWidth ≤ innerWidth` partout.
- **E5 densité** : les seuils de B.3, au pointeur fin et au doigt (≥ 44 px ; rouge aujourd'hui sur les segments).
- **E6 verdicts** : mutation « afficher `resume` dans l'en-tête du Résultat » → rouge.
- **N1 à N13** (rapport « sélection ») sur la poutre, avec ses mutations : retirer le SEUIL → N3 ; retirer la garde du double appui → N4 ; retirer le RAYON → N12 ; retirer la media query `(hover:hover)` → N6 ; retirer `rafraichir()` → N7.
- **Python** :
  - chaque classe définie dans `etabli.css` commence par `et-`, ou figure dans la liste nommée `REPRISES` ;
  - la règle `[hidden]` est présente ;
  - T1 à T8 sur la poutre, avec `MIN_CIBLES["poutre"]`.

**Acceptation.** E1 à E6 et N1 à N13 au vert aux trois formats ; référence à 239 OK ; captures des 18 modules aux trois formats regardées, défauts corrigés ; les rouges de l'ancien filtre `formats:` classés « suit ».

#### LOT S3 — Socle : ouvrir un calcul directement, l'enregistrer comme note (format v1, ≈ 2 j)

**Fichiers.**
- `construire.py` :
  - L206-245 : tuiles `<a class="tile" href="…" data-module="…">`, section renommée « Nouveau calcul » ;
  - **retirer** la fenêtre (L218-227), le bouton (L234) et `data-nouvelle-module` (L212) ;
  - L268-271 : fil = bouton, indice réel.
- `js/accueil.js` : retirer L48-71 ; texte de l'état vide en L25.
- `js/etabli.js` L22-48 et L79-96 : cycle de vie de la note.
- `js/notes.js` : écrire les caches sans toucher `modifie`.
- Nouveau `js/rangement.js` : panneau partagé (nom, chantier et dossier en texte avec liste de saisie, indice).
- `verifier_site.js` et `tests/test_site_web.py`.

**Comportement.**
- **Sans `?note=`** : calcul en mémoire, rien d'écrit, adresse inchangée.
- **Première action qui modifie** (`modifier`, `choisir`, `ajouter`, `dupliquer`, `supprimer`, météo, renommer, ranger ; mais pas `nouveau`, `calculer`, `activer`, `selecteur` ni `telecharger`) :
  - la note est écrite « Non classée » ;
  - `replaceState(?note=id)` ;
  - un message unique propose « Enregistrer comme note… ».
- **`?note=` d'un autre module** : `location.replace` vers le bon établi.
- **`?note=` inconnue** : message persistant « Cette note n'est pas dans ce navigateur (supprimée, ou enregistrée ailleurs) » avec « Importer .stab », puis calcul neuf non enregistré.
- **`?chantier=&dossier=&nom=`** : accepté, la note naît à la première saisie.
- **Rouvrir sans rien changer** ne réécrit pas `modifie`.
- **Panneau « Enregistrer comme note… » / « Ranger… »** :
  - ouvert par la barre, par le fil et par le nœud Note ; au téléphone, par le menu ⋯ et la 2ᵉ ligne de l'en-tête, en feuille du bas ;
  - boutons « Enregistrer comme note », « Exporter .stab », et « Jeter ce calcul » (avec confirmation) tant que la note n'est pas classée.

**Tests qui doivent rougir** (filtre `ouverture`, 3 formats) :
- les 18 tuiles sont des `a[href]`, et le clic n'est pas intercepté (`defaultPrevented === false`) ;
- 0 note avant toute saisie, encore après moteur prêt + 3 s ;
- 1 note après une saisie, et le rechargement garde le même id et les mêmes valeurs ;
- message pour une note inconnue ; redirection pour un autre module ; `modifie` inchangé à la réouverture.
- Mutations : écrire dans `appliquer` (l'ancien comportement) → rouge ; retirer la redirection → rouge.

**Harnais réécrit dans le même commit** :
- `nouvelle()` (L78-90) et `neuve()` (L481-495) : tuile, puis panneau de rangement ; **garder le chantier au nom long** ;
- L70-73 (compte `data-module`), L551-555, L985-988 (`etatNote` : ajouter une saisie), L1420 ;
- `test_site_web.py` L116-121 : plus d'identifiants `nn-*`, plus de `#nouvelle`.

**Acceptation.** Filtre `ouverture` vert aux trois formats ; mutations rouges ; référence à 239 OK.

#### LOT S4 — Socle : démarrage mesuré, en arrière-plan, réutilisé (≈ 4 j)

**Fichiers.** `api_web/registre.py`, `site/web/js/travailleur.js`, `js/moteur.js`, nouveau `js/app.js`, `js/etabli.js` et `js/accueil.js` en fonctions d'ouverture, `construire.py`, `css/site.css`, `tests/test_site_web.py`, nouveau `tests/test_api_web_registre.py`, `verifier_site.js` (filtre `demarrage`).

**a. Registre paresseux.**
- `module(cle)` : clé conforme à `^[a-z][a-z0-9_]*$`, fichier `<clé>.py` présent ; il ne lit **que** ce fichier (`MODULE == clé` et `PAGE` présent) et garde le résultat en mémoire.
- `cles()` est inchangé pour `construire.py`.
- Mêmes refus qu'aujourd'hui : `commun`, `routeur`, `registre`, `_x`, `../poutre`, `None`.
- Test : un espion sur `registre._lire` → `module('poutre')` lit au plus 1 fichier, puis 0 au deuxième appel. Mutation : appeler `cles()` dans `module()` → rouge.
- Gain mesuré par l'enquêteur : 10,8 → 3,2 ms par requête pour la poutre.

**b. Aperçus.**
- `construire.py` écrit `moteur/apercus/<clé>.json` = la réponse `traiter({"module": clé, "action": "nouveau"})` de CPython, sauf `SANS_APERCU = {"age"}` (réponse qui dépend du jour, liste nommée).
- Test : chaque aperçu est égal, octet pour octet, à la réponse du routeur. Il rougit si un module change sans reconstruction.

**c. App unique, un seul travailleur.**
- **Coquille** : `construire.py` écrit **la même coquille, octet pour octet**, dans `index.html` et dans chaque `<page>.html` (copies nommées ; un test les dit identiques). `app.js` choisit l'écran d'après `location.pathname`.
- **Navigation** : un `a[href]` du même site, sans touche de modification, sans `target` ni `download`, passe par `history.pushState`. Écouter aussi `popstate` et `pageshow`.
- **Moteur** : un seul `Moteur`, créé au premier `requestIdleCallback` (délai 2 000 ms) ou à l'ouverture de l'établi. `travailleur.js` ne charge plus numpy au départ.
  - Nouveau message `{type:"prechauffer", module}` : `import api_web.<clé>`, plus numpy si `avec_numpy` le demande.
  - Il est envoyé sur `pointerenter`, `focusin` et `pointerdown` d'une tuile, et à l'ouverture de l'établi.
  - Les modules restent importés ; numpy est chargé une fois au plus.
- **`etabli.js`** : `export function ouvrirEtabli(hote, {module, params}) → {fermer}`, sans effet au chargement du fichier ; écouteurs délégués sur l'hôte et retirés par `fermer`.
- Les adresses `/poutre?note=…`, `?chantier=&dossier=&nom=` et `?dossier=<id>` restent valides.

**d. Affichage du démarrage** : celui de B.3 ; le voile est retiré.

**Tests qui doivent rougir** (`demarrage`, au PC et au téléphone) :
- un seul `loadPyodide` sur le parcours accueil → poutre → assemblages → poutre ;
- après l'ouverture de la poutre, une requête sonde sur `sys.modules` ne trouve aucun autre `api_web.<module>` et pas numpy ;
- sur l'accueil pendant le démarrage, aucune tâche longue de plus de 200 ms sur le fil de la page (`PerformanceObserver` `longtask`) ;
- passage d'un module à l'autre ≤ 0,5 × démarrage à froid du même tirage ; ouverture depuis l'accueil après 3 s d'attente ≤ 0,5 × lien direct ;
- lien direct : aperçu ou vignette visible ≤ 300 ms après `DOMContentLoaded`, champs inertes jusqu'à la première réponse ;
- la première réponse de Pyodide égale l'aperçu (CPython) pour les 17 modules : cela détecte un écart entre les deux interpréteurs ;
- retour arrière : aucune nouvelle navigation de document.
- Mutations : un `Moteur` par établi → rouge (2 `loadPyodide`) ; `prechauffer` qui importe tout → rouge ; voile remis → rouge.

**Acceptation.** Les contrôles ci-dessus au vert. Mesure `docs/site/mesures/AAAA-MM-JJ_demarrage_apres.md` versée selon le protocole du lot 0. Le vrai téléphone reste SUPPOSÉ, à un facteur 2 à 4 : sa mesure s'affichera dans Paramètres (lot C).

---

#### Lots de modules (M1 à M5)

**Ce que chaque module reçoit** :
1. `C.liste_elements` au lieu de la liste écrite à la main.
2. Ses sous-objets en `C.objet` (préfixe `…/`), avec le regroupement indiqué.
3. `noeud=` sur les lignes et groupes du résultat, et `sel=`/`nom=` pour les pièces nées du calcul.
4. Ses cibles de dessin.
5. Ses entrées dans les tables nommées de `tests/test_api_web_modules.py` :
   - `MIN_NOEUDS[clé]` ;
   - `MIN_CIBLES[clé]`, ou `SANS_CIBLE`, liste qui rougit si un module listé gagne une cible ;
   - sortie de la liste d'exemptions `SANS_NOEUD` de A3.

**Tests du contrat**, qui rougissent dès qu'un module y entre :
- **T1** : chaque `data-sel` d'un dessin se résout dans les Données ou le Résultat.
- **T2** : chaque `data-champ` est un `data-chemin` des Données, et pas un bouton.
- **T3** : identifiants uniques.
- **T4** : un identifiant en `…/` contient tous les champs de son préfixe.
- **T5** : chaque `.obj` a un `data-nom` non vide.
- **T6** : minimums de cibles.
- **T7 et T8** : comme K5 et K6.
- **A3** : chaque `.el` a un `data-noeud` qui existe, ou son module figure dans `SANS_NOEUD`.
- Après `ajouter` puis `supprimer` du premier sous-objet, chaque `data-sel` se résout encore.

**Acceptation commune à M1-M5.**
- Référence à 239 OK, sans regel, `etat` inchangé.
- Tests propres du module au vert.
- `etabli:<clé>` et `selection:<clé>` verts aux trois formats.
- Captures regardées.
- Les libellés d'étape changés sont listés dans le commit.
- Les minimums sont figés à la valeur mesurée. Une valeur sous le plancher annoncé doit être expliquée.

#### M1 — Béton à sections : poutre (pilote), dalle, prédalle (≈ 5 j)

- **poutre** (`api_web/poutre.py` L147-195, liste L351-362, `dessin_poutre.py`).
  - Regrouper `so` et `arm{i}` en **un objet par section** (`sections/i/`), qui porte sollicitations, lits inférieurs et supérieurs, étriers et pas.
  - Étapes : 1 Géométrie et matériaux (`pt`), 2 Sections (A, B…, puis « + Ajouter une section »), Options (Méthode).
  - Nœuds du résultat : Hauteur → `pt` et `sections/i/` ; inf, sup, τ et étriers → `sections/i/`.
  - Cibles : béton → `pt` ; b et h → leurs champs ; « c = 30 » → `enrobage` ; barres → `sections/i/inf/` et `…/sup/` ; étrier (anneau) → `sections/i/` ; efforts → `sections/i/`.
  - **Corriger le `title` qui affiche du HTML brut** (`poutre.py:186`). Test : aucun `&lt;` dans un attribut `title` des 18 réponses `nouveau` — rouge aujourd'hui (1 occurrence, VÉRIFIÉ).
  - Planchers : `MIN_CIBLES` ≥ 8 par section ; `MIN_NOEUDS` ≥ 6.
- **dalle** (L489-552 ; verrou L471-486 ; `_face` L446 ; `_couche` L418).
  - Même regroupement : objet `sections/i/` = sollicitations + 4 faces (→ couches en lignes).
  - `me` (γs et jeu, communs à la note, L538-541) passe en `parent="note"`.
  - Cibles : coupe et plan selon le rapport « sélection » §3, environ 29 (SUPPOSÉ).
- **prédalle** : la même chose. **Fusionner** ses copies `_couche`, `_face` et `_formulaire_verrouille` (`predalle.py` L506/538/568) avec celles de `dalle.py` (L418/446/471), en refactorisation pure : référence et parité de la prédalle au vert.

#### M2 — Assemblages (7 configurations + pied manuel) et kit des schémas (≈ 4 j)

- **Kit** `acier/assemblages/poutre_poutre/doubles_cornieres/schemas.py` :
  - `Options.etiquettes: Optional[dict] = None` (L77-111) ;
  - `_prim_svg` (L1311-1337) écrit `data-sel`/`data-champ` après `data-dim` ;
  - `Feuille.label` (L374-383) lit les étiquettes ;
  - une seule fonction `_piece(...)` remplace les 16 copies locales de `piece()` (tableau du rapport « sélection » §2.3) ;
  - **avec `None`, la sortie reste identique à l'octet** : parités Streamlit et note PDF au vert, et nouveau test d'octets sur les 21 vues.
- **Site** : `opt.etiquettes` vient de `S.CLE_PAR_COTE` et d'une table des pièces (beamS → `v/prof_S`, cleat → `v/corn_u`, plate → `v/tp_u`, bolts → `v/boulon_u`, weld → `v/a_w`, grout → `v/t_g`, pedestal → `v/bc_u`, recut → `v/d_nt`).
  - Points d'insertion : `assemblages_poutre.py` L121-131, `assemblages_poteau.py` L120-131, `assemblages_pied.py` L566-572.
  - Les copies nommées suivent dans le même commit : `test_api_web_assemblages.py` L545-550, `_poteau` L597-601, `_pied` L700-705.
- **Objets** :
  - `pt` scindée en Principale et Secondaire ;
  - `bo` : objets Groupe S et Groupe P, plus `C.repli` « Réglages des boulons » (trou, catégorie, filet, d₀) ;
  - pied manuel : `tiges[]`, `elements[]`, `raidisseurs[]`.
- **Résultats** : table `NOEUD_PAR_PIECE` (volets de `synthese.ELEMENTS` : `plat-d-ame` → `pl`, `boulons` → `bo`, `poutre-*` → `pt/…`).
- **Ne pas casser** les découpes sur `data-sec` de `test_api_web_assemblages_poteau.py` L353-365.
- **Acceptation** :
  - 168 cotes posées sur 21 vues, dont **≥ 139 avec `data-champ`** (mesuré par l'enquêteur) ;
  - N12 : ≥ 97 % de bons choix au téléphone sur la grille ±8 px de l'élévation des doubles cornières ;
  - les 39 cas de référence des assemblages au vert (≈ 35 s).

#### M3 — Garde-corps, soudures, cornière : les autres utilisateurs du kit (≈ 4,5 j, après M2)

- Une table `CHAMP_PAR_COTE` **par vue** dans `api_web/garde_corps.py`, `soudures.py` et `corniere.py` : « ch0 » désigne deux choses selon la vue.
- **garde-corps**
  - Objets : `geometrie/troncons/k/`.
  - Pièces nées du calcul (montant, main courante, lisse basse, barreaux, platine, soudure) : objets du Résultat, avec leur `data-pk`. Elles apparaissent dans la liste sous Résultat.
  - Tronçons, Remplissage et Fixation restent des étapes.
  - Plancher : ≥ 34 cotes.
- **soudures**
  - En C8 : `cordons[]` et `params/elements/k/` en objets.
  - L'identification passe en `parent="note"`.
  - Plancher : ≥ 48 cotes de C1 à C8.
- **cornière** : la `.secard` reste une étape (fiche de propriétés). Plancher : ≥ 10 cotes avec champ.

#### M4 — Plancher, sol, taux : collections et plans (≈ 5 j)

- **plancher**
  - `dessin_plancher._INTERACTION` (L33, appliqué L53) devient une **traduction** :
    - `charge:`/`engin:<ref>` → `data-sel="charges/<i>/"` ;
    - `rep:<ref>` → `appuis/<k>/` ;
    - `fil:<id>` → `filieres/<j>/` ;
    - `data-key="L"` → `data-champ="plancher/L"` ;
    - `sens:` → `plancher/sens` ;
    - `n_elements:` → `plancher/hourdis/n_elements`.
  - `data-drag`, `class="ed"` et `<title>` restent retirés.
  - Objets : charges, filières, appuis. `bi` va sous Note.
  - `test_api_web_plancher.py` L300-301 reste vrai ; on ajoute un contrôle positif.
  - Planchers de groupes : 13 en hourdis, 11 en dalle, 2 en prédalle (mesurés).
- **sol** : sondages en objets (clés `s{id}`, déjà stables) ; couches en lignes ; cibles semelle, B, D, q et couches.
- **taux** : `familles/k/` et `zones/k/` en objets ; `types[]` en lignes ; verdict en kg/m³ sur l'élément.

#### M5 — Les 8 modules simples (≈ 3,5 j)

Flambement, profil, bois, catalogue, maturité (age), armatures, enrobage, données béton. Tous passent par `liste_elements` et `noeud=`. Cibles :

| Module | Cibles | Plancher |
|---|---|---|
| flambement | axes y-y et z-z en objets ; N, L, axes, profil | 5 |
| profil | étiquette des efforts → `ef` (la courbe sert à choisir, pas à sélectionner) | 1 |
| bois | G, Q, appuis, L, `lim_inst`/`lim_fin`, section | ≈ 7 |
| age | courbe → `bt`, te → `age`, seuil → `se`, essai → `es`, comparateur | — (pas d'aperçu) |
| enrobage | `diam_barre`, `diam_etrier`, `delta_dev` | 3 |
| catalogue | `SANS_CIBLE` ; `vtab` et `l2` intacts | — |
| données béton | `SANS_CIBLE` ; `vtab` et `l2` intacts | — |
| armatures | au mieux la nappe → `di`, sinon `SANS_CIBLE` | — |

À la fin de M5, `SANS_NOEUD` ne contient plus que les modules de consultation, liste nommée et justifiée.

---

#### LOT C — Accueil, chantiers et dossiers, navigation, états vides, migration sans perte (≈ 6 j)

**C'est une migration de FORMAT** : elle part comme proposition et attend l'accord de l'ingénieur avant d'être déployée.

**Fichiers.**
- Nouveaux : `js/classement.js` (modèle v2 et `migrer()` pure), `js/chantiers.js`, `js/parametres.js`.
- Modifiés : `js/notes.js` (délègue), `js/rangement.js` (listes), `js/accueil.js`, `js/commun.js` (`cablerRail` et message « bientôt »).
- `construire.py` :
  - rail L153-163 et `.mnav` L166-174 ;
  - volet L177-188 et tuiles L207-217, par `familles_ouvertes()` ;
  - `modules_js()` L339-347 exporte `FAMILLES` ;
  - écrans `chantiers` et `parametres` dans la coquille ;
  - L240 : « sur cet appareil, dans ce navigateur » au lieu de « sur ce PC ».
- CSS ; `tests/test_site_web.py` (L138-142 et nouveaux tests) ; `verifier_site.js` (L553-555 et nouvelles sections) ; `docs/site/CAHIER_DES_CHARGES.md` (§1 bis, L25-31, L61-68, L144-146) ; `site/web/README.md`.

**Modèle.**
- Une clé `stab.v2` = `{version:2, chantiers:{id:{id,nom,cree,modifie}}, dossiers:{id:{id,chantier,nom,…}}, notes:{id:{champs v1 inchangés, idChantier|null, idDossier|null}}, supprimes:{id:t}, migration:{depuis,le,notes,illisibles}}`.
- Noms uniques après `trim` et normalisation NFC, sans plier la casse.
- Supprimer un chantier ou un dossier ne supprime **jamais** une note en silence : on donne le compte, et on choisit entre « deviennent non classées » et « supprimer aussi ».

**Migration** `migrer(v1)`, pure et idempotente, jouée au chargement de tout écran :
- chantier et dossier vides → note non classée ;
- chantier seul → racine du chantier ;
- dossier sans chantier → chantier « Sans chantier », que l'on peut renommer ;
- champs inconnus et `etat` gardés à l'identique ; entrée illisible conservée dans `illisibles`.
- **`stab.notes.v1` n'est jamais réécrit** : c'est la sauvegarde, que l'on supprime à la main depuis Paramètres, avec sa taille affichée.
- **Quota** : si v2 ne s'écrit pas, réessayer sans les vignettes (un cache que le moteur refait à chaque ouverture) ; sinon rester en v1, intact, et dire « Mémoire du navigateur pleine : exportez vos notes ».
- **Notes v1 ultérieures** : une note v1 inconnue ou plus récente est refusionnée à chaque chargement (cas d'un ancien onglet), sauf si une suppression plus récente l'interdit.

**Deux onglets.**
- Une écriture relit la mémoire, puis n'écrit que les champs de **cette** note.
- Écouter l'évènement `storage` ; garder les `supprimes`.
- Une note dont le dossier a disparu se lit « non classée ».

**.stab.**
- Note seule : format inchangé (`version:1`), avec chantier et dossier écrits en NOMS. À l'import, chaque nom est retrouvé ou créé ; on propose « Importer dans… ».
- `id_origine` : « déjà importée : remplacer ou garder les deux ? ».
- Nouveaux « Exporter le chantier » et « Tout exporter » en `version:2`.
- Le tri de sécurité existant (`notes.js` L59-69 : textes de 200 caractères au plus, vignette jetée) s'étend aux noms de chantier et de dossier.

**Écrans.**
- **Chantiers** (maquettes `pc_chantiers.png` d'atelier et `chantiers_pc.png` de note) :
  - PC : arbre de 268 px (filtre, Toutes, Non classées, chantiers › dossiers avec leurs comptes, « + Nouveau chantier », « + Nouveau dossier ») ; liste (Note · Module · Taux · Indice · Modifiée · ⋯) ; fiche de 340 px (Ouvrir, Ranger…, .stab, Dupliquer, Supprimer…) ;
  - « + Nouveau calcul ici » : choix du module **par familles**, puis `<page>?dossier=<id>` ;
  - tablette : la fiche en tiroir ; téléphone : descente chantier → dossiers → notes, actions en feuille.
- **Accueil** :
  - « Reprendre » (6 cartes, ⋯ par carte) et « Toutes les notes (n) → » ;
  - « Nouveau calcul » par familles (au téléphone, segments Béton · Acier · Autres) ;
  - bloc Chantiers : arbre cliquable, « + Nouveau chantier », « Non classées » ;
  - bloc « Où sont mes calculs ? » : « Sur cet appareil, dans ce navigateur · aucun serveur, aucun compte » ; avertissement (autre navigateur, fenêtre privée, effacement des données du site) ; jauge « n notes · x ko sur ≈ 5 Mo » ; « Tout exporter », « Importer » ; essai de `navigator.storage.persist()`.
- **États vides** : textes de la proposition « note » §11.2.
- **Navigation** :
  - rail à libellés (Accueil, Chantiers, Modules, Paramètres | Équipe grisée « bientôt » | Thème) ;
  - téléphone : les 5 mêmes entrées, nommées ;
  - plus aucun `data-bientot` muet.
- **Paramètres** :
  - thème ;
  - données du navigateur : espace, export, import, ancienne sauvegarde v1, sauvegarde illisible du lot 0 bis à télécharger ;
  - moteur : dernier démarrage mesuré sur cet appareil, version de Pyodide, empreinte du paquet.

**Tests qui doivent rougir.**
- **Migration** (node depuis `test_site_web.py`), sur des fixtures (chantier+dossier, chantier seul, dossier seul, rien, NFC/NFD, espaces, module inconnu, champs inconnus, entrée illisible) :
  - même nombre de notes, `etat` égal en profondeur, chaque couple (chantier, dossier) retrouvable ;
  - `migrer(migrer(x)) == migrer(x)` ; clé v1 identique à l'octet après migration.
  - Mutations : branche « dossier sans chantier » retirée → rouge ; NFC retiré → rouge ; v1 réécrit → rouge.
- **Quota simulé** : v1 intact.
- **Deux onglets** (navigateur) : supprimé en A puis saisie en B → reste supprimé ; renommé en A puis saisie en B → le nom de A tient.
- **Familles** : l'ordre des tuiles, du volet et de `FAMILLES` de `modules.js` égale `MQ.FAMILLES` filtré par `PAGES`. Mutation : remettre `for k in PAGES` → rouge.
- **Navigation** (L138-142 réécrites) : mêmes entrées dans le même ordre au PC et au téléphone ; libellés visibles à 1 440, 820 et 390 px ; Équipe en `aria-disabled` et hors tabulation.
- **États vides** : avec une mémoire vide, « + Nouveau chantier » est visible et crée un chantier ; aucun « sur ce PC ».
- **.stab** : un ancien fichier s'importe dans un chantier par son nom ; aller-retour v2 ; piège de noms longs ou en HTML assaini.
- **Chantiers, 3 formats** : chantier → dossier → « Nouveau calcul ici » → première saisie → la note est dans le dossier.

**Acceptation.** Tout ce qui précède au vert ; référence à 239 OK ; captures de l'accueil (plein et vide), de Chantiers et de Paramètres aux trois formats, regardées.

#### LOT V — Vérification (≈ 2 j)

1. `/usr/bin/python3.12 lancer_tests.py` VERT. Référence : 239 OK, et `git diff e3a0ca0 -- tests/reference_refonte/` ne montre que l'ajout du lot 0.
2. Mutations rejouées, avec leur résultat dans le rapport : `mutations.py` (7/7) et les tables de S1, S2, S3, S4, C et M.
3. `verifier_site.js` complet, sans rien reconstruire pendant qu'il tourne :
   - PC 1440×900, tablettes 820×1180 et 1180×820 **avec `hasTouch`**, téléphone 390×844 ;
   - filtres `etabli`, `selection`, `ouverture`, `demarrage`, `enregistrement`, `chantiers`, `migration`, `formats`, `modules`, `telecharger`, `meteo`, `piege` ;
   - couverture E1 rejouée sur deux états de la référence par module ;
   - chaque rouge classé.
4. Captures regardées à l'œil : 18 modules × 3 formats, plus ces états :
   - Options avec une valeur modifiée ;
   - sélection depuis le dessin ;
   - feuille du téléphone ;
   - panneaux réduits ;
   - échec d'enregistrement ;
   - démarrage ;
   - Chantiers ;
   - accueil vide.

   Liste des captures relues dans le rapport, défauts corrigés.
5. Mesure `docs/site/mesures/AAAA-MM-JJ_demarrage_apres.md` : même protocole qu'au lot 0, tableau avant/après.
6. Documentation :
   - `docs/site/MODULES_SITE.md` (contrat §3-4 : briques, attributs, tables nommées) ;
   - `CAHIER_DES_CHARGES.md` (§1 bis, §5, §7, §8 et les décisions D1 à D9) ;
   - `site/web/README.md`.
7. Ce qui reste SUPPOSÉ :
   - la mesure sur un vrai téléphone (Paramètres l'affichera) ;
   - le clavier virtuel ;
   - la hauteur réelle de Safari (≈ 390×664) ;
   - les seuils d'appui de 5 et 10 px (figés par N3 et N12, pas mesurés au vrai doigt).
---

## Révision du 08/11/2026 — l'écran de la poutre devient la référence (étape 1, décidée avec l'ingénieur)

**Pourquoi.** Les modules ont été faits un à un, sans modèle commun. L'ingénieur a validé, sur la poutre puis sur les
dalles et les prédalles, une organisation qu'il veut retrouver partout. Suivre la spécification ci-dessus jusqu'au lot V
avec l'écran de B.3, puis refaire chaque module à cette organisation, ferait chaque module deux fois — et ses tests
avec. **Décision de l'ingénieur (08/11/2026)** : la présentation se refait **module par module, dans les lots M** ; les
points suivants remplacent ce qui, plus haut, les contredit. Le reste de la spécification tient.

1. **L'écran de référence** remplace B.3 (« Dispositions », « La liste », « Le panneau ») pour tous les modules :
   l'établi compact de la poutre, décrit dans les sections datées en tête de ce document et rendu par
   `api_web/etabli_compact.py`.
   - À gauche, l'arborescence de la note : chaque élément « nom · dimensions », le nom de l'élément ouvert modifiable sur
     sa ligne, son menu ⋯ (renommer, copier, supprimer) ; sous l'élément ouvert, ses rubriques numérotées en accordéon
     (une ouverte par défaut, les secondaires repliées), les données condensées à deux champs par ligne
     (`commun.ligne2`), les explications dans les ⓘ (une bulle au doigt).
   - Au centre, le dessin sous son en-tête technique (`entete`) ; chaque pièce cliquable sous la clé de sa ligne de
     données (`data-sel` = `data-cible`).
   - À droite, les résultats : un bloc par section ou par élément (`bloc`), coloré par son état, deux lignes par
     vérification, « tout le calcul ».
   - Au téléphone, les onglets Données · Dessin · Résultat (D6 inchangée).
   - De B.3 restent : les seuils de densité et de cibles au doigt, la persistance, l'enregistrement (son tableau), le
     démarrage vu par l'utilisateur.

2. **B.2.1 revue — le regel des photos est GARDÉ.** « Jamais de `--geler` » devient : une photo se refait quand l'ÉCRAN
   change ; `tests/test_reference_refonte.py --geler` **refuse** quand le CALCUL change (état de la note, messages,
   avis, verdict, une valeur qui ne se voit plus : fonction `garde`), sauf `--accepter="raison"` — la raison s'écrit
   alors, avec les écarts, dans la photo elle-même (clé « regels »). Les trois regels faits avant la garde y sont écrits,
   leurs écarts recalculés par elle :
   - poutre `98085c7` : le moteur ajoute les dispositions constructives EC2 8.2(2) à la demande de l'ingénieur (aucune
     valeur d'avant ne disparaît) ; une note sans effort dit « Données incomplètes » au lieu de « Conforme » ; le
     sélecteur de barres de la note neuve ne propose plus les dispositions de 1 885 à 4 825 mm² ;
   - poutre `50c37dd` : les dispositions passent de la fenêtre de calcul à la liste de contrôle du résultat ;
     l'espacement libre des armatures latérales (235 mm) ne s'affiche plus ;
   - dalle et prédalle `b3ec83a` : une note sans effort dit « Données incomplètes » au lieu de « Conforme ».
   Les 15 autres modules sont tels que le lot 0 les a gelés. La garde est figée par `tests/test_reference_garde.py`
   (chacun de ses contrôles vu rouge sans elle).

3. **S1 réduit.** Les briques communes sont celles de `api_web/etabli_compact.py` (arbre, menu, rubriques, en-tête,
   bloc, carte de section), de `api_web/plaque_compact.py` (dalles, prédalles) et `commun.ligne2` ;
   `section(…, secondaire=True)` reste (D5). Sont abandonnés, remplacés par ces briques : `liste_elements`, `objet`,
   `repli`, `cible`, `zone`, et les contrôles K1 à K6 qui les testaient. Le contrat de sélection devient : « chaque pièce
   du dessin a sa ligne dans les données, sous la même clé », testé module par module (comme
   `tests/test_api_web_dalle.py`). `marquer_modifies` (« Options · n modifiée(s) ») attend une demande de l'ingénieur.

4. **D7 revue.** Aucun champ d'ÉCRAN dans l'état de la note (ni identifiant, ni état d'interface). Un champ de
   DONNÉES demandé par l'ingénieur est permis — la poutre : lits multiples, cadres, justifications des observations —,
   avec la lecture des anciens `.stab` prouvée par un test, et l'écart accepté par la garde du regel.

5. **Un lot M, module par module** :
   1. le module passe sur l'établi compact (`construire.ETABLI_COMPACT`), données condensées — la carte de l'accueil
      ouvre alors le calcul directement : S2 et S3 se font ainsi, module par module ;
   2. l'ingénieur le regarde sur le site et donne ses remarques ;
   3. elles sont corrigées ;
   4. il est fermé quand : les cas de référence passent (un regel seulement par la garde), ses tests et le harnais aux
      trois formats (et au doigt) sont verts, ses captures sont regardées.

6. **Ordre.** M5 (les 8 modules simples) → M4 (plancher, sol, taux) → M2 (assemblages et kit des schémas) → M3
   (garde-corps, soudures, cornière, qui dépendent du kit) ; un module que l'ingénieur utilise davantage peut passer
   devant. S4 (démarrage) quand il tombe bien : la présentation ne le défait pas. C (chantiers et dossiers) attend
   l'accord de l'ingénieur : c'est une migration de format. V à la fin.
