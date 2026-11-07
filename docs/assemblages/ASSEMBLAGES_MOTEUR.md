# Assemblages métalliques sur le site « Stab Méthodes » — moteur, écran, parité

> **Défauts NON CONSERVATIFS de l'écran Streamlit** (reproduits par le moteur, que le site importe tel quel ; le site
> les REFUSE, à la saisie comme à la relecture d'une note — §5 et §9) :
>
> 1. **γ nul accepté** (poutre–poutre et poutre–poteau) : γ<sub>M2</sub> = 0 rend boulons, pressions diamétrales et
>    cordons infinis ; le plat d'âme par défaut est déclaré « ASSEMBLAGE VÉRIFIÉ » à 60 % au lieu de 74 %.
> 2. **V<sub>Ed</sub> vide pris égal à 0 sans message** (toutes les attaches sur poutre et sur poteau) : « vérifié » à
>    60 % au lieu de 74 % (plat d'âme par défaut). Pied de poteau : un effort vide vaut 0 (N<sub>Ed</sub> : 79 % →
>    138 %, ici défavorable, mais faux).
> 3. **Rangées non entières arrondies en silence** (n₁ = 2,4 → 2 rangées) et **dimensions négatives acceptées** :
>    taux faux dans les deux sens.
> 4. **Aucune borne sur les nombres de rangées ou de tiges** : n₁ = 10⁹ épuise la mémoire (14 Go mesurés, processus
>    tué) — l'écran Streamlit s'arrête ; le site borne à 50 rangées et 20 × 20 tiges (§9).
>
> Une note .stab (ou un état du navigateur) qui porte l'une de ces valeurs n'est PAS calculée en silence par le site :
> le résultat dit « À corriger » et nomme le champ (§9). Avant cette relecture, le site la calculait comme Streamlit
> (V<sub>Ed</sub> vide relu → « Conforme » à 60 %).

Le module `assemblages` du site (`api_web/assemblages.py`) porte les sept attaches de l'écran Streamlit
« Assemblages métalliques » (`modules/assemblages.py`) dans le navigateur. Une **note** du site est le **projet**
multi-assemblages de Streamlit (`acier/projet.py`) : chaque élément de la note est un assemblage, de l'une des sept
configurations du registre (`acier/registre.py`).

| Famille | Configurations | Fichier d'aide du site |
|---|---|---|
| Poutre–poutre | plat d'âme soudé, doubles cornières, platine d'about (partielle) | `api_web/assemblages_poutre.py` |
| Poutre–poteau | les trois mêmes attaches, sur l'aile ou dans l'âme du poteau | `api_web/assemblages_poteau.py` |
| Pied de poteau | platine sur béton (vérification complète EN 1993-1-8 + EN 1992-4, et « platine à la main ») | `api_web/assemblages_pied.py` |

## 1. Le calcul : aucun moteur extrait, aucun recalcul

Les moteurs existent déjà SANS Streamlit (`acier/assemblages/<famille>/<config>/` : `entrees`, `moteur`, `formules`,
`synthese`, `notation`, `texte`, `methode`, `benchmark`, `schemas` ; `acier/assemblages/mecanique.py`). Le site les
importe tels quels — jamais `interface.py`, `ecran_*.py`, `note.py`, `rapport.py`, `carte.py`, `ui_commun.py`, qui
chargent streamlit ou reportlab (le relevé du paquet, `site/web/construire.py`, l'interdit ; le test 11 le rejoue).

Le chemin d'un jeu de saisies est **celui de l'écran Streamlit**, pas à pas :

| Étape | Écran Streamlit | Site |
|---|---|---|
| lecture d'un fichier | `entrees.charger_json` | `assemblages.valeurs_normales` → `entrees.charger_json` |
| forme des widgets | `carte.valeur_widget` (nombre flottant ou vide, option valide, texte) | `assemblages.valeur_widget` — **copie nommée**, comparée à l'original champ par champ (test 2) |
| dimensions vides d'une pièce « Personnalisé » | `carte.dimensions_vides` + `carte.provisoire` : vérification suspendue, dessin provisoire | `dimensions_vides` + `a_calculer`, valeurs provisoires = `ecran_saisie.DIMENSIONS` (copie nommée, test 2) |
| calcul | `moteur.compute` (pied : `calcul.compute`) | le même |

Le site ne calcule rien : verdict, taux, lignes par pièce (`synthese.taux_par_element`), E<sub>d</sub>/R<sub>d</sub>
(`synthese.ed_rd`), formules substituées (`formules.textes` / `formules.lignes`), alertes (`notation.ligne_alerte`),
dessins (`schemas`), note texte (`texte.construire_texte`), méthode (`methode.METHODE`), validation
(`benchmark.run_bench`) viennent du moteur ; le site les MET EN FORME avec les briques de `api_web/commun.py`.

Les textes que l'écran Streamlit écrit dans ses fichiers d'écran (qu'on ne peut pas importer) sont des **copies
nommées**, liées à l'original par un test qui fait tourner l'écran avec un faux `st` :
`NON_REVERIFIES_PA/DC/PB` (= `ecran_resultats.NON_REVERIFIES`), les « paramètres retenus »
(= `ecran_resultats._parametres_retenus`), le « détail de la solution retenue par défaut » et la règle de choix de
l'onglet Prédim (= `ecran_resultats.onglet_predim`), les dimensions provisoires (= `ecran_saisie.DIMENSIONS`).

## 2. L'architecture

- `api_web/assemblages.py` : l'**aiguillage** (MODULE, PAGE, état de la note, liste des assemblages, menu
  Configuration, saisie, sélecteurs) et la **carte commune** des configurations (cadre Données en sections, verdict et
  lignes de résultat, fenêtres de calcul, vues de la toile). Une configuration y est DÉCRITE par un objet `Config`
  (son moteur, ses sections, ses libellés courts, sa pièce par groupe, ses formules, ses copies nommées, ses dessins,
  son prédimensionnement) ; rien de propre à une configuration n'est écrit dans ce fichier.
- `api_web/assemblages_poutre.py`, `_poteau.py`, `_pied.py` : une famille chacun, **sans MODULE** (le registre du site
  ne voit qu'un module). Ils sont chargés à la fin de `assemblages.py` et s'enregistrent (`A.enregistrer(Config(...))`).
- `site/web/css/modules/assemblages.css` : les vues « document » (prédim, note, méthode, validation), les tableaux du
  module (en-têtes sans capitales forcées : « η » ne devient pas « Η »), le menu Configuration pleine largeur.

## 3. L'écran (famille poutre–poutre, construite et vérifiée en entier)

**Cadre Données** — sections numérotées et repliables, dans l'ordre du travail :
1. *Assemblage* : Configuration (menu pleine largeur, les sept attaches), Repère, Mode (Vérification / Prédim.) ; en
   prédimensionnement, le taux cible, Ø min et max, les règles e₁ = k·d₀, p₁ = k·d₀, e₂ = k·d₀, et les champs pilotés
   par la proposition verrouillés (◆ de Streamlit).
2. *Poutres* : principale (profilé, nuance, dimensions si « Personnalisé »), secondaire, position (Δz, g<sub>h</sub>).
3. *Grugeage* : d<sub>c,sup</sub>, d<sub>c,inf</sub>, c, rayon, maintien au déversement.
4. *Plat d'âme* / *Cornières* / *Platine d'about* : la pièce, sa géométrie, sa soudure.
5. *Boulons* : boulon, classe, trou, catégorie, filetage, trou imposé (case + d₀), glissement (μ, k<sub>s</sub>,
   ELS/ELU), puis le groupe (rangées, files, entraxes, pinces). Doubles cornières : un groupe par aile boulonnée, une
   section *Soudures* quand une aile est soudée.
6. *Efforts ELU* : V<sub>Ed</sub>, N<sub>Ed</sub>, M<sub>Ed</sub> (H<sub>Ed</sub> pour les cornières ; nature de
   N<sub>Ed</sub>, ELU ou tying, pour la platine).
7. *Coefficients et modèle* : γ<sub>M</sub>, β<sub>Lf</sub>, intempéries, excentricité g<sub>A</sub> et k<sub>rot</sub>
   (cornières), acier personnalisé.
8. *Cartouche et affichage* : rédacteur, visserie, projet, date, niveau de cotation des dessins (simple / édition /
   toutes), note texte complète, « Charger » un exemple publié, « Rétablir les valeurs par défaut ».

Chaque champ porte le libellé COURT du site ; son aide (« i ») est le libellé long et l'aide de l'écran Streamlit
(`entrees.GROUPES`, source unique). Les conditions d'affichage sont celles de l'écran (`entrees.champ_visible`).

**Résultat** — le verdict (η de l'assemblage, la vérification gouvernante, « E<sub>d</sub> ≤ R<sub>d</sub> », motif si
la géométrie ou les pinces ne sont pas conformes), une ligne par pièce (Boulons, Plat / Cornières / Platine, Poutre
secondaire, Poutre principale, Soudure) avec sa pièce (le boulon, le profilé, la cornière ouvrent leur sélecteur), la
ligne Géométrie (Tableau 3.3), la Proposition du prédimensionnement (ouvre les solutions), puis les alertes du moteur.

**Fenêtres de calcul** — une par pièce : hypothèses (aciers, boulon, γ), tableau des vérifications (formule, E<sub>d</sub>,
R<sub>d</sub>, taux, nature EC/COMP/INT, référence), puis le détail « symbole = formule = valeurs = résultat » de
chaque vérification et la ligne η colorée ; la répartition élastique par boulon dans la fenêtre Boulons ; la matrice
aile A · aile B dans la fenêtre Cornières ; la fenêtre Géométrie (Tableau 3.3 et paramètres retenus) ; la fenêtre
Prédimensionnement (solutions, détail de la solution retenue, épaisseur nécessaire par vérification).

**Toile** — Élévation, Plan, Vue de droite (les dessins à l'échelle des `schemas` du moteur, au niveau de cotation
choisi, cotes des alertes en rouge), Prédim (toutes les combinaisons géométriquement possibles), Note (texte),
Méthode (le modèle, le périmètre, les natures), Validation (les exemples publiés et les calculs manuels rejoués).

**Sélecteurs** — boulons (tous, taux recalculé par le moteur), profilés (par famille, triés par hauteur), cornières
(EN 10056-1, ailes puis épaisseur), solutions du prédimensionnement : retenu ✓, optimal ★ (le plus petit qui tient ;
pour les solutions : la proposée par défaut). Choisir une solution = « Appliquer » de Streamlit
(`moteur.apply_solution`, retour en vérification).

**La note** — « Dans cette note » : chaque assemblage (repère, configuration, taux) ; Ajouter (même configuration,
défauts, premier repère libre « A2… » — `projet.repere_libre`), Copier (= 📋 Dupliquer : « A1 (copie) », insérée après
l'original, ouverte), Supprimer (jamais le dernier). Un assemblage neuf a les valeurs de « Créer » de la page projet
(`entrees.defaults()` mis en forme des widgets — un pied de poteau neuf s'ouvre donc sur le calcul à la main, comme
dans Streamlit ; ce n'est PAS la lecture d'un fichier vide, que `charger_json` lit comme un fichier d'avant le calcul
à la main). Le résumé de la note est son assemblage le plus chargé (un assemblage non conforme sous 100 % passe
devant). Changer de configuration repart des valeurs par défaut et garde le repère, le cartouche, la visserie et les
saisies qui gardent leur SENS (`assemblages.GARDEES` : profilés, grugeage, boulon et classe, efforts et nature de
N<sub>Ed</sub>, coefficients, prédimensionnement ; dans la famille poutre–poteau, le cas d'appui et le jeu latéral) —
jamais la géométrie de la pièce (la largeur d'un plat d'âme, 100 mm, donnerait à une platine d'about une pince e₂ de
5 mm). D'une famille à l'autre (poutre–poutre ↔ poutre–poteau), les PIÈCES ne suivent pas : ni la porteuse (une poutre
principale n'est pas un poteau), ni la poutre portée et son grugeage (`PORTEE`, `GRUGEAGE`) — la pièce neuve a la
géométrie par défaut de sa famille, dimensionnée pour la poutre par défaut de sa famille ; gardé, le HEA 300 grugé des
poutre–poutre sous le plat posé à z<sub>p</sub> = 40 mm des poutre–poteau donnait deux alertes « Impossible » à
l'ouverture (t<sub>f</sub> + r = 41 mm). Rien que le cartouche ne suit vers ou depuis le pied de poteau
(N<sub>Ed</sub> y est une compression positive, une traction ailleurs).

**Vérifications d'information** (pied de poteau : béton d'ancrage EN 1992-4 quand le périmètre est l'EN 1993-1-8
seule ; niveau 2, pression maximale ; au calcul à la main, tiges et appui) : comme dans Streamlit
(`synthese.info`), elles ne comptent ni dans le taux de leur pièce ni dans le verdict ; leur ligne est grise et dit
« information », leur ligne de fenêtre dit « information, hors verdict », et la phrase d'information de l'écran
Streamlit (`ecran_resultats.information`, `manuel/ecran.information` — copies nommées, test 2) suit le verdict.

## 4. Inventaire de l'écran Streamlit poutre–poutre et couverture

| Écran Streamlit | Site |
|---|---|
| 67 / 84 / 64 clés d'entrée (plat d'âme / cornières / platine), carte par objet + replis « Paramètres avancés » et « Cartouche » | toutes, dans les sections 1 à 8 (la géométrie, qui ne se saisit que sur le dessin dans Streamlit, est dans les sections) |
| aides « ? » (`AIDES`) | aide « i » = libellé long + aide de `entrees.GROUPES` |
| mode Vérification / Prédimensionnement, champs ◆ pilotés | segment Mode, champs verrouillés, aide |
| dessin interactif : clic sur une pièce → ses paramètres, glisser la poutre (g<sub>h</sub>, Δz) | **non** : dessins fixes (besoin commun « dessin cliquable ») ; tous les paramètres sont dans les sections |
| niveau de cotation (vue simple / édition / complète) | « Cotes » (section 8) : Simple · Édition · Toutes |
| bandeau : statut, η max, gouvernante, géométrie / pinces non conformes, taux par élément | verdict + une ligne par pièce |
| alertes, alerte sélectionnée → cotes en rouge, champs en cause modifiables sur place, « Localiser » | alertes listées ; TOUTES les cotes en cause en rouge sur les dessins ; les champs se modifient dans les sections |
| Tableau 3.3 : valeur, min, max, ✓ / ✗ | fenêtre Géométrie : valeur, min, max, verdict du moteur ; le « taux » d'une pince (limite / valeur) est une PROJECTION pour la jauge et la couleur, que Streamlit ne montre pas |
| onglet Vérifications : tables par élément (E<sub>d</sub>, R<sub>d</sub>, η, réf., nature), matrice aile A/B, légende, répartition par boulon, périmètre, formules, Tableau 3.3, paramètres retenus | fenêtres de calcul (une par pièce + Géométrie), vue Méthode (périmètre, natures, références) |
| onglet Prédim : 9 solutions « Appliquer », détail de la retenue, épaisseurs, toutes les combinaisons « Appliquer » | sélecteur des solutions (toutes les combinaisons ; un clic = Appliquer), vue Prédim, fenêtre Prédimensionnement |
| onglet Note : texte essentiel / complet, export .txt ; note PDF 2 pages ; rapport détaillé 3 pages (cornières) | texte essentiel / complet (à copier) ; **pas** de fichier .txt (besoin commun « télécharger un texte ») ; PDF hors périmètre |
| onglet Benchmark : statut, source, tables, « Charger ces données », sans exemple, calculs manuels | vue Validation ; « Charger » dans la section Cartouche |
| onglet Méthode | vue Méthode |
| barre : 🔄 réinitialiser, 💾 / 📂 (projet ou assemblage seul), 📄 PDF | « Rétablir les valeurs par défaut » ; .stab / Ouvrir du site (la note entière) ; PDF hors périmètre |
| page projet : liste (repère, type, pastille, η, gouvernante, description), Nouvel assemblage (catalogue), ⋯ : repère, 📋 Dupliquer, 🧬 Variante, ↑ ↓ ordre du rapport, 🔒 verrou, 🗑️ supprimer avec confirmation ; infos du projet ; rapport PDF du projet | liste « Dans cette note », Ajouter + menu Configuration, Copier (= Dupliquer = Variante, ouverte), Supprimer ; **pas** d'ordre ↑ ↓ ni de verrou (besoins communs) ; infos du projet = en-tête de la note du site ; rapport PDF hors périmètre |

Différences voulues (meilleures que Streamlit, écrites ici) :
- un profilé (ou une cornière, un acier) passé en « Personnalisé » part des dimensions de la pièce quittée — Streamlit
  les laisse vides et suspend la vérification (le site sait aussi suspendre : « À compléter », test 3) ;
- les saisies que Streamlit ramène EN SILENCE à une valeur fausse sont **refusées** avec un message (section 6).

## 5. Défauts trouvés dans l'écran Streamlit (reproduits par le calcul, refusés à la saisie)

1. **Non conservatif** : un coefficient partiel nul est accepté (`number_input` sans minimum). γ<sub>M2</sub> = 0 rend
   les résistances des boulons, des pressions diamétrales et des cordons infinies : leurs taux tombent à 0 % et le
   plat d'âme par défaut est déclaré **« ASSEMBLAGE VÉRIFIÉ » à 60 %** (au lieu de 74 %) ; la platine d'about reste
   « vérifiée » avec ses boulons à 0 %. γ<sub>M0</sub> = 0 fait échouer le calcul (OverflowError, « Données
   incomplètes »). Le site refuse un γ ≤ 0.
2. **Non conservatif** : V<sub>Ed</sub> vide est pris égal à 0 sans message (plat d'âme : « vérifié » à 60 % au lieu de
   74 %, la gorge minimale gouverne). Le site exige une valeur.
3. Un nombre de rangées non entier est arrondi en silence (n₁ = 2,4 → 2 rangées : 135 % au lieu du taux de 3
   rangées) ; n₁ = 0 devient 1. Le site exige un entier ≥ 1.
4. Une dimension négative est acceptée : η = 10⁹ % (t<sub>p</sub> = −10, a = −5, p₁ = −60) ou « Données incomplètes :
   ValueError: math domain error » (h<sub>p</sub> < 0). Le site refuse une dimension négative.
5. En mode prédimensionnement, l'élévation et la vue de droite cotent la gorge SAISIE (« a 5 ») et non la gorge
   PROPOSÉE que la vérification emploie (a = 4 mm, ligne Soudure). Défaut des `schemas` du moteur, reproduit (le site
   montre les dessins du moteur) ; la ligne Soudure du résultat dit la bonne gorge.
6. La liste du projet (`projet_ui.resume`) calcule sans les dimensions provisoires : un assemblage dont le profilé
   « Personnalisé » n'a pas de dimensions y est « non vérifié · η = 100 000 000 000 % » quand son écran dit « Données
   incomplètes ». La liste du site dit « à compléter ».
7. Un taux estimé du prédimensionnement peut être indéterminé (NaN, combinaison dégénérée) : Streamlit l'écrit « — » ;
   le site le montre rouge (999 %, « indéterminé »), et ne s'arrête plus (le site levait ValueError sur ce cas avant
   le test 5).

## 6. La preuve : `tests/test_api_web_assemblages.py`

1. Registre : MODULE, PAGE, les sept configurations dans l'ordre du registre, chacune calcule.
2. Copies nommées : `valeur_widget` = `carte.valeur_widget` (7 configurations × chaque champ × 20 saisies),
   assemblage neuf = `projet_ui.creer`, dimensions provisoires, `NON_REVERIFIES`, paramètres retenus et détail du
   prédimensionnement (≈ 95 résultats, `_parametres_retenus` et `onglet_predim` rejoués avec un faux `st`), phrase
   d'information du pied de poteau (12 résultats).
3. Parité fichier → moteur, **392 jeux** (défauts, exemples publiés, `tests/ref_*.json`, tirages au hasard :
   profilés du catalogue ou personnalisés, dimensions vides, textes « 12,5 », options inconnues, V vide,
   prédimensionnement) : chemin de l'écran Streamlit et chemin du site donnent la même empreinte **au bit près**
   (repr des flottants : chaque vérification E<sub>d</sub>, R<sub>d</sub>, η, active, vérifiée ; alertes ; pinces ;
   statut). Les jeux couvrent vérifié, non vérifié, à compléter, erreur et prédimensionnement.
4. L'écran Streamlit LUI-MÊME (AppTest, application réelle) : 12 jeux par attache poutre–poutre chargés dans son état
   comme le fait « Ouvrir » ; son résultat (`_fpl_R`, `_asm_R`, `_pab_R`) a l'empreinte de celui du site ; une saisie
   au clavier (V<sub>Ed</sub> = 210) aussi.
5. Ce que le site affiche = le moteur : verdict, lignes par pièce, ligne gouvernante, E<sub>d</sub>/R<sub>d</sub> de
   chaque vérification dans sa fenêtre, alertes, dessins (SVG identiques au caractère près), note texte, méthode,
   validation, prédim ; chaque vérification a son détail et sa ligne η ; matrice des cornières ; répartition par boulon.
6. Saisie : valeurs acceptées comme le widget ; 12 refus (dont les défauts 1 à 4) avec l'assemblage intact ; le défaut
   lui-même contrôlé dans le moteur ; « Personnalisé », trou imposé, exemple publié, réinitialisation, menu
   Configuration.
7. La note : ajouter, copier, supprimer, ouvrir, résumé, notes abîmées.
8. Sélecteurs : boulons (taux recalculés, ✓, ★), profilés (tri), solutions (= `apply_solution`), cornières.
9. Échappement.
10. Mutations : abîmer `valeur_widget` (virgule, option inconnue), les dimensions provisoires, un refus, ou V<sub>Ed</sub>
    de 10⁻⁷ fait rougir le comparateur.
11. Pureté : le module se charge et calcule les sept configurations dans un sous-processus où streamlit, pandas,
    reportlab, matplotlib, scipy, PIL, fitz sont refusés ; rien d'autre que la bibliothèque standard et numpy.

Résultats du 07/11/2026 : `tests/test_api_web_assemblages.py` 142 contrôles, 0 échec ; contrat commun 17 / 17 ;
contrôle navigateur (`site/web/verifier_site.js … module:assemblages`) 19 / 19 ; parcours des sept configurations
dans Chromium (chaque section ouverte, chaque vue, chaque fenêtre de calcul, chaque sélecteur, le prédimensionnement ;
PC 1440 × 900 pour les sept, tablette 1024 × 1366 pour les six attaches poutre–poutre et poutre–poteau) : aucun nom
coupé, aucun débordement, aucune erreur JavaScript.

```bash
python3.12 tests/test_api_web_modules.py assemblages     # le contrat commun
python3.12 tests/test_api_web_assemblages.py             # ce fichier (≈ 40 s ; TIRAGES=…, CAS_ECRAN=… pour en faire plus)
python3.12 tests/test_api_web_assemblages_poteau.py      # la famille poutre–poteau (§7, ≈ 15 s)
python3.12 tests/test_api_web_assemblages_pied.py        # le pied de poteau (§8, ≈ 5 min ; CONTRAT_SEPT=1 : le contrat sur les sept)
```

## 7. La famille poutre–poteau (construite et vérifiée en entier)

`api_web/assemblages_poteau.py` décrit les trois attaches sur poteau — plat d'âme soudé (`plat_ame_poteau`), doubles
cornières (`doubles_cornieres_poteau`), platine d'about partielle (`platine_about_poteau`) — avec leurs moteurs
(`acier/assemblages/poutre_poteau/<config>/`), au même niveau que la famille poutre–poutre.

**Cadre Données** — 9 sections (10 pour les cornières en partie soudées) :
1. *Assemblage* : Configuration, Repère, Mode, **Appui** (segments « Sur l'aile / Dans l'âme » : le cas d'appui
   commande toute la mécanique ; « Aile du poteau / Âme du poteau » ne tenait pas dans son segment au format tablette —
   mesuré, la colonne débordait) ; cornières : fixation des ailes A (poteau) et B (poutre portée) ; prédimensionnement.
2. *Poteau porteur* : profilé, nuance, dimensions si « Personnalisé ».
3. *Poutre portée* : profilé, nuance, dimensions ; *Position* : g<sub>h</sub> (plat, cornières), et dans l'âme la
   recoupe des semelles r<sub>c</sub> et le jeu latéral j (`entrees.champ_visible`).
4. *Grugeage* (replié, « non grugée » : un poteau n'oblige pas à gruger).
5. *Plat d'âme* / *Cornières* / *Platine d'about* et sa soudure.
6. *Boulons* : boulon, classe, trou, catégorie, trou imposé, glissement, puis le groupe (plat : rangées, files,
   entraxes, pinces ; cornières : groupe P sur le poteau, groupe S sur l'âme portée ; platine : 2 files, p<sub>3</sub>,
   avec la règle de dégagement de l'écrou). Cornières soudées : section *Soudures*.
7. *Efforts ELU* (et la nature de N<sub>Ed</sub>, ELU ou tying, dès qu'il y a traction). 8. *Coefficients et modèle*.
9. *Cartouche et affichage*.

Chaque champ que l'écran Streamlit montre est dans le cadre, et seulement lui — contrôlé sur l'aile, l'âme, le
prédimensionnement, le tying et les pièces personnalisées.

**Résultat** — verdict, une ligne par pièce (Boulons, la pièce, Poutre portée, **Poteau porteur** avec son profilé
et le cas d'appui — « HEB 300 · âme » —, Soudure), Géométrie, Proposition. Plat d'âme sur l'aile : pas de ligne
Poteau porteur (aucune vérification locale de l'aile n'est demandée par les guides) ; la vue Méthode dit pourquoi.

**Fenêtres de calcul** — une par pièce ; les hypothèses nomment les aciers « poteau » et « poutre portée »
(`Config.noms_aciers`) ; répartition élastique par boulon (plat, cornières) ; matrice aile A / aile B des cornières
avec la légende de l'écran poteau (« Aile A sur le poteau ») ; Géométrie : Tableau 3.3 avec la règle de l'écran de
chaque attache (`Config.regle_33` : « p₂ et p₃ ≥ 2,4·d₀ » pour les cornières, « p₃ (p₂ du tableau) » pour la platine)
et les paramètres retenus (cas d'appui, M<sub>col</sub> apporté au poteau en information, largeurs libres, recoupe
minimale, variante b0/b1/b2, tronçons en T de la platine et de l'aile du poteau…).

**Toile** — Élévation et Vue en plan (les deux dessins de l'écran Streamlit), plus la **Coupe A-A** (celle de la
note de calcul), aux trois niveaux de cotation, cotes des alertes en rouge ; Prédim (colonnes de l'écran : variante,
h<sub>p</sub>, η boulons, η p. diam. âme ; cornières : h<sub>c</sub>, cornière, η S et P ; platine : η boulons, η âme
portée, η poteau = max(pression diamétrale ; cisaillement local) dans l'âme — `colonnes` peut dépendre du résultat) ;
Note ; Méthode (limites de `methode.limites(ame)`, la source unique de l'écran, et le « sans objet » de l'aile,
`Config.sans_objet`) ; Validation (décimales de l'onglet Benchmark de chaque attache, `Config.decimales_bench` ; les
nombres arrondis comme `acier.formats.F`, `fr_js` : 31,25 s'écrit « 31,3 » comme dans Streamlit, pas « 31,2 »).

**Sélecteurs** — « Poteau · profilé » (HEA, HEB, HEM… par famille, triés par hauteur, taux recalculé), « Poutre
portée · profilé », boulon, cornière, solutions du prédimensionnement (= `apply_solution`, le cas d'appui est gardé).

**Copies nommées** (test : `tests/test_api_web_assemblages_poteau.py`, faux `st` sur 48 à 62 résultats par attache,
aile et âme) : `PERIMETRE_AILE_PA/DC`, `_parametres_pa/dc/pb`, `detail_predim_pa/dc/pb`, `REGLE_PA/DC/PB`,
`LEGENDE_AILES` ; le repli Périmètre de l'écran (non revérifiés et sans objet) est comparé à la vue Méthode, la
légende du Tableau 3.3 à la fenêtre Géométrie, les tableaux de répartition par boulon à ceux de l'onglet
Vérifications, chaque cellule des colonnes du prédimensionnement à celle de l'écran, chaque référence et valeur du
Benchmark à la vue Validation.

**Ce que l'aiguillage a reçu pour cette famille** (crochets sans effet sur les autres configurations) : `noms_aciers`,
`sans_objet`, `regle_33`, `decimales_bench`, `colonnes` en fonction du résultat ; `GARDEES` garde le cas d'appui, le
jeu latéral et la nature de N<sub>Ed</sub> ; d'une famille à l'autre les pièces ne suivent pas (§3) ; les mots
français des formules (« aucun bord libre », « paroi », « pince »…) ne sont plus pris pour des symboles indicés
(`_STOP`) — défaut du site, visible dans les fenêtres de toutes les familles.

**Défauts de l'écran Streamlit, propres à cette famille ou retrouvés sur elle** (reproduits par le calcul) :
- les défauts 1 à 4 du §5 (γ = 0, V<sub>Ed</sub> vide, rangées non entières, dimensions négatives) sont dans les trois
  moteurs poutre–poteau (contrôlé : n₁ = 2,4 → 2 rangées, γ<sub>M2</sub> = 0 → boulons à 0 %, V vide → 0) ; le site
  les refuse à la saisie, recoupe et jeu latéral négatifs compris ;
- un acier « Personnalisé » sans f<sub>y</sub> (fichier, ou champ vidé) vaut f<sub>y</sub> = 0 dans le moteur :
  « non vérifié » à η = 1 000 000 000 % au lieu de « données incomplètes » (`carte.dimensions_vides` ne surveille que
  les dimensions). Reproduit sur le site (même verdict ; les hypothèses de la fenêtre ne nomment pas un acier à
  f<sub>y</sub> = 0). Non conservatif ? Non : le verdict est « non vérifié » — mais le message trompe sur la cause.

**Preuve** — `tests/test_api_web_assemblages_poteau.py` (≈ 15 s, 138 contrôles) : les trois configurations et leur
cadre Données ; les copies nommées ; la parité fichier → moteur au bit près (218 jeux : défauts aile et âme, tying,
prédimensionnement, cornières soudées, exemples publiés, `tests/ref_fpc.json`, `ref_dcc.json`, `ref_pac.json`,
tirages) ; l'écran Streamlit LUI-MÊME (AppTest, `fpc_`, `dcc_`, `pac_` : 10 jeux chacun dont 7 dans l'âme, et la
bascule « Âme du poteau » au clavier) ; ce que le site affiche = le moteur (verdict, lignes, gouvernante, cas d'appui,
E<sub>d</sub>/R<sub>d</sub> et ligne η de chaque vérification dans sa fenêtre, alertes, trois dessins au caractère
près, note, limites, hypothèses) ; la saisie et ses refus ; le changement de configuration ; les sélecteurs ;
l'échappement ; cinq mutations qui rougissent (une ligne de paramètre retirée, une référence abîmée, V × 1,0000001,
le « sans objet » de l'aile oublié, η poteau sans le cisaillement local).

Contrôle navigateur (07/11/2026, copie `/tmp/site_assemblages`, port 8801) : `verifier_site.js module:assemblages`
19 / 19 ; parcours des trois attaches sur PC 1440 × 900 et tablette 1024 × 1366 — chaque section ouverte, chaque vue,
chaque fenêtre, chaque sélecteur, le prédimensionnement, puis le cas « dans l'âme » (sections, plan, coupe A-A,
fenêtres Poteau porteur et Géométrie) : rien de coupé, aucun débordement, aucune erreur JavaScript.

**Non couvert** (comme pour la famille poutre–poutre) : le dessin interactif (cliquer une pièce, glisser la poutre),
le panneau « Cotes », l'export .txt, la note PDF.

## 8. Le pied de poteau (construit et vérifié en entier)

`api_web/assemblages_pied.py` décrit la platine de pied de poteau sur béton (`platine_beton`), avec ses DEUX moteurs —
le calcul « platine à la main » (`manuel/moteur`, l'écran neuf de Streamlit) et la vérification complète EN 1993-1-8
+ EN 1992-4 (`moteur`) — que `calcul.compute` choisit selon le mode. Chaque mode a sa synthèse, son texte, sa méthode,
ses dessins, son périmètre et sa validation : la configuration porte une VARIANTE (`MANUEL`), que `assemblages.pour(cfg,
R)` retient pour un résultat du calcul manuel.

**Cadre Données — calcul manuel** (9 sections) : 1. *Assemblage* : Configuration (« Pied de poteau · Platine sur béton »),
Repère, Mode (segments « Manuel / Complet » — « Calcul manuel / Vérif. complète » de l'écran ne tenait pas à 1024 px),
l'état de la bascule (« Vérification complète : disponible / indisponible — raison », `manuel/ecran.etat_bascule`),
Critère (Élastique / Plastique), Appui (Bande c / Toute la platine). 2. *Platine* : b<sub>p</sub>, l<sub>p</sub>,
t<sub>p</sub>, nuance, gorge a (« ne sert qu'au bras de levier »). 3. *Boulons* : boulon, classe, ancrage, h<sub>ef</sub>,
d<sub>h</sub>, la table des boulons (n°, x, y, retirer), « Ajouter un boulon » (au bord bas, à droite du dernier, comme
`_ajouter_boulon`), « Poser une grille » (files, rangées, entraxes ; remplace les boulons, comme `_poser_grille`).
4. *Éléments soudés* : une carte par élément (type aux noms courts de l'écran, profilé, x, y, rotation 0/90°, dimensions
d'une section sans catalogue), ajouter, retirer. 5. *Raidisseurs* : une carte par raidisseur (x, y, sens, L, t, h).
6. *Béton et massif* (béton, massif, mortier). 7. *Efforts ELU* et le point d'application (x<sub>N</sub>, y<sub>N</sub>).
8. *Coefficients* (et acier personnalisé). 9. *Cartouche et affichage* (visserie par défaut de l'écran du pied : « 1
rondelle + 1 écrou + 1 contre-écrou »).

**Cadre Données — vérification complète** (8 sections) : Assemblage (Mode, Périmètre « EC3 seul / EC3 + EC2-4 »),
Poteau (forme I / tube, profilé, dimensions, nuance, rigidité L<sub>c</sub> et contreventement), Platine, Tiges
d'ancrage (grille n<sub>x</sub> × n<sub>y</sub>, w<sub>x</sub>, w<sub>y</sub>, tige, classe, ancrage, h<sub>ef</sub>,
d<sub>h</sub>, valeurs de l'ETA), Béton et massif (fissuré), Efforts, Coefficients (γ<sub>Mc</sub>), Cartouche. Dans les
deux modes, chaque champ que l'écran Streamlit montre (`entrees.champ_visible`) est dans le cadre, et seulement lui ; la
géométrie qui ne se saisit que sur le dessin dans Streamlit (platine, massif, mortier, grille, ancrage, gorge) est dans
les sections. La clé `ancr` s'appelle « Boulon » au calcul manuel, « Tige » en vérification complète (`LIBELLE_MODE`).

**La bascule de mode** = `ecran_saisie.changer_mode` : la disposition est reprise (`manuel/conversion` : la grille et le
profilé deviennent des boulons et un élément ; vers la vérification complète seulement si la disposition est régulière
— un seul profilé en I ou RHS à l'axe, non tourné, sans raidisseur, boulons en grille centrée). Sinon la saisie est
refusée avec la raison de l'écran (« Vérification complète impossible : un seul élément soudé est admis. ») et la note
reste intacte.

**Résultat** : verdict (la vérification gouvernante, E<sub>d</sub> ≤ R<sub>d</sub>, le critère qui gouverne
l'épaisseur — le bandeau de Streamlit), une ligne par pièce (`synthese.ELEMENTS` du mode), les lignes d'information en
gris (tiges et appui au calcul manuel ; béton d'ancrage hors périmètre en vérification complète), puis, SOUS la ligne
Platine, les quatre critères de l'épaisseur requise (t / t<sub>p</sub> : tronçon en T du boulon, section au nu, bande
comprimée, pression diamétrale ; ou console, appui, flexion composée, tronçon en T) — les pastilles du bandeau de
l'écran ; chacun ouvre la fenêtre Platine. La ligne Tiges ouvre le sélecteur des diamètres, la ligne Poteau celui des
profilés. La phrase d'information de l'écran suit le verdict.

**Fenêtres de calcul** : Platine (EN TÊTE le calcul à la main : l'épaisseur requise et ses quatre critères, puis côté
comprimé / côté tendu et la comparaison de Navier — `bloc_platine` — ou, au calcul manuel, boulon par boulon, face par
face, la bande comprimée et les trous — `detail_main`), Tiges (l'effort de chaque tige ; en complet, les rangées en
tronçon en T), Pied (le domaine résistant par direction, niveaux 1 et 2), Ancrages (EN 1992-4), Béton, Poteau, Soudure,
Trous, Appui, et Géométrie (Tableau 3.3 avec la légende de l'écran du pied, paramètres retenus du mode).

**Toile** : Plan, Coupe A-A, Coupe B-B (les dessins des `schemas` du mode, à l'échelle), Domaine (vérification
complète), Efforts (calcul manuel : `onglet_efforts`) ou Répartition (complet : `onglet_repartition`), Note, Méthode
(périmètre « non revérifiés » / « non vérifiés ici », natures EC · EC + INT · INT avec le sens de l'écran), Validation
(complet : VAL-A, VAL-B, fiches BM1 et BM2 à compléter, vérifications sans exemple, validation croisée du 31/10/2026 ;
manuel : VAL-M1 à VAL-M4). Les six cas se rechargent depuis la section Cartouche (« Exemples et cas de validation »,
chacun dans son mode, comme `_charger` des deux onglets Benchmark).

**Ce que l'aiguillage a reçu pour cette famille** (crochets de `Config`, sans effet sur les autres configurations) :
`visserie_defaut`, `exemples`, `etat` et `actions` (la grille à poser, un état d'écran gardé dans la note), `sur_saisie`
(la bascule), `aides` (les « ? » de la carte Streamlit du mode), `hypotheses`, `normes`, `blocs_tete`, `lignes_resultat`
(sous la ligne d'un groupe), `complement_verdict`, `legende_33`, `natures`, `titre_non_verifies`, `validation`,
`nom_menu` ; le genre `("lib", clé, libellé)` des sections (un libellé propre au mode). Corrigés au passage, pour toutes
les configurations : un taux au-delà de 999 % — infini ou indéterminé compris (aucune épaisseur ne suffit) — s'affiche
« 999 % » rouge, comme le « > 999 % » de Streamlit (`acier.formats.pct`), au lieu d'arrêter l'écran (OverflowError) ou
d'écrire « 100 000 000 000 % » (`assemblages.taux`) ; la formule d'une vérification est coupée au premier « ; »
HORS parenthèses (« t<sub>p</sub> ≥ max(t<sub>T</sub> ; t<sub>S</sub> ; …) » était tronqué) ; « HEA 300 », « au nu »
ne sont plus pris pour des symboles indicés ; « Cotes : Simple · Édition · Toutes » (« Vue simple », « Complète » rognés à
1024 px) ; une valeur longue des paramètres retenus, un nom de vérification long, une formule sans espace se replient
au lieu d'élargir la fenêtre.

**Copies nommées** (`tests/test_api_web_assemblages_pied.py`, faux `st` sur 17 à 45 résultats) : `AFFICHAGE`,
`AFF_TYPES`, `DIMS_ELEM`, `AIDES_CO` / `AIDES_MA` (entrée par entrée), `NON_REVERIFIES`, `NON_VERIFIES`, sens des
natures, `CRITERES_NOMS`, `LIBELLE_MODE`, légendes du plan, `NOTE_GORGE_MA`, `LEGENDE_33`, `etat_bascule`,
`ajouter_boulon`, `poser_grille`, `platine_complet`, `platine_manuel`, `detail_main`, `parametres_complet`,
`parametres_manuel`, `repartition`, `efforts_manuel`, `note_information`, les cas de `exemples`, la bascule, chaque
cellule des deux onglets Benchmark. Les aides qui renvoient au dessin interactif (« hef et dh se règlent sur la coupe
A-A ») sont dites autrement sur le site (`AIDES_SITE`), qui les saisit dans le cadre Données.

**Défauts de l'écran Streamlit** (reproduits par le calcul, refusés à la saisie sur le site) :
- une dimension de raidisseur négative est ramenée à 0 EN SILENCE (`elements.normaliser_raid`) ; une dimension
  d'élément négative est gardée telle quelle (géométrie absurde) : le site refuse une dimension négative dans les tables ;
- N<sub>Ed</sub> (et tout effort) vide vaut 0 sans message (calcul manuel par défaut : η 79 % → 138 %, la compression
  aidait) ; un coefficient γ<sub>M0</sub> ou γ<sub>M2</sub> nul est remplacé EN SILENCE par sa valeur par défaut (1,00 ;
  1,25 : même résultat que le défaut, mesuré dans les deux modes) : le site exige une valeur, et un γ strictement positif
  (comme pour les autres familles) ;
- (défaut du SITE, corrigé) l'épaisseur requise infinie (tige M12, ou crosse, en vérification complète) arrêtait l'écran
  du site (OverflowError) : elle s'affiche « 999 % » rouge.

**Preuve** — `tests/test_api_web_assemblages_pied.py` (≈ 5 min, 82 contrôles ; `CONTRAT_SEPT=1` ajoute le contrat
commun rejoué avec une note neuve de chacune des sept configurations et du pied en vérification complète) : cadre
Données = `champ_visible` (9 cas) ; copies nommées ; parité fichier → moteur au bit près (49 jeux : défauts des deux
modes, fichier sans mode, les six cas, tirages — boulons libres et en grille, éléments de chaque type, raidisseurs,
rotations, critère, appui, point d'application, tube, ETA, périmètre, profilé vide) ; l'écran Streamlit LUI-MÊME
(AppTest `ppb_`, 16 jeux des deux modes) ; ce que le site affiche = le moteur (lignes, information grise, quatre
critères, E<sub>d</sub>/R<sub>d</sub> et détail dans chaque fenêtre, alertes, plan, coupes et domaine au caractère près,
note, méthode, vues) ; la saisie (tables, refus, grille, bascule acceptée et refusée, exemples, réinitialisation,
visserie) ; sélecteurs ; échappement ; quatre mutations qui rougissent.

Contrôle navigateur (07/11/2026, copie `/tmp/site_assemblages`, port 8801) : `verifier_site.js module:assemblages` au
vert ; parcours du pied (`scratchpad/asm_pied/parcours_pied.js`) sur PC 1440 × 900 et tablette 1024 × 1366 : calcul
manuel (sections, vues, fenêtres, sélecteur), ajouter / retirer un boulon, grille 3 × 2, raidisseur, second élément
(tube), bascule refusée puis acceptée, vérification complète (fenêtres, sélecteurs), périmètre complet, ETA, tube,
VAL-M2 rechargé — rien de coupé, aucun débordement, aucune erreur JavaScript ; au téléphone, le cadre Données (l'établi
n'y montre pas encore le résultat). Parcours des sept configurations (PC et tablette) : au vert.

**Non couvert** : le dessin interactif (glisser un boulon, un élément, un raidisseur ; fenêtres d'objet sur le dessin)
et le panneau « Cotes » — tout se saisit dans les sections ; « Localiser » une alerte (toutes les cotes en cause sont en
rouge) ; l'export .txt et la note PDF ; les figures de la validation croisée (le rapport est nommé).

## 9. Relecture indépendante du 07/11/2026

Une relecture (contrôle des chiffres affichés, couverture, écran, robustesse) a trouvé et corrigé :

1. **Non conservatif — une note RELUE n'était pas contrôlée** : la saisie refusait V<sub>Ed</sub> vide, γ ≤ 0,
   n₁ = 2,4 ou une épaisseur négative, mais une note .stab (ou l'état du navigateur) qui les portait était calculée
   comme Streamlit : V<sub>Ed</sub> vide relu → « Conforme » à 60 % (74 % avec l'effort). Désormais
   (`assemblages.saisies_refusees`, la règle même de la saisie, `_lire_saisie`) : la valeur reste dans son champ, le
   résultat dit « À corriger » avec un message par champ, aucun taux ; la liste de la note dit « à compléter ». Aucune
   valeur par défaut, d'exemple publié ni de cas de référence n'est refusée (contrôlé).
2. **Robustesse — rangées et tiges sans borne** : n₁ = 10⁹ (saisi, ou relu d'une note) épuisait la mémoire (le moteur
   pose un boulon par case : 14 Go, processus tué). Bornes `COMPTES_MAXI` (50 rangées, 20 × 20 tiges en vérification
   complète), grille à poser du calcul à la main 20 × 20, tables de 400 lignes au plus (`LIGNES_MAXI`). Hors bornes, le
   moteur n'est JAMAIS appelé (`assemblages.moteur`, `trop_grand` : résultat, sélecteurs, solutions) ; la saisie est
   refusée avec la borne, la note relue dit « Calcul impossible : … 50 au plus ».
3. **Affichage — la tolérance du moteur** : le moteur déclare vérifiée une vérification à η ≤ 1 + 10⁻⁹ (l'écran
   Streamlit écrit « 100,0 % · VÉRIFIÉ ») ; le site écrivait « 101 % » rouge, « Non conforme », à côté de « E<sub>d</sub> ≤
   R<sub>d</sub> ». De même une pince conforme pour le moteur (p₁ = 48,4 mm pour 2,2·d₀ = 48,400000000000006 mm) donnait
   une ligne Géométrie « 101 % » rouge. Un taux que le moteur déclare vérifié s'affiche désormais au plus 100 %
   (`eta_vu`, `taux_c`, `eta_R`) ; au-delà de sa tolérance, rien ne change.
4. **Couverture — onglet Benchmark** : la colonne de contrôle des calculs manuels des doubles cornières s'appelle
   « Excel » (un classeur recalculé), pas « Contrôle » ; la légende de chaque écran (`LEGENDE_MANUEL`,
   `LEGENDE_MANUEL_DC`, copies relues dans le source par le test), l'adresse du document source (écrite, jamais
   chargée) et « ces tableaux sont recalculés à chaque ouverture » sont repris.
5. **Formules** : « F,Ed » du moteur s'écrit F<sub>Ed</sub> ; un exposant n'empêche plus l'indice (« √(Fz² + Fx²) » →
   √(F<sub>z</sub>² + F<sub>x</sub>²)).
6. **Couverture — doubles cornières aux deux ailes soudées** : le site cachait la section Boulons, que l'écran
   Streamlit garde (`entrees.champ_visible`) et que le moteur lit encore : un trou « Surdimensionné » sans d₀ y bloque
   le verdict (alerte `d0_manq`, « NON VÉRIFIÉ ») sans qu'on puisse le corriger. La section reste, repliée, « sans
   objet (ailes soudées) », avec une aide. Le test compare désormais le cadre Données des trois attaches poutre–poutre
   aux champs de l'écran (11 variantes), comme celui de la famille poutre–poteau.
7. **Mise en forme** : l'en-tête « Valeur » des paramètres retenus est aligné sur ses valeurs ; le verdict ne met en
   minuscule que la première lettre du nom de la vérification (« ailes B – gorge minimale », et non « ailes b », « groupe
   s ») ; le pied des sélecteurs écrit « A<sub>s</sub> mm² » sans l'espace de la pastille (« A ₛ mm² »).

Chiffres : 39 cas (les sept configurations, courants, extrêmes — η > 999 %, ∞ —, pile à la limite de part et d'autre
de 100 % et à 95 %, prédimensionnement, personnalisé, à compléter, tying, âme du poteau, deux modes du pied) passés par
`api_web.routeur.traiter` : verdict, taux de chaque pièce, E<sub>d</sub> / R<sub>d</sub> / η de chaque vérification
dans sa fenêtre et note texte = ce qu'affiche l'écran Streamlit (son chemin de saisie, `synthese.ed_rd`,
`acier.formats.pct`), à l'arrondi d'affichage près. Seul écart : les trois cas « pile au-dessus » (η = 1 + 10⁻¹⁰),
corrigés (point 3). Tests : `tests/test_api_web_assemblages.py` partie 12 (19 contrôles, dont trois mutations) —
160 contrôles en tout ; poutre–poteau 138 ; pied de poteau 90 (`CONTRAT_SEPT=1`) ; contrat commun 17 / 17 ; contrôle
navigateur `verifier_site.js module:assemblages` 23 / 23 et parcours de relecture (`scratchpad/rv_parcours.js`,
`rv_parcours2.js` : poutre–poutre sur PC, tablette et téléphone, note « à corriger », n₁ = 10⁹, texte piégé,
cornières soudées, colonne « Excel », pile à la limite) au vert, captures regardées.
