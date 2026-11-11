# Maturité du béton — le moteur et le module du site

*Refait le 10/11/2026 (Lot Q) sur la conception `docs/site/LOT_Q_QUATRE_MODULES.md` § 2. Moteur `beton/maturite/` :
bibliothèque standard seule (plus de numpy), Python 3.12 et 3.13 (donc Pyodide). Il ne reproduit PLUS l'écran
Streamlit `modules/age_beton.py` (dont les défauts restent relevés dans `docs/site/DEFAUTS_STREAMLIT.md`) : il date la
résistance d'un béton jeune selon la méthode ci-dessous, prouvée à la main. Module du site : `api_web/age.py` (adresse
`maturite-beton`), dessins `api_web/dessin_age.py`, feuille `site/web/css/modules/age.css`.*

Le module répond à une question de chantier : **« mon béton, coulé tel jour par ce temps, aura-t-il X MPa tel jour ?
Sinon, quand ? »** — décoffrer un voile, désétayer une dalle, mettre en charge, protéger du gel. Il donne une **date
par objectif**, à l'heure près, avec sa **variante prudente** (des prévisions trop optimistes), et un verdict quand
l'opération a une date. C'est une ESTIMATION : un décoffrage annoncé trop tôt est un défaut non conservatif, donc
**dans le doute, le moteur prend le côté qui annonce la date la plus tardive**, et la règle est écrite (ici, sous le
résultat et dans les fenêtres de calcul).

## 1 · Les fichiers

| Fichier | Rôle |
|---|---|
| `beton/maturite/lois.py` | les TABLES des normes (classes, s du ciment, classes de durcissement du béton, tableau k de la NBN B 15-400), chacune une fois avec sa source ; les formules : β_cc, f_cm(t), f_ck(t), l'âge équivalent requis en forme fermée, l'intégration de t_T sur un profil journalier et son inverse exact, le calage de s sur un essai |
| `beton/maturite/entree.py` | le COULAGE d'une note neuve ; les bornes ; `normaliser` (strict : `ValueError` en français qui nomme le champ, jamais une valeur ramenée à une borne) ; la relecture d'une note de l'ancien module |
| `beton/maturite/moteur.py` | `calculer(coulage)` : le profil jour par jour, les objectifs, le gel, le rappel « temps froid », l'essai, les courbes, le tableau, les avertissements ; `candidats_classes` (le sélecteur) ; `MODELES_OBJECTIF` ; `EXPRESSIONS` (le gabarit de chaque ligne de calcul) — le FORMAT de sortie dans sa docstring |
| `beton/maturite/releve.py` | le relevé de températures journalières : sa lecture (collé, ou lu dans un fichier), son écriture, sa synthèse |
| `beton/maturite/meteo.py` | la météo du lieu, SANS réseau : le plan des appels et la lecture des réponses d'Open-Meteo (§ 4 bis) |
| `api_web/age.py` | l'écran (établi compact) : rubriques, verdict, blocs, fenêtres de calcul, sélecteur de classe |
| `api_web/dessin_age.py` | les vues : la courbe des résistances, les températures jour par jour |

## 2 · Le coulage

`{nom, classe (C16/20 … C50/60), durcissement {mode: ciment | beton | r, ciment: R | N | S, beton: rapide | moyen |
lent, r}, date (null | AAAA-MM-JJ), heure (0 à 23), temperature {mode: constante | releve | meteo, T (°C), releve,
fichier, lieu, debut, fin, meteo}, marge (°C), objectifs [...], loi {mode: nbn | ec2 | perso, Ea_kJ}, essai {actif,
age (j), mpa, eprouvette: cube | cylindre, caler}}`.

- un **objectif** : `{nom, mode: pct | mpa, pct (5 à 100 % de f_ck), mpa (0,5 à 100), base: fck | fcm (ce à quoi la
  cible se compare ; en %, toujours f_ck), age (l'âge PRÉVU de l'opération, 0,1 à 365 j ; null = non daté)}` — huit au
  plus ;
- **t = 0 est la FIN DU BÉTONNAGE** (date et heure entière ; midi si l'heure n'est pas donnée) ;
- bornes : T de −20 à 60 °C ; marge de 0 à 10 °C ; r de 0,15 à 0,70 ; E_a de 20 à 60 kJ/mol ; l'essai jusqu'à 150 MPa.
- **Défaut à l'ouverture** : C30/37, béton à durcissement **moyen**, fin du bétonnage à midi sans date, 20 °C
  constante, loi NBN B 15-400, marge 3 °C, un objectif « Décoffrage » à 70 % de f_ck, non daté ; aucun essai.
- Une note de l'ancien module (ses clés : `ciment`, `date_coulage`, `seuil`, `optim`, `comparateur`…) se rouvre sur des
  coulages NEUFS qui gardent leurs noms : les formules ont changé, aucune conversion n'est tentée, et l'écran le dit
  une fois.

## 3 · La méthode (et ses sources)

    f_cm(t) = β_cc(t_T)·f_cm                 f_cm = f_ck + 8                    EN 1992-1-1:2004 §3.1.2(6), (3.1)
    β_cc(t_T) = exp{ s·[1 − √(28/t_T)] }                                         (3.2)
    f_ck(t) = f_cm(t) − 8  pour 3 < t_T < 28 j ;  f_ck pour t_T ≥ 28 j           §3.1.2(5)
    t_T = Σ Δt_i · k(T_i)                    l'âge équivalent à 20 °C

**Le durcissement — le coefficient s.** Deux lectures, au choix :

| Lecture | Choix | s | r = f_cm,2j / f_cm,28j supposé |
|---|---|---|---|
| le CIMENT — EN 1992-1-1 §3.1.2(6) | R : CEM 42,5 R, 52,5 N, 52,5 R | 0,20 | 0,58 |
| | N : CEM 32,5 R, 42,5 N | 0,25 | 0,50 |
| | S : CEM 32,5 N | 0,38 | 0,35 |
| le BÉTON — développement de la résistance de la NBN EN 206, écrit sur le bon de livraison | rapide : r ≥ 0,50 | 0,253 | 0,50 |
| | moyen : 0,30 ≤ r < 0,50 | 0,439 | 0,30 |
| | lent : 0,15 ≤ r < 0,30 | 0,692 | 0,15 |
| | r connu (centrale, essais) | −ln r / (√14 − 1) | r |

s se déduit de r par (3.2) écrite à 2 jours : r = β_cc(2) = exp{s·(1 − √14)}. Pour une classe du bon de livraison, le
moteur prend s à la **borne basse** de la classe : le durcissement le plus lent qu'elle admet. Le ciment le plus
courant en Belgique (CEM III/A 42,5 N LA) est « N » par la lettre de l'EC2 mais « moyen » sur les bons de livraison :
la lecture par le ciment est alors optimiste (à 3 jours, β_cc = 0,60 en N contre 0,41 en « moyen »). **Le défaut
d'ouverture est donc « béton à durcissement moyen »**, et l'écran montre toujours le r que le choix suppose.

**La température — le coefficient k(T)** :

| T (°C) | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| NBN B 15-400 (par défaut) | 0 | 0,15 | 0,30 | 0,45 | 0,60 | 0,80 | 1,00 | 1,00 |
| EC2 annexe B, (B.10) | 0 | 0 | 0,37 | 0,48 | 0,62 | 0,79 | 1,00 | 1,57 |
| E_a = 45 kJ/mol (exemple) | 0 | 0 | 0,26 | 0,37 | 0,52 | 0,73 | 1,00 | 1,84 |

- **NBN B 15-400** (complément belge à la NBN EN 13670, « âge fictif ») : les six points de −5 à 20 °C, interpolés
  linéairement — l'exemple de la norme (trois jours à 5, 7 et 5 °C = 1,41 j) est reproduit par un test. **Deux choix
  du moteur, qui ne sont pas dans la norme** : sous −5 °C, la droite est prolongée jusqu'à k = 0 à −10 °C ; au-dessus
  de 20 °C, k reste à 1,00 (aucun bonus compté). Valeurs relevées sur deux sources concordantes (le projet d'annexe
  devenu NBN B 15-400) : **à contrôler sur l'édition 2024 du texte.**
- **EC2 (B.10)** : k = exp{−4 000·[1/(273 + T) − 1/293]}, de 0 à 80 °C ; sous 0 °C, hors domaine : k = 0, et un
  avertissement le dit. (B.10) écrit « 13,65 » pour 4000/293 = 13,652 : le moteur prend la référence exacte, k = 1 à
  20 °C. Écrite pour la cure thermique : son emploi pour un béton coulé en place est une extension.
- **Arrhenius, E_a saisie** (20 à 60 kJ/mol ; R = 8,314 J/(mol·K)) : pour un béton dont la centrale donne l'énergie
  d'activation. (B.10) suppose 33,26 kJ/mol.
- **Un écart mesuré avec la conception**, qui écrivait « la loi belge n'est jamais plus rapide que (B.10) sous 20 °C » :
  c'est vrai de 0 à 12,2 °C (par temps froid la loi belge compte nettement moins), mais de **12,3 à 19,9 °C le tableau
  belge dépasse (B.10) de 1,5 % au plus** (0,80 contre 0,789 à 15 °C). L'écart est borné par un test ; la loi belge
  reste le défaut — c'est la norme d'ici, et l'écart est sans effet devant la marge de la variante prudente.
- **Ce que « prudente » veut dire** : la loi belge est plus prudente que (B.10) de 0 à 12 °C. Sous 0 °C elle compte
  encore un peu de durcissement (0,30 à 0 °C) là où (B.10) et Arrhenius comptent 0 — c'est l'alerte de gel qui commande
  alors. Et une énergie d'activation de plus de 36 kJ/mol (les ciments au laitier montent à 45–60 kJ/mol) donne MOINS
  que le tableau belge au froid (à 5 °C : 0,37 pour 45 kJ/mol, contre 0,45) : l'aide de l'écran le dit, et la loi
  « E_a saisie » sert à cela.
- La température est celle du BÉTON ; à défaut de mesure, celle de l'air — une hypothèse, dite à l'écran.

**Le temps.** La température est constante par jour civil ; le jour du bétonnage ne compte que pour la fraction qui en
reste, (24 − heure)/24. t_T(t) est donc linéaire par morceaux : il s'intègre et s'inverse **exactement**, sans
dichotomie (`lois.age_equivalent`, `lois.age_reel`). Au-delà du dernier jour connu, le relevé est PROLONGÉ à la moyenne
de ses cinq derniers jours, et c'est dit (avertissement, et l'objectif concerné est nommé).

**Un objectif.** La cible : p % de f_ck, ou une valeur en MPa. Ce à quoi elle se compare : la résistance
CARACTÉRISTIQUE f_ck(t) (une exigence de calcul) ou la résistance MOYENNE f_cm(t) (un seuil de chantier — ce que
mesurent des cubes conservés sur chantier).
- l'âge équivalent requis, en forme fermée : **t_T,req = 28/(1 − ln β_req/s)²**, β_req = (cible + 8)/f_cm
  (caractéristique) ou cible/f_cm (moyenne) ; une cible au-dessus de ce que la classe donne à 28 jours est « non
  atteignable » ; puis l'âge réel t se lit en inversant t_T(t) ;
- **aucun gain n'est compté au-delà de 28 jours équivalents** (β_cc plafonné à 1) : une échéance de chantier ne
  s'appuie pas sur le gain d'après 28 jours ; f_ck(t) n'est jamais négatif ;
- l'**instant** est donné à l'heure près, arrondi à l'heure SUPÉRIEURE (une échéance n'est jamais annoncée plus tôt que
  le calcul), jamais tronqué au jour ;
- **avant 3 jours équivalents**, la valeur est marquée « indicative — à confirmer par essais » (hors du domaine où
  l'EC2 donne f_ck(t) = f_cm(t) − 8) ;
- **opération datée** (l'objectif porte un âge prévu) : η = cible / f(t_prévu) ; « conforme » si η ≤ 1, « conforme,
  attention » si la variante prudente ne passe pas, « non conforme » sinon ;
- six **modèles**, chacun avec sa source : « Hors gel » et « Décoffrage vertical » 5 MPa, « Reprise de bétonnage »
  10 MPa (moyenne) ; 50, 70 et 100 % de f_ck (caractéristique, à fixer par le bureau d'études).

**La variante prudente.** Les jours PRÉVUS et PROLONGÉS — tous, en température constante — sont abaissés de la marge
(3 °C par défaut) ; les jours MESURÉS ne le sont pas. Chaque objectif donne ses deux dates. Dans les fenêtres de calcul
ses grandeurs portent l'indice « prud » — T_prud = T − marge, k_prud, t_prud — ; t_p reste l'âge PRÉVU d'une opération.

**Le gel** — EN 13670:2009 §8.5(12) : la surface du béton ne descend pas sous 0 °C avant d'avoir 5 MPa. Le moteur
cherche le premier jour de gel (la minimale horaire sous 0 °C quand le détail horaire de la météo est chargé, sinon la
moyenne du jour à 0 °C ou moins — et il dit laquelle) et le compare à l'âge où f_cm(t) atteint 5 MPa : alerte si le gel
vient avant (le jour de gel est pris à son début). **Le rappel « temps froid »** (Buildwise, NIT 285 §9.4) : une
température sous 5 °C au bétonnage ou dans les 72 premières heures.

**L'essai.** La moyenne d'éprouvettes de chantier à un âge réel (un cube de 150 mm ramené au cylindre par
f_ck/f_ck,cube de la classe) face à la prévision f_cm(t) : l'écart en %. « Caler la courbe sur l'essai » :
s_cal = ln(f_mes/f_cm)/(1 − √(28/t_T)), accepté de 0,10 à 0,80 (hors de là le calage est refusé, la raison écrite ; un
essai à 28 jours équivalents ou plus ne renseigne pas le durcissement). Toutes les dates suivent, et le r calé
s'affiche. L'essai est lu comme une MOYENNE — l'ancien écran le lisait comme une résistance caractéristique.

**Le sélecteur de classe** (`candidats_classes`) : chaque classe avec le même durcissement et les mêmes températures,
face au premier objectif — ✓ la classe du coulage, ★ la plus petite qui tient l'âge prévu.

## 4 · Le relevé

Une ligne par jour, **du jour du bétonnage au dernier jour**, collée (un tableau Excel, le tableau « Données météo »
de Streamlit tel quel) ou lue dans un fichier .csv/.txt :

    9,5                          la température moyenne du jour (°C), virgule ou point
    2026-11-03;8,0               précédée de sa date (AAAA-MM-JJ ou JJ/MM/AAAA), à chaque ligne, jours consécutifs
    04/11/2026;7,0;prévision     suivie de sa source : mesuré (défaut), prévision, prolongé ; « — » : jour sans donnée

Une ligne d'en-tête est ignorée ; une ligne illisible est **refusée avec son numéro** (le texte reste à l'écran avec
l'erreur, le calcul garde le relevé précédent). Le moteur pose le relevé sur le CALENDRIER du coulage :

- les jours du relevé d'**avant** la fin du bétonnage ne comptent pas (information) ;
- un relevé qui commence **après** le bétonnage laisse des jours sans température, qui ne comptent AUCUN durcissement
  (avertissement grave : compléter le relevé, ou corriger la date) ;
- un **jour sans donnée** (« — ») ne compte AUCUN durcissement — l'ancien moteur le remplaçait par la moyenne du relevé,
  ce qui inventait de la maturité ;
- sans date de bétonnage, celle de la première ligne d'un relevé daté la donne ;
- un mode « relevé » sans relevé exploitable se calcule à la température constante saisie, et c'est dit.

## 4 bis · La météo du lieu (Open-Meteo), importée à la demande

La troisième source des températures, « Météo du lieu » (`temperature.mode = "meteo"`). Le site calcule **hors
ligne** ; la météo est une **importation de données**, faite par le navigateur quand l'ingénieur clique « Récupérer la
météo ». La lecture est celle de l'écran Streamlit, gardée telle quelle et prouvée contre lui (§ 7) :

1. **le plan** (`meteo.plan(lieu, début, fin, aujourd'hui)`) : le géocodage du lieu (`name`, `count=1`,
   `language=fr`), puis l'archive pour les jours jusqu'à J−7 et les prévisions de J−6 à J+16 (`start_date`,
   `end_date`, `timezone=auto`, `daily=temperature_2m_mean`). Dates par défaut : le début = la date du bétonnage, sinon
   aujourd'hui − 14 j ; la fin = max(aujourd'hui, début + 5 j). Plan impossible (pas de lieu, fin avant le début, plus
   de 400 jours, période entièrement au-delà de l'horizon) : `ValueError`, le bouton est grisé et le message écrit ;
2. **les appels** : `site/web/js/meteo.js`, vers les trois adresses d'Open-Meteo et elles seules (la politique de
   sécurité du site n'en permet pas d'autre) ; les réponses partent BRUTES au module ;
3. **la lecture** : `lire_geocodage` (le premier résultat ; None → « Lieu introuvable »), `lire_journalier` (jours sans
   doublon ; « prévision » = postérieur au jour de la récupération ; prolongation à la moyenne des 5 derniers jours
   servis), `lire_horaire`, `extremes_horaires` (T min h, T max h de chaque jour) et `minimales_apres` (la plus basse
   valeur horaire de chaque jour, comptée à partir de l'heure de fin du bétonnage le premier jour : le gel) ;
4. **ce que la note garde** (`temperature.meteo`) : le lieu demandé et trouvé, les dates, le jour du plan et l'heure de
   la récupération, les réponses **réduites à ce qui est lu**, le détail horaire s'il a été chargé ; relu STRICTEMENT
   par `meteo.normaliser` (un .stab vient d'ailleurs). Les températures sont écrites en **relevé**, que l'ingénieur
   peut corriger à la main ; `meteo.synthese` dit alors `modifie` et `demande_changee`. La note se recalcule **sans
   aucun appel**.

Une réponse **mal formée** (une température en texte, des dates qui n'en sont pas, des listes de longueurs différentes,
un jour manquant entre l'archive et les prévisions, une date qui n'existe pas) est **refusée avec un message** en
français. **Relecture adverse du 07/11/2026**, trois trous bouchés et figés par un contrôle chacun : les jours reçus
doivent être ceux DEMANDÉS (une archive décalée d'un jour faisait glisser tous les âges, ou comptait la veille du
coulage — non conservatif) ; une date qui n'existe pas est refusée en français ; des températures effacées à la main
après une récupération donnent l'avertissement `meteo_effacee`, et non « pas encore récupérée ». Côté écran : le plan
du bouton porte le rang du coulage (une récupération revenue alors qu'un autre coulage est ouvert est refusée), et une
réponse n'est lue que jusqu'à 5 Mo.

## 5 · L'écran (`api_web/age.py`)

- **Rubriques** : 1 Béton (la classe ; le durcissement, un menu en deux groupes — le bon de livraison, puis le ciment
  de l'EC2 — et dessous le r qu'il suppose ; la fin du bétonnage, date et heure) · 2 Températures (constante, relevé
  collé ou lu, météo du lieu) · 3 Objectifs (une carte par objectif : son nom, sa cible, la résistance à laquelle elle
  se compare, l'âge prévu facultatif ; « + Objectif » et les six modèles) · 4 Hypothèses (la loi de température, la
  marge sur les prévisions, l'essai et son calage).
- **Vues** : la courbe des résistances (f_cm(t) et f_ck(t) en âge réel, les dates sous l'axe, chaque objectif avec son
  point, la variante prudente en tirets, le domaine t_T ≤ 3 j grisé, l'essai), les températures jour par jour (par
  source, le gel, l'âge équivalent cumulé), le tableau jour par jour (T, k, t_T, β_cc, f_cm, f_ck).
- **Résultats** : le VERDICT (le premier objectif : son âge et sa date, sa variante prudente ; son taux s'il est
  daté) ; les blocs « Objectifs », « Températures » (les jours par source ; « Jours suivants » : la température
  supposée après le dernier jour connu et son k ; l'âge équivalent ; le gel), « Essai » ; les avertissements du
  moteur ; la méthode, ses sources et ce qui n'est pas couvert.
- **Fenêtres de calcul** : chaque ligne « symbole = expression = valeurs = résultat » est le gabarit du moteur
  (`moteur.EXPRESSIONS`) où chaque terme est remplacé par sa valeur — l'écran n'écrit aucune formule, et un test
  réévalue chaque ligne affichée à partir des valeurs écrites.
- Toute saisie passe par `entree.normaliser` : hors domaine, elle est REFUSÉE avec le message du moteur, et le coulage
  reste intact. Une date de fin du bétonnage lue sur le relevé se voit dans son champ (en grisé) et se dit dessous.
- **Un défaut de franchise trouvé sur un cas d'hiver, le 11/11/2026** (une dalle coulée le 16/11, onze jours de relevé,
  décoffrage prévu à J + 7) : à J + 7 le béton n'a que 5,7 MPa, et l'écran écrit aussi « 21 MPa atteints J + 28,4 » —
  une échéance qui tombe APRÈS le relevé, donc calculée sur des températures prolongées (2,5 °C). Le moteur ne regardait
  que le jour PRÉVU d'un objectif daté et se taisait. Il signale maintenant toute échéance AFFICHÉE (la nominale, la
  prudente, le jour prévu) qui repose sur la prolongation : l'avertissement nomme l'objectif et la température.
- **Relecture des captures du 11/11/2026** (PC, tablette, téléphone), figée par six contrôles et douze contre-épreuves :
  la notation de la variante prudente ; la dernière colonne des fenêtres (« Résultat » : un taux ou une échéance, jamais
  « Taux » sur une échéance) ; la fenêtre du gel titrée ; les légendes en mots ; la légende du tableau (ligne surlignée :
  l'objectif ; numéro bleu : un jour de gel) ; plus aucune phrase coupée dans les résultats.
- **Aucune seconde ligne de résultat coupée** : elle ne se replie pas, et sa place a été MESURÉE au navigateur aux six
  formats (313 px au PC, le plus étroit : 59 caractères). « Jours » compte « 10 j mesurés · 16 prévus · 12 prolongés ·
  1 sans donnée » ; l'essai sur cube n'écrit qu'une fois l'unité. Tenu par un contrôle (58 caractères au plus, sur
  69 lignes de treize écrans, dont les cas les plus longs).
- **« Gel avant 5 MPa » dit sa date et sa base** : « 5 MPa (f_cm) à J + 1,9 · gel le 21/11 (moyenne du jour) ». Le gel
  est jugé sur la minimale horaire quand le détail horaire de la météo est chargé, sinon sur la MOYENNE du jour — et une
  nuit peut geler sous une moyenne positive : l'ingénieur doit savoir laquelle des deux il lit. La variante prudente se
  lit dans la pastille (« à surveiller ») et dans l'avertissement du moteur.

## 6 · Ce qui n'est pas couvert (dit, jamais tu)

- **c'est une estimation** : pour une opération critique (désétaiement, mise en charge, précontrainte), la résistance
  se confirme par des éprouvettes conservées sur chantier ou une maturométrie étalonnée, et le délai se valide avec le
  bureau d'études ;
- les délais minimaux de décoffrage par type de coffrage de la NBN B 15-400 et les durées de cure (tableau F.8-ANB) :
  leurs valeurs n'ont pas pu être relues sur le texte ;
- les critères en proportion des charges (charge au décoffrage rapportée à la charge de service) ;
- les flèches au jeune âge (module E_cm(t)) et le ré-étaiement sur plusieurs niveaux ;
- l'EN 1992-1-1:2023 (annexe B.4, classes CS, CN, CR) : pas encore d'annexe belge ;
- la chaleur d'hydratation et la température propre d'un élément massif ou isolé : la température est celle qui est
  saisie ;
- un béton « très lent » (r < 0,15) n'est pas proposé ; la note PDF du module n'existe pas encore.

## 7 · Les preuves

| Suite | Contrôles | Ce qu'elle prouve |
|---|---|---|
| `tests/test_maturite_moteur.py` | 208 contrôles, 46 mutations rejetées | chaque table et chaque loi à la main (`math` seul) ; la forme fermée face à un balayage numérique fin ; l'heure de fin de bétonnage, le profil et son inverse ; l'exemple complet ci-dessous ; les objectifs, les deux bases, la variante prudente, l'opération datée, les instants à l'heure près ; le gel et le temps froid ; l'essai et le calage ; le sélecteur ; chaque borne refusée ; le format (JSON strict, chaque ligne de calcul retrouvée à partir de ses seuls termes) ; le relevé (10 relevés mal formés refusés) ; la météo contre l'écran Streamlit rejoué avec un faux service (30 scénarios, 2 113 périodes : mêmes appels, mêmes températures à 1e-12 près, mêmes refus — le vrai service n'est jamais appelé) ; la pureté (un sous-processus où numpy, streamlit, matplotlib, pandas, requests, scipy sont refusés à l'import) |
| `tests/test_api_web_age.py` | 81 contrôles, 7 mutations vues rouges | l'écran = le moteur (chaque nombre affiché) ; chaque formule des fenêtres réévaluée à partir des valeurs écrites ; refus ; les gestes ; une échéance jamais tronquée au jour ni arrondie à l'heure inférieure ; nominal et prudent jamais confondus |
| `tests/test_api_web_modules.py age` | 22 | le contrat commun des modules (parcours, champs, fichiers, éléments) |
| `tests/test_reference_refonte.py age` | 19 cas | la photo gelée (regel accepté le 10/11/2026) |

**Le coulage d'ouverture, à la main** : C30/37, f_cm = 38 MPa ; « moyen » : s = −ln 0,30/(√14 − 1) = 0,4391 ;
« Décoffrage » 70 % de f_ck = 21 MPa (caractéristique) : β_req = 29/38 = 0,7632 ; t_T,req = 28/(1 + 0,2703/0,4391)²
= **10,73 j** ; à 20 °C, k = 1 : atteint à **J + 10,73** (6,46 j avec le ciment N, s = 0,25). Prudent : 17 °C,
k = 0,88 : J + 12,19.

**Un exemple complet, à la main** (`tests/test_maturite_moteur.py` § 3) : C30/37 moyen, fin du bétonnage le
02/11/2026 à 16 h, six jours mesurés à 12, 8, 5, 7, 10 et 15 °C, loi belge, marge 3 °C.
- k = 0,68 ; 0,54 ; 0,45 ; 0,51 ; 0,60 ; 0,80 ; le premier jour ne compte que 8/24 : t_T cumulé = 0,227 ; 0,767 ;
  1,217 ; 1,727 ; 2,327 ; **3,127 j** à la fin du relevé (âge réel 5,33 j) ;
- « Hors gel » 5 MPa (moyenne) : β_req = 5/38 ; t_T,req = 0,887 j ; atteint le troisième jour (k = 0,45) :
  t = 1,333 + (0,887 − 0,767)/0,45 = **1,60 j**, soit le 04/11 à 6 h 25 → annoncé « 04/11 vers 7 h », indicatif ;
- « Décoffrage » 21 MPa (caractéristique) : t_T,req = 10,73 j ; au-delà du relevé, prolongé à (8 + 5 + 7 + 10 + 15)/5
  = 9 °C (k = 0,57) : t = 5,333 + (10,729 − 3,127)/0,57 = **18,67 j** ; prudent à 6 °C (k = 0,48) : **21,17 j** ;
- prévu à J + 14 : t_T = 3,127 + 0,57 × 8,667 = 8,067 j ; β_cc = 0,6845 ; f_ck(14 j) = 38 × 0,6845 − 8 = 18,01 MPa
  pour 21 demandés : **η = 117 %, non conforme**.

**Le défaut n° 1 de l'ancien écran**, rejoué : 7 jours à 2 °C puis 7 jours à 20 °C, C30/37, ciment N, loi (B.10),
15 MPa caractéristique — atteint à **7,23 j** (le huitième jour), et non à 4,8 j comme l'annonçait le calcul à la
température moyenne (11 °C) : trois jours trop tôt pour un décoffrage.
