# Maturité du béton — le moteur sans Streamlit

*Livré le 07/11/2026. Moteur `beton/maturite/` : bibliothèque standard et numpy (présent dans Pyodide 0.29.5),
Python 3.12 et 3.13. Extrait de l'écran Streamlit `modules/age_beton.py` VERSION 3.2, qui reste la **référence** ; la
preuve de parité est `tests/test_maturite_moteur.py`. Le module du site est `api_web/age.py` (établi
`maturite-beton`), ses dessins `api_web/dessin_age.py`, ses tests `tests/test_api_web_age.py`. Relu le 07/11/2026 (relecture
indépendante : chiffres de bout en bout, couverture, écran sur trois formats, robustesse).*

## 0 · Défauts NON CONSERVATIFS de l'écran Streamlit — à trancher par le bureau

Le site les **reproduit** (parité : mêmes chiffres que Streamlit) et les **signale** ; il ne les corrige pas en silence.

1. **② Le seuil est cherché à la température MOYENNE du relevé : quand les jours froids viennent d'abord, il est annoncé
   trop tôt.** Le module calcule ① jour par jour sur le relevé, mais ② (et ③) à température constante, la moyenne.
   Exemple (cas 46 de la parité) : 7 j à 2 °C puis 7 j à 20 °C (moyenne 11 °C), C30/37 ciment N, seuil 50 % = 15 MPa :
   l'écran annonce « après **4,8 j** » ; jour par jour, f_ck(5 j) = **11,36 MPa** et le seuil n'est atteint qu'au
   **jour 8**. Pour un décoffrage, c'est trois jours trop tôt. Le moteur nomme la lecture du relevé (`seuil.releve` :
   premier jour atteint, f_ck au jour annoncé, drapeau `optimiste`, sans formule nouvelle — les valeurs du tableau, celles
   de ①) ; le site affiche alors un **avertissement grave** et la fenêtre du seuil montre le jour du relevé.
2. **⑤ L'essai est lu comme une résistance CARACTÉRISTIQUE.** Une mesure sur éprouvette est plutôt une résistance
   moyenne : le module surestime f_ck,28 de 8/β_cc MPa. Exemple : 21,6 MPa à 7 j, 20 °C, ciment N → le module estime
   **30,0 MPa (C30/37, « cohérent »)**, la lecture en moyenne donne **19,7 MPa (C20/25)**. Le site donne, sous le
   résultat et dans la fenêtre de l'essai, la lecture en moyenne (`essai.fck28_est_moyenne`), « à trancher ».
3. **② La date du seuil ignore la fraction de jour** (non conservatif d'au plus un jour) : 7,93 j après un coulage du
   02/11 s'affiche « le 09/11 » (au plus tôt le 09/11 au soir). Reproduit ; la fenêtre du site le dit.

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/maturite/ec2.py` | les formules du module, **recopiées** (même code, même ordre d'opérations, mêmes appels numpy) : β_cc, f_ck(t), âge équivalent, facteur de maturité, seuil par dichotomie, classe minimale, classe estimée d'un essai, variante défavorable, âge équivalent à un âge donné |
| `beton/maturite/releve.py` | le relevé de températures journalières : sa **lecture** (tableau collé, fichier, ou écrit par la météo récupérée) et sa **synthèse**, recopiée du mode « Météo » de l'écran |
| `beton/maturite/meteo.py` | la **météo du lieu**, sans réseau : le **plan** des appels à Open-Meteo (géocodage, archive, prévisions — `_plages_endpoints` de l'écran) et la **lecture** des réponses (`geocode_open_meteo`, `meteo_journaliere`, `meteo_horaire`), recopiés au bit près ; ce que la note **garde** d'une récupération (§ 4 bis) |
| `beton/maturite/entree.py` | les valeurs de l'écran au premier affichage, ses bornes, la normalisation stricte d'un coulage (ValueError avec le chemin du champ) |
| `beton/maturite/moteur.py` | `calculer` : tout ce que l'écran affiche (et ce que le site ajoute, nommé comme tel) ; `candidats_classes` : le sélecteur de ③ ; le **format complet** (entrée, sortie) dans sa docstring |
| `tests/test_maturite_moteur.py` | parité au bit près avec l'écran rejoué (faux `st`), contrôles à la main, couverture, relevé, format, pureté, **mutations**, **de bout en bout** (ce que l'écran affiche est lu sur le site) et la **météo** (section 10 : plan et lecture contre l'écran, faux `requests`) — 189 contrôles |
| `tests/faux_open_meteo.py` | un faux Open-Meteo au **format documenté** du service (géocodage, archive, prévisions, journalier et horaire, erreurs 400 / 429) : les tests n'appellent jamais le vrai service |
| `api_web/age.py`, `api_web/dessin_age.py`, `site/web/css/modules/age.css`, `tests/test_api_web_age.py` | le module du site, ses graphiques SVG, sa feuille de style, ses tests (66 contrôles, plus les 21 du contrat commun) |

numpy est **nécessaire** à la parité : `np.exp` et `math.exp` peuvent différer au dernier bit, et le module mélange les
deux (`math.exp` pour le facteur de maturité, `np.exp` pour β_cc). Le moteur fait exactement les mêmes appels.

## 2 · L'API

```python
from beton.maturite import moteur as M

M.nouvel_element(nom) -> dict      # C30/37, ciment N, 20 °C constante, 28 j, seuil 75 %, 30 MPa en 14 j, 7 classes, pas d'essai
M.normaliser(element) -> dict      # complété et contrôlé ; ValueError hors du domaine de l'écran (jamais une valeur ramenée)
M.calculer(element) -> dict        # tout ce que l'écran affiche, nombres NON arrondis (le site formate)
M.candidats_classes(element) -> list   # les 7 classes pour la cible de ③ : f_ck au délai, taux, retenue, optimale
```

L'entrée (un coulage) et la sortie complète sont décrites champ par champ dans la docstring de `moteur.py`. Les
grandeurs principales de la sortie :

| Clé | Ce que c'est | Écran Streamlit |
|---|---|---|
| temperature · `T` | la température de calcul : la constante, ou la **moyenne** du relevé (`T_moy` de la synthèse) | `T` de `show()` |
| temperature · `facteur` | jours équivalents à 20 °C par jour réel à T : exp(−E_a/R·[1/(T + 273,15) − 1/293,15]) | `age_equiv(1.0, T, Ea)` |
| temperature · `t28_reel` | l'âge réel où t_e = 28 j : 28 / `facteur` (le repère de la figure) | `t28_real` |
| temperature.releve · `t_e_releve` | t_e du relevé entier (avec E_a de l'EC2, voir §7) | mesure « t_e » de « Données météo » |
| temperature · `defav` | la variante défavorable : jours prévus et prolongés − 3 °C (`T_moy`, profil, comptes) | `defav` |
| age · `t_e` | l'âge équivalent à l'âge choisi : somme jour par jour sur le relevé, sinon t·`facteur` | `t_e_sel` |
| age · `fck` | f_ck(t) = β_cc(t_e)·(f_ck + 8) − 8 si t_e < 28 j, f_ck sinon | `fck_val` |
| age · `pct` | f_ck(t) / f_ck × 100 | `pct_val` |
| age · `hors_domaine` | t_e ≤ 3 j (« hors domaine EC2 ») | `etat == "warn"` |
| seuil · `cible` | la cible du seuil : p·f_ck/100, ou la résistance saisie | `target_s` |
| seuil · `t` (`t_seuil`) | l'âge réel où f_ck(t) atteint la cible, par dichotomie à T constante ; None : non atteignable | `t_seuil` |
| seuil · `date` | date de coulage + `t_seuil` (jours entiers, voir §7) | « le JJ/MM/AAAA » |
| optim · `classe` | la plus petite `classe` (C20/25 → C50/60) qui atteint la cible au délai ; None : aucune | `res_opt` |
| optim · `actuelle` | la classe minimale ne dépasse pas celle du coulage | « (actuelle) » |
| essai · `fck28_est` | (f_mes + 8)/β_cc(t_e) − 8 (f_mes si t_e ≥ 28 j) | `est["fck28_est"]` |
| essai · `classe` | la classe la plus proche de `fck28_est` (à égalité, la plus faible) | `est["classe"]` |
| essai · `ecart` | (`fck28_est` − f_ck)/f_ck × 100, comparé au coulage | `ec` |
| essai · `verdict` | « coherent » si \|`ecart`\| ≤ 8 %, « dessus » au-delà de + 8 %, « sous » sous − 8 % | blocs ⑤ |
| `courbes` | les arguments de `build_figure` : 500 abscisses de 1 à 40 j, référence, défavorable, comparateur, classe ③, point ①, seuil ②, essai ⑤, repère t_e = 28 j, titre | la figure |

Ce que le site ajoute, **sans calcul nouveau** (des lectures des résultats du module, nommées comme telles dans la
docstring) : `seuil.taux_age` = f_cible / f_ck(t) (le seuil est-il atteint à l'âge choisi ?), `seuil.controle` (la
forme fermée t_e = 28/(1 − ln((f_cible + 8)/f_cm)/s)², qui contrôle la dichotomie), `seuil.releve` (le seuil lu dans le
relevé jour par jour, §0), `optim.classes` (les sept classes au
délai, avec leur taux cible/f_ck — la boucle même de `classe_min_pour_delai`), `temperature.t3_reel` = 3/`facteur` (le
domaine t_e ≤ 3 j grisé sur la courbe), `tableau` (les valeurs que la courbe trace, jour par jour, par les fonctions de
①), et `formules` (chaque résultat avec ses termes : « formule = valeurs = résultat », recalculable — le test le prouve).

## 3 · La méthode (EN 1992-1-1 §3.1.2 et annexe B)

    f_ck(t) = β_cc(t_e)·f_cm − 8      pour t_e < 28 j          f_cm = f_ck + 8
    f_ck(t) = f_ck                    pour t_e ≥ 28 j
    β_cc(t_e) = exp(s·[1 − √(28/t_e)])                          s = 0,20 (R) · 0,25 (N) · 0,38 (S)
    t_e = Σ Δt_i·exp(−E_a/R·[1/(273,15 + T_i) − 1/293,15])     E_a/R = 4 000 K (éq. B.10), E_a = 33,256 kJ/mol

Domaine de validité : 3 j < t_e < 28 j (§3.1.2(5)) ; en dessous, la résistance se détermine par essais (le module et le
site le disent). Ciments : R = CEM 42,5R · CEM 52,5N · CEM 52,5R ; N = CEM 32,5R · CEM 42,5N ; S = CEM 32,5N.

- **① Résistance à un âge** : avec un relevé, la somme jour par jour sur ses n premiers jours ; sinon t·facteur.
- **② Seuil** : la dichotomie du module sur [0,05 ; 90] j (80 pas, arrêt à 0,001 MPa), **à température constante**
  — la moyenne du relevé en mode relevé ; non atteignable si la cible dépasse f_ck ou n'est pas atteinte en 90 j.
- **③ Classe pour un délai** : la première classe de C20/25 à C50/60 dont f_ck(délai) + 10⁻⁶ ≥ cible, avec le ciment
  du coulage, à température constante.
- **④ Comparateur** : les classes choisies (sauf celle du coulage) avec le ciment du comparateur, à T.
- **⑤ Essai** : la classe estimée d'après une mesure, comparée à celle du coulage (± 8 %).

## 4 · Le relevé

Le mode « Météo » de l'écran va chercher chez Open-Meteo les températures moyennes journalières (archive + prévision,
jours au-delà de l'horizon de 16 j prolongés à la moyenne des 5 derniers jours prévus). Sur le site, ces températures
deviennent un **relevé** — récupéré chez Open-Meteo à la demande (§ 4 bis), ou **saisi** hors ligne : une ligne par
jour, collée (un tableau Excel, le tableau « Données météo » de Streamlit tel quel) ou lue dans un fichier .csv/.txt :

    9,5                          la température moyenne du jour (°C)
    2026-11-03;8,0               précédée de sa date (AAAA-MM-JJ ou JJ/MM/AAAA), à chaque ligne, jours consécutifs
    04/11/2026;7,0;prévision     suivie de sa source : mesuré (défaut), prévision, prolongé ; « — » : jour sans donnée

Une ligne d'en-tête est ignorée ; les colonnes « T min h », « T max h » du tableau de Streamlit aussi. Une ligne
illisible est **refusée avec son numéro** (le site garde le texte à l'écran avec l'erreur, le calcul garde le relevé
précédent). Une fois les températures en main, la synthèse est celle du module, au bit près : jour sans donnée remplacé
par la moyenne, comptes mesurés / prévus / prolongés, T min / moy / max, variante défavorable (− 3 °C sur les jours
prévus et prolongés), t_e du relevé.

## 4 bis · La météo du lieu (Open-Meteo), importée à la demande

La troisième source de la cure, « Météo du lieu » (`temperature.mode = "meteo"`), à côté de « Constante » et « Relevé ».
Le site calcule **hors ligne** ; la météo est une **importation de données**, faite par le navigateur quand l'ingénieur
clique « Récupérer la météo » :

1. **le plan** (`meteo.plan(lieu, début, fin, aujourd'hui)`) : le géocodage du lieu (`name`, `count=1`,
   `language=fr`), puis l'archive pour les jours jusqu'à J−7 et les prévisions de J−6 à J+16
   (`start_date`, `end_date`, `timezone=auto`, `daily=temperature_2m_mean`) — `_plages_endpoints` de l'écran. Les dates
   par défaut sont celles de l'écran : le début = la date de coulage, sinon aujourd'hui − 14 j ; la fin = max(aujourd'hui,
   début + 5 j). Plan impossible (pas de lieu, fin avant le début, plus de 400 jours, période entièrement au-delà de
   l'horizon) : `ValueError`, le bouton est grisé et le message écrit ;
2. **les appels** : `site/web/js/meteo.js`, vers les trois adresses d'Open-Meteo et elles seules (la politique de
   sécurité du site n'en permet pas d'autre) ; les réponses partent BRUTES au module ;
3. **la lecture** : `lire_geocodage` (le premier résultat ; « nom, région, pays » ; None → « Lieu introuvable »),
   `lire_journalier` (jours sans doublon ; `d_prev` : postérieur au jour de la récupération ; prolongation à la moyenne
   des 5 derniers jours servis, `np.mean`), `lire_horaire` et `extremes_horaires` (T min h, T max h de chaque jour) —
   `meteo_journaliere`, `meteo_horaire`, `geocode_open_meteo` de l'écran, au bit près ;
4. **ce que la note garde** (`temperature.meteo`, `meteo.enregistrement`) : le lieu demandé et trouvé (nom, région, pays,
   position), les dates, le jour du plan et l'heure de la récupération, les réponses **réduites à ce qui est lu** (les
   instants et les températures), et le détail horaire s'il a été chargé ; relu STRICTEMENT par `meteo.normaliser` (un
   .stab vient d'ailleurs : clés connues, nombres bornés, réponses qui se relisent). Les températures sont écrites en
   **relevé** (`meteo.releve`), que l'ingénieur peut corriger à la main ; `meteo.synthese` dit alors `modifie` (le relevé
   n'est plus celui reçu) et `demande_changee` (le lieu ou les dates ne sont plus ceux de la récupération). Le calcul suit
   le relevé — la suite est celle du § 4, et la note se recalcule **sans aucun appel**.

Ce qui diffère de l'écran : une réponse **mal formée** (une température en texte ou en booléen, des dates qui n'en sont
pas, des listes de longueurs différentes, un jour manquant entre l'archive et les prévisions) est **refusée avec un
message** (l'écran l'aurait lue de travers : `float("12.5")`, un relevé décalé) ; une période de plus de 400 jours est
refusée ; le détail horaire d'une période sans jour servi est refusé (bouton grisé) au lieu d'un tableau vide ; « -0.0 »
s'écrit « 0,0 ». Rien de ceci n'arrive avec une réponse conforme au format d'Open-Meteo.

**La preuve** (`tests/test_maturite_moteur.py`, section 10) : l'écran Streamlit va chercher sa météo
(`geocode_open_meteo`, `meteo_journaliere`, `meteo_horaire`) chez un faux `requests` ; le site fait son plan, et le même
faux service (`tests/faux_open_meteo.py`, au format documenté) rend les réponses que le navigateur rendrait : sur
30 scénarios (passé, à cheval, futur, au-delà de l'horizon, bornes J−6 et J+16, jours et heures sans donnée, lieu
introuvable, refus) et 2 113 périodes de `plages`, **mêmes appels** (adresses, paramètres un à un et dans l'ordre),
même lieu trouvé, **mêmes températures au bit près**, mêmes refus ; huit mutations du plan et de la lecture (retard de
l'archive, horizon, prolongation, fuseau, géocodage, variable, « prévision » dès le jour même, nom du lieu) rougissent
toutes. Cinq coulages « météo » de la section 3 sont rejoués dans l'écran avec la météo que l'écran va **lui-même**
chercher (le faux rend les réponses que la note garde) : badge, mesures, tableau « Données météo » avec T min h / T max h,
figure — tout est identique, et lu sur le site (section 9).

## 5 · La preuve de parité

`tests/test_maturite_moteur.py` rejoue `show()` de l'écran avec un faux `st` : chaque coulage est traduit en saisies
(par clé de widget, sinon par libellé), le mode Météo est rejoué avec la météo **déposée dans `session_state`** comme
après « Récupérer la météo » ; des espions relèvent `_bloc`, `_mini_metrics`, `build_figure` (toutes ses données),
`t_real_for_target`, `classe_min_pour_delai`, `classe_estimee_depuis_mesure`, `age_equiv_from_series`, le tableau
« Données météo », les légendes, et le dict `ab_rapport`.

- **113 coulages** (46 écrits pour couvrir chaque branche, 21 classes × ciments, 40 tirés au sort, graine fixe, 6 en
  météo du lieu) ;
- **171 526 comparaisons numériques, toutes identiques AU BIT PRÈS** (les 500 points de chaque courbe compris) ;
  les textes de l'écran (blocs, légendes, badge du relevé) sont réécrits depuis le moteur et comparés ;
- **couverture** : 48 situations atteintes (dont la météo : récupérée, archive, prévisions, prolongée, détail horaire,
  pas encore récupérée) (chaque état de ①②③⑤, chaque ciment, chaque classe, t_e ≥ 28, seuil
  daté, E_a ajusté, essai cohérent / dessus / sous / t_e ≥ 28 / hors domaine, repère t_e = 28 j dans et hors de l'axe,
  relevé prévu / prolongé / sans donnée / défavorable / vide, âge au-delà de l'axe…) ;
- **valeurs par défaut** : `nouvel_element()` est l'écran au premier affichage, lu dans les widgets rejoués ;
- **à la main** : β_cc, t_e, f_ck(t), seuil (forme fermée), essai et classe minimale recalculés avec `math` seul ;
- **mutations** : onze défauts plausibles glissés dans le moteur (β_cc avec 27, f_cm = f_ck + 7, T_ref = 293,0 K,
  t_e < 3 au lieu de ≤ 3, ± 10 % au lieu de ± 8 %, marge 2,5 °C, ① à la moyenne, tolérance de la dichotomie, E_a ignoré
  par ③, classe inférieure à l'essai, jour sans donnée à 0 °C) : la parité rougit **à chaque fois** ;
- **pureté** : dans un sous-processus où `sys.meta_path` refuse streamlit, matplotlib, pandas, requests, scipy…, le
  moteur se charge et calcule les 107 coulages, sous 3.12 et 3.13, avec les mêmes résultats au bit près ;
- **de bout en bout** (section 9, ajoutée à la relecture) : sur 39 coulages variés (dont 6 en météo du lieu), chaque valeur que l'écran Streamlit
  AFFICHE (blocs ①②③⑤, légendes t_e et défavorable, mesures et badge du relevé, tableau « Données météo », annotations et
  légendes de la figure : 486 valeurs) est lue, au même arrondi, dans les fragments HTML et SVG que le site rend
  (`api_web/routeur.traiter`). Cinq défauts d'affichage glissés dans le site (arrondi, date, valeur de l'essai, t_e du
  relevé, pourcentage tronqué) la font rougir.

## 6 · Ce que le site ne couvre pas (par rapport à Streamlit)

- Le **cache** de 30 min de l'écran (une seconde demande identique ne repart pas) : sur le site, la récupération est
  GARDÉE dans la note et ne repart qu'à un nouveau clic. Le re-essai unique après un « trop de demandes » (429) est fait
  par le navigateur (`js/meteo.js`), comme l'écran. La récupération météo elle-même est couverte (§ 4 bis).
- La structure `ab_rapport` (préparée pour un futur PDF, sans PDF dans l'écran) : le bouton Note PDF du site est grisé.

Tout le reste est sur le site (l'inventaire champ par champ est le test n° 1 de `tests/test_api_web_age.py`), avec en
plus : plusieurs coulages par note, la fenêtre de calcul de chaque analyse (formules substituées), le sélecteur de
classe trié (✓ la classe du coulage, ★ la plus petite qui tient), la vue « Températures » (barres par source,
défavorable, moyenne, t_e cumulé), la vue « Tableau » (les valeurs jour par jour), le domaine t_e ≤ 3 j grisé.

## 7 · Défauts constatés dans le module Streamlit (reproduits, non corrigés)

1. **⑤ L'essai est lu comme une résistance CARACTÉRISTIQUE — non conservatif (§0, n° 2).** Le module inverse
   f_ck(t) = β_cc·(f_ck + 8) − 8 : f_ck,28,est = (f_mes + 8)/β_cc − 8. Or une mesure sur éprouvette est une résistance
   *moyenne* (f_cm(t) = β_cc·f_cm, EC2 éq. 3.1) ; lue ainsi, f_ck,28 = f_mes/β_cc − 8 (f_mes − 8 à 28 j et plus).
   Exemple : 21,6 MPa à 7 j, 20 °C, ciment N : le module estime **30,0 MPa (C30/37, « cohérent »)**, la lecture en
   moyenne donne **19,7 MPa (C20/25)**. L'écart est 8/β_cc MPa : 8 MPa à 28 j, 10,3 MPa à 7 j, davantage au jeune âge.
   À trancher par le bureau ; le site reproduit le module et donne, dans la fenêtre de l'essai, la lecture en moyenne
   (`essai.fck28_est_moyenne`, pour information).
2. **② La date du seuil ignore la fraction de jour — non conservatif d'au plus un jour (§0, n° 3).** `date + timedelta(days=7,93)`
   rend coulage + 7 jours : « le 09/11/2026 » pour un seuil atteint 7,93 j après un coulage du 02/11 (au plus tôt le
   09/11 au soir). Reproduit ; la fenêtre du site le dit.
3. **② et ③ ignorent le relevé** : le seuil, la classe minimale et la courbe sont calculés à la température MOYENNE du
   relevé, alors que ① suit le relevé jour par jour. **Non conservatif quand les jours froids viennent d'abord (§0, n° 1)**.
   Reproduit ; le site l'écrit sous le résultat, et l'avertit en grave quand le relevé dément le jour annoncé.
4. **Le t_e de « Données météo » garde E_a de l'EC2** même quand l'énergie d'activation est ajustée (le module lit
   ⚙️ après la saisie des températures). Reproduit (`releve.synthese`).
5. **f_ck(t) négatif au très jeune âge** : β_cc·f_cm < 8 donne une résistance négative (−3,99 MPa à 1 j à −5 °C), que
   le module affiche et trace. C'est hors domaine (t_e ≤ 3 j) et signalé comme tel ; le site l'affiche aussi, et le
   taux d'une classe dont f_ck(délai) ≤ 0 est « sans objet » (999 %) au lieu d'un infini.
6. **③ « (actuelle) » aussi quand la classe minimale est INFÉRIEURE** à celle du coulage (C25/30 minimal pour un
   C45/55 : « Classe minimale : C25/30 (actuelle) »). Le libellé veut dire « la classe du coulage suffit » ; le site
   l'écrit ainsi.
7. **Bornes des champs** : l'écran ramène en silence une saisie hors bornes (T < −5 °C, E_a < 20, délai < 0,5 j…) à la
   borne ; le site la **refuse avec un message** qui nomme le champ. L'écran n'a pas de borne HAUTE pour le délai ③,
   l'âge de l'essai ⑤ et les résistances saisies : le site en met une (`JOURS_MAX` = 3 650 j, `MPA_MAX` = 200 MPa,
   `entree.py`) — au-delà rien n'est physique, et un délai de 10³⁰⁸ j à 60 °C déborde en infini (la page plantait sur un
   message Python). Le champ « Résistance (MPa) » du seuil est borné à
   f_ck, mais une valeur saisie reste quand la classe baisse (le seuil devient « non atteignable ») ; le site refuse la
   frappe au-dessus de f_ck et garde, comme l'écran, une valeur déjà saisie.
8. Les valeurs par défaut de ③ (cible = f_ck(t) arrondi, délai = max(3 ; t/2)) et de la T de cure de ⑤ (= T) sont
   celles du **premier affichage** (widgets à clé) : elles ne suivent pas les changements suivants. Le site part des
   mêmes valeurs ; elles se changent à la main.

## 8 · L'écran du site (ce que la relecture a ajusté)

- **Le résultat** dit la réponse de chaque bloc de l'écran : ① la résistance (en grand) et sa ligne ; la variante
  défavorable ; ② « cible · après t j · date » ; ③ **« Classe minimale » : la classe et f_ck au délai** (★ ; un clic ouvre
  les sept classes, ✓ celle du coulage) — auparavant la ligne montrait la classe du coulage en rouge et ne nommait la
  classe minimale qu'au survol ; ⑤ la classe estimée et f_ck,28, puis le verdict de l'écran (« sous C30/37 (écart
  −15 %) », grave pour « sous ») et, à part, la lecture en moyenne à trancher.
- **La courbe** : légende au-dessus du tracé (elle cachait les courbes des petites classes), axe des résistances arrêté
  juste sous une valeur négative (il descendait à −10 pour −1), étiquette du point ① posée là où elle ne couvre ni un
  repère ni une courbe, noms des classes entiers (« C35/45 ③ » était coupé), 580 unités de large (textes de 10 à 11 px
  sur téléphone au lieu de 6 à 7).
- **La tablette en hauteur** : le graphique était coupé à droite (la grille de la feuille prenait la largeur qu'il
  aurait eue à pleine hauteur) ; `age.css` borne les pistes.
- **Le signe moins** des températures négatives (« −5 °C », plus « -5 °C »).
- **Les dates sous l'axe du relevé** (vue « Températures », 07/11/2026, vu sur la capture de la météo du lieu au
  téléphone) : au-delà de 12 jours, le jour 1 et le jour 2 sont voisins alors que les autres vont de deux en deux, et
  leurs dates se chevauchaient (« 24/1025/10 ») ; une date n'est plus écrite que si elle a sa place après la précédente
  (`tests/test_api_web_age.py` : relevés de 1 à 60 jours, mutation vue).
