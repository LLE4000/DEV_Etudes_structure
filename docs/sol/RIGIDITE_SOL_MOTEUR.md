# Rigidité du sol — le moteur sans Streamlit et le module du site

L'écran Streamlit `modules/rigidite_sol.py` v5.0 (« Raideur élastique des sols », avec `sol_base.py`, `sol_theorie.py`,
`sol_import.py`) reste la **référence** et le **secours** : il n'est pas modifié. Son calcul est extrait dans le paquet
`sol/` (bibliothèque standard seule), qui tourne aussi dans le navigateur (Pyodide) ; le site « Stab Méthodes » le met en
forme dans `api_web/sol.py` (clé `sol`, établi `rigidite-sol`).

> **Défauts NON CONSERVATIFS de l'écran Streamlit** (au sens du module lui-même : « un k surestimé est un écart du
> mauvais côté pour les moments de la dalle »). Le moteur les REPRODUIT (parité) ; le site les signale.
>
> 1. **Le conseil « Prolonge le profil : … il faut typiquement descendre à 2,5·B m » ne dit pas d'où se comptent ces
>    2,5·B.** Le profil se saisit depuis le terrain naturel, la zone d'influence commence à l'assise : sous une semelle
>    à D = 3 m de 2 m de large, l'écran conseille « 5 m » alors qu'il faut au moins D + 2,5·B = 8 m. Qui suit le conseil
>    garde un profil tronqué et un k SURESTIMÉ (le message reste affiché, l'erreur n'est donc pas silencieuse). Le site
>    reprend le message et ajoute « D + 2,5·B = … m sous le TN » ; il met la note « À compléter » (un profil épuisé est
>    une donnée incomplète) et pose ce message sous le verdict, avant les lignes.
> 2. **L'abaque (mode 5) n'a pas de longueur L** : la correction de forme de Terzaghi (sable : ·(1 + 0,5·B/L)/1,5) n'est
>    jamais appliquée ; pour une semelle filante sur sable, le k de l'abaque est SURESTIMÉ jusqu'à 1,5 fois. L'écran (et
>    le site) le présentent comme un ordre de grandeur.
> 3. **Profil trop court quand l'arrêt est le plafond 6·B** : k surestimé lui aussi ; l'écran le dit, mais en accusant à
>    tort le profil saisi (défaut 5 ci-dessous).

## 1 · Ce qui est livré

| Fichier | Rôle |
|---|---|
| `sol/base.py` | la base des sols (22 types) : COPIE CONFORME de `modules/sol_base.py` v2.0 (octet pour octet hors de l'en-tête) |
| `sol/theorie.py` | Boussinesq/Newmark, tassement par tranches, k zoné, Robertson, découpage CPT : COPIE CONFORME de `modules/sol_theorie.py` v1.0 |
| `sol/imports.py` | lecture GEF et CSV, contrôle des profondeurs, exports CSV : les fonctions de `modules/sol_import.py` v1.0, sans les PDF (PyMuPDF) |
| `sol/rigidite.py` | ce que l'ÉCRAN fait de ces briques : fondation, sondages, couches, préremplissage par type, import d'un essai (aperçu, découpage, « Remplir le tableau »), k encadré et zoné, enveloppe, profil trop court, données des graphiques, modes 3 à 6, relecture du fichier « Enregistrer », bornes des champs |
| `api_web/sol.py` | le module du site : état de la note (plusieurs fondations), cadre Données, résultat, fenêtres de calcul, vues |
| `api_web/dessin_sol.py` | les graphiques et coupes en SVG (axes, graduations, légendes ; coupes à l'échelle) |
| `site/web/css/modules/sol.css` | la feuille propre au module (k en grand, plages de k, cartes de couches, import, tableaux) |
| `tests/test_sol_moteur.py` | la PARITÉ avec l'écran Streamlit (faux `st`, faux pandas, faux matplotlib), à la main, mutations, pureté |
| `tests/test_api_web_sol.py` | ce que le site affiche = ce que le moteur calcule ; saisies, refus, effets de bord, import, échappement, SVG |

## 2 · La méthode (celle de l'écran, inchangée)

Une **fondation** B × L (B le petit côté : échangés si L < B, modes 1 et 2 seulement), assise à D sous le terrain naturel,
pression de contact q (ELS). Le sol est un profil de couches depuis le TN : épaisseur h, module OEDOMÉTRIQUE M, γ.

1. σ_v0(D) = Σ γ·h au-dessus de l'assise (au-delà du profil : le γ de la dernière couche) ; q_net = max(q − σ_v0(D) ; 0)
   (option « pression NETTE », cochée par défaut) ou q (pression brute).
2. Δσ(z) = q_net · I(z) ; I = 4·I_coin(B/2 ; L/2 ; z) au centre, 2·I_coin(B ; L/2) au milieu du grand côté, 2·I_coin(B/2 ; L)
   au milieu du petit côté, I_coin(B ; L) à l'angle (Newmark ; l'arctangente prise par atan2).
3. w = Σ Δσ_i·h_i / M_i par tranches de 5 cm sous l'assise ; arrêt à la première tranche au-delà de B où
   Δσ ≤ critère · σ'_v0 (critère 10 à 30 %, 20 % par défaut), ou à la fin du profil saisi (« profil épuisé »), ou à 6·B
   (« plafond 6·B »). Une couche SANS module dans la zone d'influence : k non calculable (jamais incompressible).
4. k = q_net / w (kN/m³ ; ÷ 1 000 → MN/m³), aux quatre positions, pour la borne BASSE (M) et la borne HAUTE (M haut de
   l'import CPT) des modules : k encadré [min ; max].
5. Le panneau SCIA : k de chaque sondage à la position de référence ; l'enveloppe des sondages ; le zonage de la dalle
   (min des k bas, max des k haut par position).

Modes 3 à 6 : k = q / w (3) ; sol homogène de 60 m comparé aux formules élastiques (E/(B(1−ν²)·Is), Is = 0,88 et 1,12), à
Vesić et à l'ancienne formule E/(2B) (4) ; abaque des 21 sols de la base, k de plaque 0,30 m corrigé par Terzaghi
(sable ·((B+0,3)/2B)², argile ·0,3/B), q = k·w_réf (5) ; plat en béton k_c = E_c/h_c, h_c = α·min(B, L), mortier en série
1/k_eq = 1/k_c + 1/k_g (6).

L'import d'un essai : GEF (colonnes et nappe du fichier), CSV (en-têtes reconnus), ou tableau collé (site seulement : le
MÊME analyseur CSV ; sans en-tête, colonnes profondeur, qc, fs). Le découpage (`profil_depuis_cpt`) : Ic de Robertson point
par point, regroupement sur la stabilité de Ic (finesse 0,10 à 0,60), fusion des couches plus minces que l'épaisseur
minimale, modules M bas / M haut encadrés. « Remplir le tableau » : type de sol reconnu d'après la classe de Robertson,
Rf = fs/qc, M bas arrondi au dixième, M haut gardé ; la nappe de l'aperçu reportée sur le sondage ET sur la fondation.

## 3 · La preuve de parité (`tests/test_sol_moteur.py` : 117 contrôles, 0 échec)

1. **Copies conformes** : `sol/base.py` et `sol/theorie.py` = les fichiers de l'écran octet pour octet (hors en-tête) ;
   chaque fonction de `sol/imports.py` a la source de celle de `sol_import.py`.
2. **Mêmes fonctions, mêmes flottants** : 6 625 appels de la théorie rejoués dans les deux copies (Newmark, contraintes,
   tassement, k zoné, élasticité, Vesić, Terzaghi, Robertson, découpage CPT), la base sur 48 libellés, l'import sur 51
   fichiers (GEF, CSV français et anglais, tabulations, Rf, cotes, cm, vides, illisibles) — égalité stricte.
3. **À la main** : Newmark contre les tables publiées (0,1752 ; 0,1999 ; 0,0840 ; 0,1202 ; 0,2439 ; 0,0179) ; tassement d'un
   sol homogène recalculé tranche par tranche ; k = q/w ; plat en béton ; théories ; Terzaghi ; l'itération de Robertson
   refaite au bit près.
4. **L'écran rejoué** : `modules/rigidite_sol.show()` exécuté avec un faux `st` (les widgets rendent l'état), un faux
   pandas (un tableau = ses lignes), un faux matplotlib (une figure = ses séries) sur 53 fondations (défauts de l'écran, L < B,
   nappe, critères, positions, profil trop court, plafond 6·B, sondage écarté, tous écartés, couche sans M, bornes encadrées,
   pression nette nulle, assise sous le profil, roche, fondation de 0,2 m, 30 tirages au hasard, sondage importé) et 40
   fondations des modes 3 à 6 : chaque ligne de la colonne Résultats (avertissements, tableaux SCIA et zonage, enveloppe,
   profil trop court, détail, tranches, métriques), chaque série tracée (diffusion, k(B), CPT) et chaque appel à
   `tassement()` comparés au moteur — **3 027 666 comparaisons strictes**, aucune tolérance.
5. **Les couches** : la légende de 244 lignes ; 286 changements de type (préremplissage) ; les boutons (sondage ➕ 📋 🗑️,
   couche ＋ 🗑️).
6. **L'import** : l'aperçu de 46 essais (réglages changés une fois sur trois), « Remplir le tableau » sur 36 essais, les
   exports CSV.
7. **Le fichier « Enregistrer »** de l'écran relu par le moteur (93 notes) ; les défauts du moteur = l'écran neuf.
8. **Format et domaine** : JSON strict, normalisation idempotente, 38 saisies hors bornes REFUSÉES, bornes lues dans les
   widgets de l'écran.
9. **Mutations** (chacune rougit) : pression nette toujours appliquée, B et L jamais échangés, M haut ignoré, un flottant
   altéré au DERNIER BIT, critère 21 %, abaque avec correction de forme, mortier en parallèle, préremplissage quand qc est
   saisi, M bas non arrondi.
10. **Pureté** : sous Python 3.12 et 3.13 (celui de Pyodide), streamlit, pandas, numpy, matplotlib refusés par
    `sys.meta_path` : le moteur se charge, calcule, et rend les mêmes nombres au bit près.
11. **Le SITE contre l'écran** (ajouté à la relecture) : pour les 93 fondations des sections 4 (53 des modes 1 et 2, 40
    des modes 3 à 6), les chiffres que le SITE affiche — passés par le routeur, lus dans ses fragments HTML : tableau
    SCIA, zonage, enveloppe, w et k au centre, z d'influence, q_net, tranches, chaque nombre de chaque avertissement, k
    des modes 3 à 6, tableau de l'abaque — sont ceux que l'écran rejoué affiche, à l'arrondi d'affichage près (le site
    écrit 2 397,10 où l'écran écrit 2397,10). Contre-épreuve : un k du verdict décalé d'un centième, la comparaison
    rougit. C'est ce contrôle qui a trouvé le « 2.00 m » à l'anglaise des motifs d'arrêt (§ 6).

Le site (`tests/test_api_web_sol.py`, 77 contrôles) : le k, le tassement, le zonage affichés sont ceux du moteur ; les cinq
méthodes ; 14 saisies refusées avec un message qui nomme le champ, la valeur et la borne ; préremplissage au changement de
type (40 tirages) ; ajout, copie, suppression de sondages et de couches ; import GEF, tableau collé, « Remplir le tableau »
(= `valider_import`), « Abandonner », fichier illisible ; reprise du fichier Streamlit ; les avertissements et le statut de
la note (écarté, tous écartés, incalculable, profil court, pression nette nulle, plafond, encadré) ; les formules
substituées (σ_v0(D) écrit terme à terme redonne la valeur du moteur sur 60 tirages) ; échappement ; SVG bien formés.
Le contrat commun (`tests/test_api_web_modules.py sol`) : 17 OK, parcours de 200 requêtes sans erreur. La section 9 de
`tests/test_api_web_sol.py` fige les corrections de la relecture (§ 6) : chacune rougit si on la retire.

## 4 · L'écran du site : inventaire de l'écran Streamlit, et où il se trouve

| Écran Streamlit | Site |
|---|---|
| « Que veux-tu faire ? » (6 modes) | section 1 « Fondation » : menu Méthode (une méthode PAR fondation ; plusieurs fondations par note) |
| Fondation B, L, D, q, Nappe (case + profondeur), « k dépend de la fondation » | section 2 « Semelle et charge » (bornes de l'écran refusées, pas ramenées) |
| Options : critère, ν, pression nette | section « Options de calcul » (repliée, comme l'expander de l'écran) |
| Position de référence (radio du panneau SCIA, toujours visible) | section 2, ligne « k SCIA » (ouverte : le cadre Résultat du site n'a pas de champ) |
| Sondages : nom, ➕, 📋, 🗑️, source | section 3 (cartes des sondages, k de chacun, « Ajouter un sondage », menu « Dessiné ») et une section par sondage (nom, source, « Copier ce sondage », « Supprimer ce sondage ») |
| Import GEF/CSV/TXT/ASC : aperçu, avertissements, absence de fs, nappe / finesse / épaisseur, tableau découpé, « Remplir le tableau », CSV des mesures et des couches | bloc « Importer un essai » de chaque sondage (fichier ou tableau collé), carte de l'essai en attente, liste des couches découpées, boutons, liens CSV ; vues « Import » (courbes de l'essai, limites découpées) et « Découpage » (le tableau complet) |
| Couches : h, type, qc, Rf, M, γ, ＋/🗑️, légende (prise en compte, Ic, classe CPT, M encadré / suggéré, roche) ; aides de M (module OEDOMÉTRIQUE) et de γ | une carte par couche (cotes de profondeur, type avec sa description, champs avec leur unité, légende) ; les deux aides sous « Couches, depuis le terrain naturel » |
| « Exporter les mesures brutes en CSV » | lien dans la section du sondage |
| Panneau SCIA : écartés, incalculables, position, tableau, enveloppe, profil trop court, zonage | cadre Résultat (k en grand, lignes par sondage et par zone sur une barre de k commune, messages) et vue « Tableau » (les colonnes de l'écran) |
| Graphique de diffusion (sélecteur de sondage) et sa légende | vue « Diffusion » du sondage dessiné, la légende de l'écran sous le graphique |
| k en fonction de B et sa légende | vue « k(B) », la légende de l'écran sous le graphique |
| Courbes CPT et classification | vue « CPT » (si le sondage a des mesures) |
| Détail par sondage (tassement, q_net, z d'influence, k, arrêt, formules, tranches, h sans module) | fenêtre de calcul de chaque sondage (positions, formules substituées, tranches, arrêt) et fenêtre « Zonage » |
| Modes 3 à 6 : champs, bloc, formules, tableaux, avertissements, légendes | section 2 du mode, verdict, lignes, fenêtre de calcul ; vues « q = k·w », « Comparaison » et « Diffusion » (sol homogène), « Graphique » et « Tableau » de l'abaque, « Coupe » du plat |
| Fiche mémo, pied de page | section « Fiche mémo », messages du résultat |
| 💾 Enregistrer / 📂 Ouvrir (raideur_sol.json) | le fichier .stab du site ; « Reprendre un fichier de l'écran Streamlit » relit raideur_sol.json |
| Nouveaux sur le site (dessins seulement) | vue « Coupe » (semelle sur le profil, à l'échelle, profondeur d'influence) et vue « Sondages » (côte à côte) |

**Non couvert sur le site** : l'import d'un PDF à courbes vectorielles (PyMuPDF n'existe pas dans le navigateur : le
bouton accepte GEF, CSV, TXT et renvoie le PDF vers l'écran Streamlit) ; la note PDF (hors périmètre du site, bouton
grisé) ; le champ « Nom du projet » (seulement repris dans la note PDF ; le site a le nom de la note) ; la case « Détail des
calculs » (le site montre toujours le détail, dans la fenêtre de calcul) ; « Réinitialiser » (le site crée une note neuve) ;
l'enregistrement AU FORMAT de l'écran (le site relit raideur_sol.json mais enregistre son .stab). Pas de sélecteur
trié ✓/★ : ce module n'a aucun choix dans un catalogue à dimensionner (le type de sol est une description du terrain).

## 5 · Défauts constatés dans l'écran Streamlit (reproduits par le moteur, écrits ici)

1. **Le découpage d'un essai PLANTE** (TypeError dans `sol_theorie.profil_depuis_cpt`) quand les dernières mesures, sur
   moins que l'épaisseur minimale, n'ont pas d'indice Ic (frottement nul ou absent, capteur muet au refus) : l'écran
   affiche une exception. Le moteur ne corrige pas le calcul : il rend un message (« réduire l'épaisseur minimale, ou
   retirer ces dernières lignes »), le site l'affiche et trace qc et fs sans Ic.
2. **L'export « CSV des couches » PLANTE** dès qu'une couche découpée n'a pas de module (pas d'Ic : un essai sans fs)
   (`sol_import.couches_vers_csv` : `{None:.1f}`) ; l'exception s'affiche sous le bouton « Remplir le tableau ». Le site
   n'offre pas ce lien et le dit.
3. **« 💾 Enregistrer » n'enregistre pas les modes 3 à 6** : leurs clés (`rs3_*`, `rs4_*`, `rs5_B`, `rs6_*`) ne commencent
   par aucun préfixe du module (`snd`, `rs_`, `sol_`) ; le fichier rouvert rend les valeurs initiales (seul `rs_w_ref`
   survit). Même cause : « 🧹 Réinitialiser » ne les remet pas à zéro.
4. **Le fichier n'enregistre pas les mesures brutes d'un CPT** (une liste, pas un scalaire) : rouvert, le sondage perd ses
   courbes CPT et l'export des mesures.
5. **« Profil trop court — k surestimé » quand l'arrêt est le plafond 6·B** : le message dit que « les couches saisies
   s'arrêtent là » alors que le profil est plus long ; le site reprend le message et ajoute une précision.
6. **Le graphique k(B) ignore l'option « pression nette »** (toujours nette) et intègre par tranches de 10 cm (5 cm pour
   le résultat) : avec la pression brute, la courbe ne passe pas par le k affiché.
7. **Les courbes CPT de l'écran découpent avec les réglages par défaut** (finesse 0,25, épaisseur 0,50 m, ν 0,30), pas ceux
   de l'aperçu d'import : l'Ic tracé peut différer des couches du tableau.
8. **L'abaque n'applique pas la correction de forme** de Terzaghi (L n'est pas transmis à `k_terzaghi_taille`).
9. **Mode 4 : B et L ne sont pas échangés** si L < B (seulement dans les modes 1 et 2) ; les formules élastiques n'utilisent
   que B.
10. **Deux sondages du même nom** : le menu du graphique choisit par NOM, l'écran ne trace que le premier (le site choisit
    par position).
11. **La nappe proposée à l'import** est celle de la fondation même quand la nappe n'y est pas cochée.
12. **Les champs ramènent EN SILENCE une saisie hors bornes à la borne** (min_value, max_value des widgets : B ≥ 0,2 m,
    ν ≤ 0,49, γ de 8 à 26, finesse 0,10 à 0,60…) : le site REFUSE la saisie avec un message qui nomme le champ, la valeur et
    la borne (`rigidite.BORNES`, lues dans les widgets et contrôlées par le test).
13. **Le conseil « descendre à 2,5·B m »** du message « Profil trop court » ne dit pas qu'il se compte sous l'assise
    (défaut non conservatif n° 1, en tête de ce document) ; le site ajoute « D + 2,5·B = … m sous le TN »
    (`courts.profondeur_conseillee_tn`, calculé par le moteur).
14. **Les motifs d'arrêt sont écrits à l'anglaise** (« profil incomplet : 2.00 m sans module M », « Δσ ≤ 20% ») : le site
    les écrit à la française (« 2,00 m », « 20 % »).
15. **Aucune borne haute** sur la plupart des champs (q, M, B, w…) : 1e308 y donne « inf » ou une exception (B = L = q =
    1e308 : division par zéro). Le site refuse toute donnée au-delà de 10⁶ dans l'unité du champ
    (`rigidite.VRAISEMBLANCE`, « hors de toute vraisemblance »).
16. **« 📂 Ouvrir » accepte en silence un fichier mal formé** : un ordre de couches « abc » devient trois couches par défaut
    (h = 1 m, M = 0) ; un sondage sans identifiant lève une exception Python. Le moteur du site les refuse avec un message
    en français (« ordre des couches du sondage 1 illisible », « sondage illisible dans le fichier »).

## 6 · Relecture indépendante (07/11/2026)

Une relecture par un agent qui n'avait pas construit le module : chiffres du site contre l'écran rejoué (27 cas variés
à la main, puis les 93 fondations du test, § 3.11), couverture ligne à ligne de l'écran, captures regardées (PC,
tablette, téléphone, chaque vue, l'import, deux sondages, une assise profonde sur un profil court, les quatre autres
méthodes), saisies absurdes et notes abîmées. Corrigé :

- **chiffres** : aucun écart de valeur ; les motifs d'arrêt à l'anglaise « 2.00 m » (défaut 14) ; le double
  échappement guetté (aucun) ;
- **non conservatif** : la précision « D + 2,5·B » (défaut 13), le message « Profil trop court » posé sous le verdict
  avant les lignes, la note « À compléter » quand un profil est épuisé ;
- **robustesse** : plafond de vraisemblance 10⁶ (OverflowError et ZeroDivisionError rendues jusque-là par le
  routeur), mesures CPT non finies refusées dans une note, fichier Streamlit mal formé refusé en français ;
- **couverture** : les légendes de l'écran sous les graphiques de diffusion et k(B) (elles manquaient), « k dépend de la
  fondation », les aides de M et de γ, la position de référence SCIA sortie des options repliées ;
- **écran** : libellés du menu Méthode qui ne se coupent plus, unités une seule fois dans les cartes de couches,
  indices écrits en indice dans les dessins (q<sub>net</sub>, h<sub>c</sub>, E<sub>c</sub>, t<sub>g</sub>, σ'<sub>v0</sub>, et
  non « q_net »), espace fine insécable des milliers dans les dessins et tableaux, repères « assise » et « nappe » de la
  vue « Sondages » qui ne chevauchent plus les colonnes, pied sur une ligne (méthode · moteur · v5.0), l'avertissement
  de pré-dimensionnement une seule fois.

## 7 · Besoins communs relevés

- Une **ligne de résultat sans taux** (valeur et plage, sur une barre commune) et un **verdict de valeur** (k en grand) pour
  les modules de consultation : `C.ligne_res` et `C.chip` ne savent écrire qu'un taux (ici `_ligne_k` et `_verdict`, mêmes
  classes que les briques).
- Une **fenêtre de calcul à colonnes libres** : `C.fenetre_calcul` impose « Vérification, Formule, Valeur, Limite, Taux » ;
  le module remplace son tableau vide par le sien (contrôlé par une assertion).
- Un **bouton de téléchargement** (exports CSV) : le module pose un lien `download` en `data:` (permis par la politique de
  sécurité, aucune ressource externe).
- La mention « mm · à l'échelle » de la feuille ne convient pas aux graphiques : le module la masque dans sa feuille de style.
- Au téléphone, l'établi n'a pas encore d'onglets Données / Dessin / Résultat (le `data-tab` du banc est fixe).
- Un champ dans le cadre Résultat (la position de référence SCIA est un choix de LECTURE du résultat, comme le radio de
  l'écran Streamlit) : l'établi ne relie que les champs du cadre Données.
- Sur tablette, la toile est étroite : les graphiques (700 unités de large) y sont lus à environ 0,6 ; un zoom de la toile
  servirait tous les modules à graphiques.
