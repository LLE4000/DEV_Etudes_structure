# Charges sur plancher — le module du site « Stab Méthodes »

Le module `plancher` du site (`api_web/plancher.py`, établi `charges-plancher`) donne au navigateur ce que l'écran
Streamlit `modules/charges_plancher.py` offre : les mêmes données, le même calcul, les mêmes chiffres. L'écran Streamlit
reste la référence et le secours ; il n'est pas modifié.

> **Défaut NON CONSERVATIF du moteur, présent dans l'écran Streamlit (trouvé à la relecture du 07/11/2026).** Une part
> du tranchant P-3 < 1 (section 6, « part du tranchant ») **sans répartition active** (hourdis posés, ou parts de
> l'annexe C absentes) fait disparaître `(1 − part) × P` de chaque charge concentrée : aucun élément ne la porte, le
> tranchant est sous-estimé (étançon de 100 kN, part 0,7, hourdis posés : Σ réactions = 70 kN). Par défaut la part vaut
> 1 et le défaut est inactif. Le site **reproduit** le calcul (parité) et le **dit** en rouge dans le résultat et dans la
> fenêtre de l'élément ; la correction appartient au moteur (voir `CHARGES_PLANCHER.md`, en tête).

## 1. D'où vient chaque grandeur

Le moteur existait déjà sans Streamlit (`plancher/charges`, voir `CHARGES_PLANCHER.md`) : le site l'importe **tel quel**,
sans `note.py` ni `peintre.py` (reportlab). L'état d'une note du site **est** l'état du moteur (`etat.py` : cas multiples,
fiches ajoutées), plus deux petites clés propres au site : `site` (les brouillons des formulaires « nouvelle fiche ») et,
dans chaque cas, `site` (l'engin balayé, la dernière proposition et son empreinte).

L'enchaînement du calcul est exactement celui de `show()` : `etat.construire` → `verification.verifier`
(`api_web.plancher._calculer`). Les dessins sont ceux du moteur (`dessin.plan`, `dessin.coupe`, `dessin.balayage`,
`symboles.pictogramme`), habillés par `api_web/dessin_plancher.py` (voir § 4).

Ce que l'écran Streamlit faisait LUI-MÊME (dans sa vue) a été **extrait mot pour mot** dans
`plancher/charges/operations.py` :

| Opération | Fonction de l'écran d'origine | Fonction extraite |
|---|---|---|
| proposer une filière sous les appuis en défaut | `_proposer` | `proposer_filiere`, `groupes_en_defaut`, `phrase_proposition` |
| appliquer la proposition | bouton de `_carte_resultat` | `appliquer_proposition` |
| barres de la carte Résultat, lignes non vérifiées | `_carte_resultat` | `barres`, `non_verifies` |
| résumé de l'étape Plancher | `_resume_plancher` | `resume_plancher` |
| total de l'étape Charges, tableau « Appuis de E1 » | `_etape_charges` | `total_charges`, `appuis_engin` |
| balayage et sa phrase | `_balayage` | `balayage`, `legende_balayage` |
| champ « Parts annexe C » | `_parts_annexe_c` | `lire_parts_annexe_c`, `texte_parts_annexe_c`, `morceaux_ignores` |
| « Nouvelle fiche d'engin » | `_dialogue_bibliotheque` | `fiche_engin` (+ défauts `APPUIS_FICHE_DEFAUT`, `FICHE_ENGIN_DEFAUT`) |
| « Nouvel élément de répartition » | `_nouvel_element` | `fiche_repartiteur` (+ `FY_MATERIAU`, `RECTANGLE`) |

`api_web/plancher.py` **met en forme** : il ne calcule rien. Les fenêtres de calcul écrivent les formules avec les valeurs
que le moteur a nommées (`elements[…]` : `M_G`, `M_Q`, `M_max`, `x_M`, `V_A`, `V_B`, `w_g`, `w_q`, `charges` ;
`resistances.resistance_dalle`, `resistance_hourdis`, `pression_localisee_Rd`, `poinconnement` ; les extras des lignes de
`verifications`). Deux grandeurs intermédiaires sont des **projections d'affichage**, recalculées depuis le moteur et
vérifiées par les tests : Σ M_P (somme des `diffusion.moment_ponctuel` des charges de l'élément) et Σ R_A, Σ R_B (lues
dans `V_A`, `V_B`). Les sélecteurs (répartiteur d'un appui, profilé d'une filière) relancent le moteur sur une copie de
la note pour chaque ligne.

## 2. La preuve de parité

`tests/test_charges_plancher_parite.py` **exécute l'écran Streamlit** (`show()`, `_carte_resultat`, `_proposer`,
`_etape_charges`, `_balayage`, `_parts_annexe_c`, `_nouvel_element`, `_dialogue_bibliotheque`) avec un faux module
d'écran qui enregistre ce qu'il afficherait, et un espion sur `verification.verifier` et `verification.balayer` :

1. **Calcul** : plus de 120 notes tirées au hasard (graine fixe : hourdis, dalle, prédalle ; natures Rd et service ;
   chapes, annexe C, plateaux, filières, étançons, files d'étais, stocks, engins tournés…) et des cas choisis. Le résultat
   que l'écran obtient du moteur est comparé à celui du site **au bit près** (texte JSON identique, flottants compris) ;
   les refus et les plantages de l'écran aussi.
2. **Ce que l'écran compose** : résumé, total, barres, compte des lignes non vérifiées, tableau des appuis (charges au bit
   près : le format de l'écran est remplacé par `repr` le temps du contrôle), balayage (enveloppe au bit près, phrase),
   proposition de filière (au bit près), son texte, « Appliquer ».
3. **Formulaires** : douze saisies des parts annexe C (valeurs et exceptions), six fiches de répartiteur, trois fiches
   d'engin : la fiche que l'écran ajoute est celle d'`operations.py`.
4. **Défauts** de l'écran (§ 6) : reproduits par le moteur, refusés ou rendus lisibles par le site.
5. **Mutations** : dix copies altérées d'`operations.py` (seuil, arrondi, ordre des candidats, table des nuances, barres,
   total, résumé, lecture des parts, bande d'un appui, condition de proposition) font rougir la parité.

`tests/test_api_web_plancher.py` prouve que **l'écran du site affiche ce que le moteur calcule** (verdict, une ligne par
famille, tableau des appuis, formules substituées qui se recalculent), chaque saisie et chaque refus, les boutons
(engins, charges, plateaux, filières, « idem », proposition, fiches, fichier Streamlit), les sélecteurs (tri, ✓, ★,
choix), les vues, l'échappement, les cas multiples, la relecture des fichiers, et la **pureté** : le module se charge et
calcule sous un `sys.meta_path` qui refuse streamlit, pandas, matplotlib, reportlab et scipy (numpy seul admis, comme
dans Pyodide).

## 3. L'écran Streamlit, champ par champ, et où le trouver sur le site

| Écran Streamlit | Site |
|---|---|
| sélecteur de cas, ＋ Cas, ⧉, 🗑️, nom du cas | liste « Cas de cette note » (ouvrir, ajouter, copier, supprimer) ; section 1 « Nom » |
| 💾 Enregistrer (JSON) | le fichier `.stab` du site porte toute la note (cas et fiches) — voir § 5 |
| 📂 Ouvrir (JSON) | section 7 « Ouvrir un fichier JSON du module » (même format que l'écran) |
| 📄 Note PDF | hors périmètre (bouton grisé du site) |
| vue Plan / Coupe / Balayage, cases Cotes, Aires efficaces, Taux par élément | vues « Plan », « Coupe », « Balayage » de la toile ; cases en section 7 |
| plan cliquable : glisser un engin, une charge, un plateau, une filière | champs x, y (rotation) de chaque charge (section 3), décalage Δx, Δy de chaque plateau, x, y de chaque filière (section 4) |
| cote L cliquable, poignée du sens porteur, ± hourdis | section 1 (portée, appuis A–B), section 2 (nombre d'éléments) |
| fenêtre d'un engin : angle, charge par appui, dupliquer, supprimer | carte de l'engin (section 3) : rotation, charge de chaque appui, boutons copier et retirer |
| fenêtre d'une charge : P, a, b / p, longueur, orientation / q, longueur, largeur | carte de la charge (section 3) |
| fenêtre d'une semelle : type, plateau, idem | ligne de l'appui (section 3) : menu du répartiteur, bouton « idem » ; ou sélecteur du résultat (taux de chaque choix, ✓, ★) |
| fenêtre d'une filière : élément, longueur, orientation, supprimer | carte de la filière (section 4) ; sélecteur de profilé (✓, ★) |
| coupe sous l'appui sélectionné | vue « Coupe » ; « Coupe sous » (section 7) |
| carte Résultat : pastille, phrase, barres par famille, lignes non vérifiées | verdict (taux, vérification gouvernante « valeur ≤ limite »), une ligne par famille, messages |
| 💡 Proposer une filière, ↔ Balayer la position, proposition, « Appliquer » | section 4 (ouverte quand le cas est non conforme) ; vue « Balayage » ; message du résultat |
| étape 1 Plancher (type, L, appuis, q ; hourdis : type, éléments, M_Rd, V_Rd, I, p_adm, h, nature, g, état, chape, M_cr, épaisseur, fck(t), valeurs avec chape, collaborante, classe exigée, parts annexe C, source, garder comme fiche ; dalle/prédalle : h, d, As, fyk, γs, béton, fck(t), h1, largeur, armature transversale, largeur de prédalle, joints armés, chape durcie, étais) | sections 1 et 2 (les mêmes champs, avec leurs aides P-n) |
| étape 2 Charges (résumé, bibliothèque, étançon, file d'étais, stock, liste, appuis de l'engin) | section 3 (cartes) ; section 5 (bibliothèque avec vignettes, « Poser ») ; groupe « Appuis de E1 » du résultat |
| « Nouvelle fiche d'engin » (nom, type d'appuis, châssis, flèche, tableau d'appuis, source) | section 5 (tableau d'appuis éditable, ajouter/retirer un appui) |
| étape 3 Répartiteur (appui, type, plateau, décalages, idem, filière, autres appuis portés, nouvel élément) | lignes des appuis (section 3), cartes des filières avec « appuis portés » (section 4), section 5 « Nouvel élément de répartition » |
| étape 4 Hypothèses (γG, γQ, part du tranchant, k Heft 240, joint, αcc, phrase et hypothèses du moteur) | section 6 (mêmes champs, mêmes aides, même phrase, mêmes hypothèses) |
| étape 5 Méthode (identification, sections, références, à transcrire, bibliothèque et sources) | section 7 (identification) ; vues « Méthode » et « Bibliothèque » (textes PARTAGÉS : `plancher/charges/methode.py`) |
| détail des vérifications par élément | vue « Détail » (par élément, puis par famille avec la méthode écrite une fois) ; fenêtres de calcul |

Le site ajoute : les **fenêtres de calcul** (un élément de hourdis : charges réparties, charges concentrées portées et
leur moment, M_G, M_Q, M_Ed = γG·M_G + γQ·M_Q, V_A, V_B avec la part P-3, fissuration ; la dalle : résistances par mètre,
largeurs efficaces, moment et tranchant ; les effets locaux : membrure, pression localisée, poinçonnement ; les plateaux :
bande c et aire efficace ; les filières : réactions et flexion ; les joints), et les **sélecteurs** triés par masse avec
le taux de chaque choix (éléments chargés et effet local sous l'appui ; filière et éléments qu'elle charge), le retenu ✓
et le plus léger qui tient ★.

## 4. Les dessins

Les dessins du moteur sont à l'échelle (mm). Le site les rend en mode « impression » (sans poignées), retire les
attributs d'interaction du composant cliquable, préfixe les identifiants internes par vue, **échappe tout texte venu de
la note** avant le dessin (le moteur écrit les noms tels quels) et passe au code couleur du bureau : l'orange de l'écran
d'origine (seuil d'attention 85 %, `Parametres.seuil_attention`) devient le vert foncé du site (95–100 %, `C.niveau`).
**Différence d'affichage, pas de calcul** : entre 85 et 95 %, l'écran Streamlit écrit « Conforme, attention » (orange), le
site « Conforme » (vert pâle).

## 5. Ce qui n'est pas couvert

- La note PDF (deux pages) : hors périmètre du site, elle reste dans Streamlit. *Révision du 09/11/2026* : elle se fait
  désormais sur le site, par le générateur validé inchangé, deux pages par cas (`api_web/note_cas.py`,
  docs/site/MODULES_SITE.md § 4 quinquies) ; le bouton « Note PDF » est allumé aux trois formats.
- Télécharger le fichier JSON au format de l'écran Streamlit : le site n'a pas de brique « fichier à télécharger » pour un
  module (besoin commun). Le `.stab` du site porte la note entière ; le site **ouvre** le JSON de l'écran Streamlit.
- Le glisser-déposer sur le plan : remplacé par la saisie des positions (demande du bureau). Les invites et les « toasts »
  de l'écran d'origine n'ont pas d'équivalent (le site recalcule à chaque saisie).

## 6. Défauts trouvés dans l'écran Streamlit

Reproduits par le moteur (on extrait, on ne réécrit pas), prouvés par `tests/test_charges_plancher_parite.py`, et traités
par le site comme indiqué :

1. **« Parts annexe C » à la virgule décimale** : « 0:0,5 » est coupé à la virgule et lu `{0: 0.0}` (le « 5 » est ignoré
   en silence) ; des parts nulles provoquent ensuite une **division par zéro** (`diffusion.parts_voisins`) et l'écran
   s'arrête sur une exception. Le site **refuse** la saisie (« écrire 0:0.5 ; 1:0.25 ») et, si un fichier apporte des
   parts nulles, affiche « Calcul impossible » au lieu de planter.
2. **Plateau plus étroit que l'empreinte** (une fiche ajoutée, par exemple un madrier 600 × 200 sous un pied 300 × 300) :
   `repartiteurs.aire_efficace` lève `ValueError` (et non un refus du moteur) ; l'écran Streamlit **s'arrête**. Le site
   écrit « Calcul impossible : Empreinte plus grande que le plateau. ».
3. **Seconde proposition de filière** : l'identifiant d'une filière déjà appliquée (« F1p0 ») est repris (`ref_libre`
   ne connaît que « F1 ») ; deux filières portent le même identifiant et la première devient fantôme. Le site applique
   la même proposition (mêmes profilés, mêmes positions) avec un identifiant libre.
4. **Total de l'étape Charges** : « … kN au total » ne compte que les charges par appui des engins et les étançons ; les
   files d'étais (kN/m) et les stocks (kN/m²) n'y entrent pas. Reproduit (`total_charges`) ; le site l'intitule
   « Engins et étançons (charges de service) ».
5. **Champs sans bornes** : hors L ≥ 0,5 m, n ≥ 1, q ≥ 0, M_Rd ≥ 0, V_Rd ≥ 0, As ≥ 0, h1 ≥ 0 et largeur ≥ 0,5 m, les
   champs numériques de l'écran acceptent des valeurs nulles ou négatives (épaisseurs, largeur d'élément, γ, part du
   tranchant > 1…) qui passent au moteur. Le site les **refuse** avec leur raison et laisse la note intacte ; il refuse
   aussi un fck(t) sous la plus petite classe (l'écran s'arrêtait alors sur « Donnée manquante »).
6. **Observation sur les défauts** : la dalle par défaut de l'écran (h = 200 mm, As = 524 mm²/m, C30/37, L = 7,20 m)
   dépasse M_adm sous son seul poids propre et les charges présentes (111 %) avec la méthode de l'application ; ce n'est
   pas un défaut de calcul, mais la note neuve « Dalle BA » est non conforme avant toute charge.
7. **Part du tranchant P-3 < 1 sans répartition active — NON CONSERVATIF** (relecture du 07/11/2026) : voir l'encadré en
   tête. Reproduit, et dit en rouge par le site (`_alerte_p3`, `_texte_p3`).
8. **Hauteur utile d ≥ épaisseur h** (dalle, prédalle) : l'écran l'accepte, et M_adm,b = α_b·μ·b·d² grandit avec d —
   une saisie absurde qui donnerait un résultat optimiste. Le site la **refuse** (`_coherence`).
9. **Aucune borne haute** : une portée de 1 000 m, 10 000 éléments ou un stock découpé en dizaines de milliers de charges
   de 0,30 m bloquent l'écran (le balayage refait le calcul tous les 10 cm de la portée). Le site refuse au-delà de
   bornes larges (`LIMITES` : L ≤ 40 m, 100 éléments, largeur ≤ 100 m, γ ≤ 5, stock ≤ 2 500 charges élémentaires…), à
   la saisie comme à la relecture d'un fichier.

## 6 bis. La relecture indépendante (07/11/2026)

Ce qu'elle a vérifié : **les chiffres affichés** (pas seulement ceux du moteur) — verdict, ligne de chaque famille,
tableau du détail par élément, E, R, η de chaque vérification, compte des lignes non vérifiées, statut — sur 20 cas
choisis (hourdis Rd et service, chape collaborante, sens 2 et engin tourné, joints vérifiés, part du tranchant 0,7,
charge énorme, filière dans le sens porteur, dalle avec k du Heft 240, fck(t) et revêtement, B400 sans armature
transversale, prédalle étayée, sans charge, dalle par défaut, pile à 100 %, à 100,15 %, à 94,96 %, hourdis posés,
répartiteur exigé) et 40 notes tirées au hasard, l'écran Streamlit rejoué par un faux `st` : **aucun écart** hors des
différences voulues (arrondi qui garde la bande de couleur ; seuil d'attention 95 % au lieu de 85 % ; verdict qui nomme
la ligne « répartiteur exigé » quand c'est elle qui rend le cas non conforme sous 100 %, là où l'écran d'origine nomme le
plus grand taux).

Ce qu'elle a corrigé sur le site : le défaut non conservatif P-3 dit en rouge ; « Conforme, attention » quand une
hypothèse du moteur l'est (P-8, étais en place) comme dans l'écran d'origine ; le message d'un refus écrit une fois (le
verdict y renvoie) avec « Où compléter » (la section du site qui règle ce que le moteur demande) ; les fenêtres de calcul
(une charge découpée — stock, file d'étais — ou un appui à cheval sur deux éléments sur UNE ligne, avec la somme Σ M_P
que M_Q reprend ; P-3 expliqué selon la part et la répartition ; F_Rdu, v_Rd,c, u_0, v_Rd,max écrits avec leurs valeurs ;
contraintes sous 1 N/mm² à trois décimales, mm² entiers) ; les textes du moteur à la française partout (« y = 3,74 m »,
« 100 % ») ; le nom de l'engin qui chevauchait les étiquettes de ses appuis sur le plan ; la légende des symboles du plan
(semelle, plateau, aire efficace, filière, couleurs), que l'écran d'origine écrit sous son plan ; la charge totale de
chaque engin dans sa carte ; les menus de répartiteur dont le texte passait sous la flèche ; les bornes hautes et d ≥ h ;
une proposition de filière qui disparaissait (et ne s'appliquait plus) après un aller-retour par le navigateur — `JSON.stringify` écrit 45.0 « 45 » et l'empreinte de la proposition changeait (`_canon`).
Les contrôles qui le tiennent : `tests/test_api_web_plancher.py` § 6 bis.

## 7. Se vérifier

```bash
python3.12 tests/test_api_web_modules.py plancher       # le contrat commun
python3.12 tests/test_api_web_plancher.py               # le module du site
/usr/bin/python3.12 tests/test_charges_plancher_parite.py   # la parité avec l'écran Streamlit (streamlit, pandas requis)
python3.12 site/web/construire.py --dossier /tmp/site_plancher --modules plancher
python3.12 site/web/outils/serveur_local.py 8813 --dossier /tmp/site_plancher &
node site/web/verifier_site.js http://127.0.0.1:8813 /tmp/captures_plancher module:plancher
```
