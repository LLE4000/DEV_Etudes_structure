# Charges sur plancher — document de référence (moteur, lot 1 · écran, lot 2)

*Lot 1 livré le 02/11/2026 : le moteur, sans écran. Lot 2 livré le 02/11/2026 :
l'écran (section 8). Étude de conception : `CHARGES_PLANCHER_ETUDE.md`
(décisions P-1 à P-17, section 12 pour la revue des remarques extérieures).
Ce document dit ce que le moteur calcule, avec quels paramètres, ce qu'il
refuse, ce qui reste à transcrire, comment il est tenu par ses tests, et
comment l'écran s'en sert.*

## 1 · Ce que le paquet contient

```
plancher/charges/
  modele.py         Hourdis · DalleBA · Predalle (éléments, raideur k = 3·E·I·L/(a²(L−a)²), g_total)
                    Empreinte · ChargeLineaire · ChargeSurfacique · AppuiGabarit · Gabarit · Engin
                    Plateau · Filiere · Chape (P-17)
  diffusion.py      Heft 240 : t_y, b_v, b_m · lignes d'influence · règle du levier · parts annexe C fournies · créneaux
  repartiteurs.py   bande efficace c (EN 1993-1-8 §6.2.5) · section en I (repli) · poutre sur ressorts (numpy) · flexion de la filière
  resistances.py    formules BA de l'application inversées · natures de catalogue · chape · EN 1992-1-1 §6.7 et §6.4
  verification.py   verifier() · balayer() · dimensionner_filiere()
  bibliotheque.py   gabarits, types de hourdis, plateaux (plancher/donnees/*.json) · profilés (profiles_test.json) · aciers
  parametres.py     Parametres : décisions du bureau et valeurs « à transcrire », défauts conservatifs
plancher/donnees/   engins.json · hourdis.json · plateaux.json (génériques, sources, charges à saisir)
tests/              test_charges_plancher_diffusion.py · _repartiteurs.py · _moteur.py · ref_chp.json (cas gelés)
```

Le paquet est **pur** : ni Streamlit, ni dessin, ni note (test 10 du moteur).
Dépendance : numpy, déjà présent dans l'application.

## 2 · Appel

```python
from plancher.charges import modele as M, verification as V, bibliotheque as BIB
from plancher.charges.parametres import Parametres

h = M.Hourdis(nom="SP 200", L=7.2, n_elements=8, I=1.05e-3, M_Rd=90, V_Rd=95, nature=M.RD,
              p_loc_adm=0.70, q_presentes=1.0, etat=M.CLAVETES, parts_annexe_C=None)
gab = BIB.gabarits()["Télescopique rotatif (type MRT 2150) · levage sur stabilisateurs"]
for a in gab.appuis: a.P = 60.0                      # tableau de stabilité, par opération
engin = M.Engin("E1", gab, x=3.0, y=4.8, angle=0)
plateau = BIB.plateaux()["Tôle acier S235 · 600 × 600 × 20"]
res = V.verifier(h, [engin], {e.ref: plateau for e in engin.empreintes()}, Parametres())
res["statut"], res["taux_max"], res["gouverne"], res["elements"], res["verifications"]
```

`verifier` ne lève jamais : un refus est un résultat `{"statut": "refus", "message": …}`.
`balayer(plancher, charges, repartiteurs, ref_engin)` déplace l'engin le long de la portée.
`dimensionner_filiere(plancher, charges, repartiteurs, groupes, candidats, fabrique)` parcourt
les profilés par masse croissante, une filière en travers par groupe de pieds, et retient le
premier candidat dont **tout le système** est ≤ 1 (P-14), avec la liste des taux qui ont monté.

## 3 · Ce que le moteur calcule

**Charges.** Empreintes de service `P` avec coefficient dynamique `φ` par appui (P-6) ; charges
linéaires et surfaciques discrétisées (pas 0,25 / 0,30 m) ; gabarits d'engin tournés de 0 à 270°.
Un gabarit sans charge par appui est un refus (P-13).

**Répartiteurs.** Plateau : aire efficace (a + 2c)(b + 2c), c = t·√(f/(3·p_Ed)) itéré avec la pression
ELU p_Ed = γQ·P / aire (vérification plastique de l'acier), c plafonné par le bord du plateau ; plaque
déclarée rigide : plaque entière, pression de service comparée à la valeur publiée. Filière en travers :
poutre d'Euler libre sur un ressort par élément croisé, k = 3·E·I·L/(a²(L − a)²) à l'abscisse de la
filière ; réactions = charges des éléments ; flexion de la filière à l'ELU, γQ·M ≤ W·fy/γM0 ; un
ressort qui tire est signalé. Filière dans le sens porteur : charge linéaire uniforme (hypothèse
rigide), aucune répartition entre éléments.

**Hourdis.** Par élément : règle du levier (part proportionnelle au recouvrement de l'empreinte),
parts de l'annexe C si elles sont fournies et si les joints sont clavetés ou le plancher chapé
(flexion seulement), tranchant à `part_tranchant_element` sur l'élément chargé (P-3, défaut 100 %) ;
M aux abscisses des charges et à mi-portée, V aux appuis ; les efforts sollicitants comptent les
**poids propres** (`g + chape`, part G) et les charges de chantier (`q_presentes` et charges
concentrées × φ, part Q). **Résistances de calcul** (nature Rd) : M_Ed = γG·M_G + γQ·M_Q ≤ M_Rd avec
γG = 1,35 et γQ = 1,5 (P-2, tranché le 02/11/2026 — plus de « ÷ 1,5 uniforme », qui pondérait le poids
propre à 1,5) ; **valeurs admissibles de service** : M_G + M_Q ≤ M_adm, coefficients 1 ; valeurs
« avec chape » seulement si collaborante et classe exigée atteinte (P-17). La ligne de fissuration
(M_cr) compare le moment de service. Membrure : pression P/(aire efficace
+ diffusion à 45° dans la chape) ≤ `p_loc_adm` ; sans donnée : non vérifié si un répartiteur efficace
est présent, défaut « répartiteur exigé » sinon (P-5). Joints : effort transmis par les parts de
l'annexe C, vérifié si `joint_adm_kN_m` et `joint_longueur_m` sont renseignés (P-15). Fissuration :
information si M_cr est renseigné (P-11).

**Dalle et prédalle.** Par mètre : t_y = b + 2·h1 + h ≤ 0,8·l ; b_v = t_y + 0,5·a ; b_m = t_y tant que
`k_flexion_heft240` est `None` (P-4, **très conservatif** sous charge concentrée, voir § 5) ;
superposition en créneaux, maximum sur la largeur ; résistances = formules de `poutre.py` inversées
(§ 2.1 de l'étude) sur la classe saisie ou la classe arrondie vers le bas de fck(t). Effets locaux
EN 1992-1-1 §6.7 et §6.4 à l'ELU avec γQ sur la charge concentrée (P-9) ; la flexion et le tranchant
par mètre restent en service (coefficients 1), méthode de l'application. Prédalle : largeur plafonnée à la prédalle si
joints non armés ou charge à cheval sur un joint (P-16) ; étais en place = portée réduite +
avertissement (P-8) ; chape non durcie = refus de toute charge d'engin ou d'étai (P-7).

**Statuts.** `ok`, `attention` (≥ 0,85), `nok` (> 1), `info`, `non_verifie` (paramètre à transcrire
absent), `refus`. Le statut global est `nok` dès qu'une ligne l'est ; `non_verifie` ne masque rien :
la liste `non_verifies` est rendue à part.

## 4 · Paramètres (`parametres.Parametres`)

| Champ | Défaut | Décision |
|---|---|---|
| `gamma_G`, `gamma_Q` | 1,35 · 1,5 | P-2 (02/11/2026) : ELU quand la résistance est une valeur de calcul — γG sur les poids propres, γQ sur les charges de chantier (hourdis Rd, effets locaux, filière, bande plastique du plateau) ; 1 et 1 avec une valeur admissible de service |
| `part_tranchant_element` | 1,0 | P-3 : part de la réaction d'appui d'une charge concentrée reprise en tranchant par l'élément chargé ; 1 = aucun partage par les joints (prudent : l'annexe C répartit la flexion, pas le tranchant près de l'appui) |
| `k_flexion_heft240` | `None` → b_m = t_y | P-4, **à transcrire** du Heft 240 |
| `joint_adm_kN_m`, `joint_longueur_m` | `None` → non vérifié | P-15, **à transcrire** de l'EN 1168 |
| `gamma_c`, `alpha_cc`, `beta_poinconnement` | 1,5 · 1,0 · 1,0 | P-9 ; αcc de l'ANB belge à confirmer |
| `coef_vrd_max` | 0,4 (ν·fcd) | §6.4.5(3), coefficient à confirmer sur l'ANB ; `None` = contrôle u0 non fait |
| `seuil_attention`, `seuil_defaut` | 0,85 · 1,0 | affichage |
| `pas_lineaire`, `pas_surfacique`, `pas_balayage` | 0,25 · 0,30 · 0,10 m | discrétisation |

Les parts de l'annexe C ne sont pas un paramètre global mais une donnée du hourdis
(`parts_annexe_C = {0: part de l'élément chargé, 1: chaque premier voisin, …}`), fournies par le
bureau après transcription de la figure C.1 ; `None` = l'élément chargé porte tout.

## 5 · Ce qui reste à transcrire avant un usage en production

1. **k du Heft 240** (largeur efficace en flexion) : sans lui, une charge concentrée sur une dalle
   est jugée sur t_y seul, ce qui multiplie le moment par mètre par un facteur de l'ordre de 3 à 5 par
   rapport à la largeur efficace réelle. Le moteur le dit dans une ligne d'hypothèse.
2. **Parts de la figure C.1 de l'EN 1168** : sans elles, aucun partage entre hourdis clavetés
   (conservatif pour l'élément chargé).
3. **Résistance du joint longitudinal** (EN 1168) : sans elle, l'effort transmis est calculé et la
   ligne est « non vérifié ».
4. **αcc et v_Rd,max** de l'ANB belge pour les effets locaux.
5. **Valeurs de catalogue** des hourdis (M_Rd, V_Rd, nature, I, p_loc_adm, M_cr, valeurs avec chape et
   classe exigée) : la fiche livrée est un gabarit vide.

## 6 · Tests

- `test_charges_plancher_diffusion.py` (33) : Heft 240, lignes d'influence, levier, parts annexe C,
  créneaux.
- `test_charges_plancher_repartiteurs.py` (23) : bande c itérée à la main, section en I à 2 % du
  catalogue sur tous les profilés, poutre sur ressorts (statique, rigide, souple, Σ R = Σ P, symétrie,
  erreurs), flexion de la filière.
- `test_charges_plancher_moteur.py` (82) : **copie tenue** des formules BA (les expressions de
  `poutre.py` retrouvées mot pour mot, inversions exactes, exemple de l'étude), classes et Ecm,
  natures et chape (P-17), §6.7 et §6.4 à la main, pipeline hourdis (P·L/4, levier, plateau avec
  bande plastique sous la pression ELU, chape, P-5, filière rigide et souple, annexe C, P-3, P-15,
  P-11, φ), **P-2** (poids propres dans les efforts : M_G et M_Q nommés, M_Ed = 1,35·M_G + 1,5·M_Q,
  V de même, nature service → coefficients 1, γG = γQ = 1 retrouve le service, l'ancien ÷ 1,5 refusé,
  fissuration en service), pipeline dalle et prédalle (P-4, P-16, P-8, P-7, effets locaux à γQ),
  refus (dont le hourdis sans M_Rd / V_Rd), balayage, dimensionnement (P-14), bibliothèque, JSON,
  cas gelés `ref_chp.json` (régénérés le 02/11/2026 sur la règle P-2), pureté.
- `test_charges_plancher_ecran.py` (67, lot 2, AppTest sur l'application réelle) : navigation et
  tuile, composant du plan, « Données manquantes », onglets, source unique widgets → état → moteur
  et plan (M_Rd, L, hourdis / dalle / prédalle), engin de la bibliothèque et charges par appui
  (refus P-13 puis calcul, bandes teintées, semelles cliquables), plan → état (glisser, cote,
  sens, ±, fenêtres d'engin et de semelle, plateau, idem, filière, dupliquer / supprimer),
  répartiteur par l'onglet 3, proposition de filière, vues Coupe et Balayage, cas multiples,
  JSON aller-retour, fiches ajoutées (engin, plateau azobé, madrier sur champ, fiche de hourdis
  gardée depuis l'onglet Plancher et réappliquée), étanchéité des clés.
- `tools/captures_plancher.js` : captures et mesures aux quatre largeurs (états vide, engin,
  plateau, clic, glisser, coupe, balayage) — aucun débordement, plan visible, calcul abouti,
  plateau hachuré, fenêtre ouverte au clic sur une semelle, position changée au glisser.

![Contrôles du lot 1](images/lot1_controle.png)

*Trois contrôles calculés par le moteur : a · partage d'une filière sur trois hourdis selon sa raideur
relative, de « tout sur un » à « trois tiers » ; b · balayage d'un rotatif le long de la portée ;
c · part efficace d'un plateau selon son épaisseur. Valeurs d'illustration.*

## 7 · Limites de la V1 (reprises de l'étude)

Deux appuis seulement ; filières à 0° ou 90° ; éléments supposés indépendants sous une filière
(conservatif pour le plus chargé) ; pas de chaîne d'étaiement entre étages (P-8) ; pas de
continuité ni de porte-à-faux ; prédalle avant chape refusée ; poinçonnement au sens EN 1992-1-1
réservé aux dalles et prédalles (pour un hourdis, la membrure se juge sur la donnée fabricant).

## 8 · L'écran (lot 2)

```
modules/charges_plancher.py   la vue Streamlit : barre d'outils du cas, plan, coupe, balayage, carte résultat,
                              quatre onglets (Plancher · Charges · Répartiteur · Hypothèses), détail, bibliothèque
plancher/charges/etat.py      l'ÉTAT (dict JSON, source unique) : cas multiples, fiches ajoutées ; construire() →
                              objets du moteur ; contexte_dessin() → plan ; appliquer_changements() ← plan
plancher/charges/dessin.py    le plan cliquable et glissable, la coupe sous un appui, le bandeau de balayage (SVG en mm)
plancher/charges/symboles.py  symboles d'engins en vue de dessus, au trait, À L'ÉCHELLE de leur fiche
tools/captures_plancher.js    captures et mesures PC 1440 / 1280, tablette 1024, téléphone 390 (états vide, engin,
                              plateau, clic, glisser, coupe, balayage)
tests/test_charges_plancher_ecran.py   harnais AppTest + état (source unique, plan → état, JSON, fiches)
```

`plancher/charges` reste **pur** (le test 10 du moteur le garde) : l'état, le dessin et les symboles
n'importent pas Streamlit ; seule la vue `modules/charges_plancher.py` le fait.

**Organisation** (étude § 7, maquette « digeste ») : à gauche le **plan** — appuis A et B en trait
épais, bords libres fins avec une poignée ○ qui change le sens porteur, cote L cliquable, poignées
± du nombre de hourdis, bandes teintées seulement à partir de 85 % (orange) et au-delà de 100 %
(rouge), taux et famille gouvernante **hors du dessin**, engins dessinés à l'échelle de leur fiche,
semelles et étiquettes « P2 · 95 kN », plateaux hachurés avec leur aire efficace en pointillé bleu,
filières, échelle graphique ; sous le plan, la **coupe** sous l'appui sélectionné (éléments voisins,
couche de compression, plateau, semelle, charge, diffusion à 45°). À droite la **carte résultat**
(pastille, une phrase « Flexion · H7 — 343 % », barres par famille, lignes non vérifiées,
« Proposer une filière », « Balayer la position »), puis les quatre onglets. En bas, le détail des
vérifications par élément et par famille.

**Ce qui se règle sur le plan** (composant partagé `acier/composants/svg_cliquable`, retours en mm) :
glisser un engin, une charge, un plateau, une filière ; clic sur un engin (angle, charges par appui,
dupliquer, supprimer), sur une semelle (répartiteur : aucun, plateau, filière, « idem tous les
appuis »), sur un plateau (même fenêtre, et le plateau se décale), sur une filière (élément,
longueur, orientation), sur la cote L, sur les poignées ± et ○. Grammaire des clés :
`<objet>:<réf>.<champ>` avec objet ∈ engin · charge · rep · fil (`dessin.py`, en-tête). Tout retour
passe par `etat.appliquer_changements`, qui écrit dans le cas ; les widgets des onglets (clés
`chp_w_…`) sont des **vues** du même état, relues dès qu'il change par ailleurs.

**Bibliothèque extensible** (exigence du 02/11/2026) : la fenêtre « Ajouter depuis la bibliothèque »
pose un engin de `plancher/donnees/engins.json` ou une **fiche ajoutée** (nom, type d'appuis,
silhouette, appuis avec positions, empreintes, charges, φ, source) ; l'onglet Répartiteur accepte un
**nouvel élément** (plateau ou filière : matériau, f, E, masse, dimensions, profilé du catalogue ou
rectangle b × h posé sur champ — plateau azobé, madrier C24, profilé, plaque de grue) ; l'onglet
Plancher garde les valeurs saisies d'un hourdis comme **fiche du catalogue** (« ＋ Fiche de
hourdis », P-1 : un catalogue transcrit sert de défaut, le bureau indique les siennes). Les fiches
ajoutées voyagent avec le fichier JSON du cas (`💾 Enregistrer` / `📂 Ouvrir`, `etat.charger`
contrôle le module et la version).

**Deux décisions d'écran prises le 02/11/2026, chacune tenue par un test :**

- les quatre étapes sont des **onglets à libellé fixe**, pas des expanders au titre résumé : un
  expander dont le titre change se referme à chaque saisie qui le change (Streamlit le remonte), ce
  qui coupait la saisie des charges par appui ; le résumé vit en tête de chaque onglet ;
- `diffusion.max_creneaux` nomme le **premier** plateau maximal à 1e-9 relatif près : la valeur
  d'un plateau n'est jamais plate au bruit numérique, et le cas gelé `dalle_etai` donnait
  « y = 3,65 m » sous numpy 2.4 et « y = 3,90 m » sous numpy 2.5 pour la même valeur. Le cas gelé
  a été réécrit sur la valeur déterministe (révision du 02/11/2026).

Textes du plan : constants à l'écran quelle que soit la taille du plancher (facteur k sur les tailles
de police et d'étiquette au-delà d'un plan de référence de 11,4 m de large) ; sur téléphone, le
plan garde une largeur minimale lisible et le conteneur du composant défile horizontalement.

![Écran PC](images/ecran_pc.png)

*PC 1440 : télescopique rotatif posé depuis la bibliothèque, charges d'essai par appui, plateau
600 × 600 × 20 sous P2. Les valeurs de résistance (M_Rd 60 kN·m, V_Rd 80 kN) sont des valeurs
d'ESSAI du harnais, pas un catalogue.*

![Écran téléphone](images/ecran_telephone.png)

*Téléphone 390 : mêmes fonctions, colonnes empilées, plan défilant.*
