# Charges sur plancher — document de référence du moteur (lot 1)

*Lot 1 livré le 02/11/2026 : le moteur, sans écran. Étude de conception :
`CHARGES_PLANCHER_ETUDE.md` (décisions P-1 à P-17, section 12 pour la revue
des remarques extérieures). Ce document dit ce que le moteur calcule,
avec quels paramètres, ce qu'il refuse, ce qui reste à transcrire, et
comment il est tenu par ses tests.*

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

**Répartiteurs.** Plateau : aire efficace (a + 2c)(b + 2c), c = t·√(f/(3·p)) itéré, c plafonné par le
bord du plateau ; plaque déclarée rigide : plaque entière, pression comparée à la valeur publiée.
Filière en travers : poutre d'Euler libre sur un ressort par élément croisé, k = 3·E·I·L/(a²(L − a)²) à
l'abscisse de la filière ; réactions = charges des éléments ; flexion de la filière γF·M ≤ W·fy/γM0 ;
un ressort qui tire est signalé. Filière dans le sens porteur : charge linéaire uniforme (hypothèse
rigide), aucune répartition entre éléments.

**Hourdis.** Par élément : règle du levier (part proportionnelle au recouvrement de l'empreinte),
parts de l'annexe C si elles sont fournies et si les joints sont clavetés ou le plancher chapé
(flexion seulement), tranchant à `part_tranchant_element` sur l'élément chargé (P-3, défaut 100 %) ;
M aux abscisses des charges et à mi-portée, V aux appuis, charges réparties `g + chape + q_presentes`.
Résistances : M_Rd/γF et V_Rd/γF si nature Rd (P-2), telles quelles si service ; valeurs « avec chape »
seulement si collaborante et classe exigée atteinte (P-17). Membrure : pression P/(aire efficace
+ diffusion à 45° dans la chape) ≤ `p_loc_adm` ; sans donnée : non vérifié si un répartiteur efficace
est présent, défaut « répartiteur exigé » sinon (P-5). Joints : effort transmis par les parts de
l'annexe C, vérifié si `joint_adm_kN_m` et `joint_longueur_m` sont renseignés (P-15). Fissuration :
information si M_cr est renseigné (P-11).

**Dalle et prédalle.** Par mètre : t_y = b + 2·h1 + h ≤ 0,8·l ; b_v = t_y + 0,5·a ; b_m = t_y tant que
`k_flexion_heft240` est `None` (P-4, **très conservatif** sous charge concentrée, voir § 5) ;
superposition en créneaux, maximum sur la largeur ; résistances = formules de `poutre.py` inversées
(§ 2.1 de l'étude) sur la classe saisie ou la classe arrondie vers le bas de fck(t). Effets locaux
EN 1992-1-1 §6.7 et §6.4 avec γF sur la charge (P-9). Prédalle : largeur plafonnée à la prédalle si
joints non armés ou charge à cheval sur un joint (P-16) ; étais en place = portée réduite +
avertissement (P-8) ; chape non durcie = refus de toute charge d'engin ou d'étai (P-7).

**Statuts.** `ok`, `attention` (≥ 0,85), `nok` (> 1), `info`, `non_verifie` (paramètre à transcrire
absent), `refus`. Le statut global est `nok` dès qu'une ligne l'est ; `non_verifie` ne masque rien :
la liste `non_verifies` est rendue à part.

## 4 · Paramètres (`parametres.Parametres`)

| Champ | Défaut | Décision |
|---|---|---|
| `gamma_f` | 1,5 | P-2, uniforme sur G + Q (hourdis Rd, effets locaux) |
| `part_tranchant_element` | 1,0 | P-3 |
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
- `test_charges_plancher_moteur.py` (71) : **copie tenue** des formules BA (les expressions de
  `poutre.py` retrouvées mot pour mot, inversions exactes, exemple de l'étude), classes et Ecm,
  natures et chape (P-17), §6.7 et §6.4 à la main, pipeline hourdis (P·L/4, levier, plateau, chape,
  P-5, filière rigide et souple, annexe C, P-3, P-15, P-11, φ), pipeline dalle et prédalle (P-4,
  P-16, P-8, P-7), refus, balayage, dimensionnement (P-14), bibliothèque, JSON, cas gelés
  `ref_chp.json`, pureté.

![Contrôles du lot 1](images/lot1_controle.png)

*Trois contrôles calculés par le moteur : a · partage d'une filière sur trois hourdis selon sa raideur
relative, de « tout sur un » à « trois tiers » ; b · balayage d'un rotatif le long de la portée ;
c · part efficace d'un plateau selon son épaisseur. Valeurs d'illustration.*

## 7 · Limites de la V1 (reprises de l'étude)

Deux appuis seulement ; filières à 0° ou 90° ; éléments supposés indépendants sous une filière
(conservatif pour le plus chargé) ; pas de chaîne d'étaiement entre étages (P-8) ; pas de
continuité ni de porte-à-faux ; prédalle avant chape refusée ; poinçonnement au sens EN 1992-1-1
réservé aux dalles et prédalles (pour un hourdis, la membrure se juge sur la donnée fabricant).
