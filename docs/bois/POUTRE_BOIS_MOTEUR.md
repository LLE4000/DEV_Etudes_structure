# Poutre bois — le moteur selon l'Eurocode 5, et le module du site

*Refait le 10/11/2026 (lot Q, `docs/site/LOT_Q_QUATRE_MODULES.md` § 1). Le moteur calcule selon la norme ; il ne
reproduit plus l'écran Streamlit `modules/poutre_bois.py`, dont les flèches étaient 1 000 fois trop grandes et dont
quatre vérifications étaient non conservatives (le relevé de ses défauts reste dans `docs/site/DEFAUTS_STREAMLIT.md`).*

Ce que le module vérifie : une pièce de bois MASSIF résineux (C14 à C50) ou de LAMELLÉ-COLLÉ résineux (GL24h à GL32c),
de section rectangulaire, sur DEUX APPUIS SIMPLES, sous charge UNIFORMÉMENT RÉPARTIE — une gîte, un chevron, un
madrier de plateforme, une poutre de coffrage. Il répond : la section tient-elle, quelle vérification gouverne, et
quelle est la section standard la plus légère qui tient.

## 1 · Ce qui est livré

| Fichier | Rôle |
|---|---|
| `bois/poutre/donnees.py` | ce que les normes fixent : chaque valeur UNE fois, sa norme, son édition et son tableau à côté |
| `bois/poutre/entree.py` | la poutre neuve, la relecture STRICTE de la saisie (`normaliser`), les sections standard |
| `bois/poutre/moteur.py` | le calcul (`verifier`), le sélecteur (`candidats_sections`, `optimal`), les aides de la saisie (`AIDES`) |
| `bois/poutre/diagrammes.py` | V(x), M(x) de la combinaison qui gouverne, w_inst(x), w_fin(x), w_net,fin(x) |
| `api_web/bois.py`, `api_web/dessin_bois.py`, `site/web/css/modules/bois.css` | l'écran (établi compact) et ses trois dessins |
| `tests/test_poutre_bois_moteur.py`, `tests/test_api_web_bois.py` | les preuves (§ 7) |

Bibliothèque standard seule (`math`, `re`). Unités du calcul : N et mm, avec **1 kN/m = 1 N/mm**.

## 2 · La saisie

```
{"nom", "classe" (C14…C50, GL24h…GL32c), "standard" (bool), "section" (« 75x225 »), "b", "h" (mm), "L" (m),
 "charges": {"mode": "lineaire"|"surfacique", "G", "Q" (kN/m), "g", "q" (kN/m²), "entraxe" (m), "poids_propre" (bool),
             "nature" (A, B, C, D, E, H, neige, chantier, autre), "duree", "psi2"},
 "appui": {"verifier" (bool), "a", "debord" (mm)}, "maintien": {"type": "continu"|"appuis"|"entretoises", "entraxe" (m)},
 "service" (1, 2, 3), "gamma_G", "gamma_Q", "systeme" (bool), "lim_inst", "lim_fin" (diviseurs de L), "contre_fleche" (mm)}
```

`normaliser` REFUSE (ValueError en français, le champ nommé, l'entrée intacte) : une clé inconnue, un type faux, un
nombre hors de ses bornes (`donnees.BORNES` : b 20 à 400 mm, h 40 à 2 500 mm, L 0,5 à 30 m, G et Q 0 à 200 kN/m…), une
classe, une nature ou une limite hors de son menu, une section standard qui n'est pas du type du bois, une longueur
d'appui qui atteint la portée, des entretoises plus espacées que la portée. Les valeurs DÉDUITES (b et h d'une section
standard ; la durée et ψ₂ d'une nature autre que « autre ») sont remplies quand elles manquent et refusées quand
elles contredisent ce dont elles se déduisent — rien n'est ramené à une borne en silence.

La poutre neuve : C24, 75 × 225, L = 4,00 m, G = 0,6 et Q = 0,8 kN/m (A · habitation), poids propre compté, classe de
service 1, appui de 100 mm vérifié, maintien continu, L/300 et L/250 — conforme (56 %), et le sélecteur propose
38 × 235 (95 %).

## 3 · La méthode — EN 1995-1-1:2004 + A1:2008 + A2:2014, annexe belge

**Charges.** g = G + g_pp, avec g_pp = ρ_mean·b·h·9,81/10⁹ (kN/m, compté par défaut) ; q = Q. En saisie surfacique,
G = g_s·e et Q = q_s·e (e : l'entraxe des pièces).

**ELU** (NBN EN 1990, expression (6.10)) — DEUX combinaisons, chacune avec SON k_mod (§3.1.3(2) : celui de l'action
de plus courte durée) :

| Combinaison | p_d | k_mod |
|---|---|---|
| G seule | γ_G·g | permanent |
| G + Q (absente quand q = 0) | γ_G·g + γ_Q·q | celui de la durée de Q (sa nature) |

M_d = p_d·L²/8 ; V_d = p_d·L/2, pris à l'axe de l'appui SANS réduction (le §6.1.7(3) ne vise que les charges
concentrées proches de l'appui). Chaque vérification retient la combinaison au taux le plus fort et la NOMME — elle
peut différer d'une vérification à l'autre.

1. **Flexion** (§6.1.6, §6.3.3) : σ_m,d = M_d/W ≤ k_crit·f_m,d ; f_m,d = k_mod·k_h·k_sys·f_m,k/γ_M.
   k_h = min((150/h)^0,2 ; 1,3) sous 150 mm (massif, §3.2(3)) ; min((600/h)^0,1 ; 1,1) sous 600 mm (lamellé, §3.3(3)) ;
   1 sinon.
   **Déversement** : maintien « continu » de la fibre comprimée → k_crit = 1 (§6.3.3(5)) ; « aux appuis » →
   l_ef = 0,9·L + 2·h (tableau 6.1 : charge répartie, appliquée sur la fibre comprimée) ; « entretoises » à l'entraxe
   e_m → l_ef = min(e_m ; 0,9·L) + 2·h (moment pris constant entre deux entretoises : côté sûr).
   σ_m,crit = 0,78·b²·E_0,05/(h·l_ef) ; λ_rel,m = √(f_m,k/σ_m,crit) ; k_crit = 1 (λ ≤ 0,75), 1,56 − 0,75·λ
   (0,75 < λ ≤ 1,4), 1/λ² au-delà. Les appuis sont supposés maintenus en torsion : c'est dit à l'écran.
2. **Cisaillement** (§6.1.7) : τ_d = 1,5·V_d/(k_cr·b·h) ≤ f_v,d = k_mod·k_sys·f_v,k/γ_M ; k_cr = 0,67.
3. **Compression à l'appui** (§6.1.5) : σ_c,90,d = V_d/(b·l_ef,a) ≤ k_c,90·f_c,90,d ; f_c,90,d = k_mod·f_c,90,k/γ_M ;
   l_ef,a = a + min(30 ; a ; l_1/2) + min(30 ; a ; débord), où l_1 = L − a est la distance libre entre les appuis ;
   k_c,90 = 1,5 (massif), 1,75 (lamellé, a ≤ 400 mm) si l_1 ≥ 2·h, sinon 1,0 — et un avertissement le dit.
   Vérifiée par défaut ; retirée de la note, la ligne reste à l'écran, « non vérifié ».

**ELS** (§2.2.3, §7.2), combinaison caractéristique, E_0,mean et G_mean :

- w(p) = 5·p·L⁴/(384·E·I) + 1,2·p·L²/(8·G·A) — flexion et effort tranchant, écrits séparément ;
- w_inst = w(g) + w(q) ≤ L/n_inst (300 par défaut) ;
- w_fin = w(g)·(1 + k_def) + w(q)·(1 + ψ₂·k_def) ; w_net,fin = w_fin − w_c ≤ L/n_fin (250 par défaut). Une contre-flèche
  plus grande que la flèche finale laisse la poutre cambrée : le taux est compté nul, et c'est dit.

**Effet système** (§6.6, sur demande) : k_sys = 1,1 sur f_m,d et f_v,d — pas sur la compression d'appui.

**Verdict** : le taux le plus fort ; conforme jusqu'à 100 % compris. **Sélecteur** : les sections standard du TYPE du
bois, chacune recalculée entièrement (son poids propre, son k_h, son déversement), triées par aire — donc par masse ;
✓ la retenue, ★ la plus légère qui tient.

## 4 · Les valeurs, leurs sources, ce qui reste à confirmer

Toutes dans `bois/poutre/donnees.py`. Croisées le 10/11/2026 sur des sources indépendantes (bibliothèques ouvertes
qui transcrivent les normes, fiches de fabricants, fiches Buildwise) ; le test relit chaque table contre une
transcription écrite dans le test.

| Valeur | Source | État |
|---|---|---|
| C14 à C50 (f_m,k, f_t,0,k, f_c,0,k, f_c,90,k, f_v,k, E_0,mean, E_0,05, G_mean, ρ_k, ρ_mean) | EN 338:2016, tableau 1 | confirmé (4 sources lues) |
| GL24h, GL28h, GL32h, GL24c, GL28c, GL32c | EN 14080:2013, tableaux 5 et 4 | confirmé ; GL28h f_t,0,k = 22,3 (des transcriptions écrivent 22,4) |
| k_mod, k_def | EN 1995-1-1 tableaux 3.1 et 3.2 | confirmé |
| k_h, k_crit, σ_m,crit, l_ef (tableau 6.1), k_c,90, les 30 mm de l'appui, k_sys | EN 1995-1-1 §3.2, §3.3, §6.3.3, §6.1.5, §6.6 | confirmé |
| γ_M = 1,30 (massif), 1,25 (lamellé) | NBN EN 1995-1-1 ANB (fiche Buildwise) | confirmé par une source belge |
| k_cr = 0,67 (massif et lamellé) | §6.1.7(2), fiche Buildwise | confirmé AVEC RÉSERVE : un document lu de travers place 0,80 pour le lamellé en Belgique ; 0,67 est la valeur la plus sévère |
| γ_G = 1,35, γ_Q = 1,50, ψ₂ (0,3 / 0,6 / 0,8 / 0) | NBN EN 1990 ANB, expression (6.10), tableau A1.1 | confirmé, surtout sur l'édition précédente de l'ANB (celle de 2021 est en vigueur) |
| durée de la NEIGE : court terme | conception du lot Q | **NON CONFIRMÉ** : le tableau 2.2 de l'EN cite la neige au moyen ET au court terme et renvoie à l'annexe nationale, dont le texte n'a pas pu être lu. Plus sévère : la nature « autre », moyen terme (k_mod 0,80 au lieu de 0,90) |
| limites de flèche L/300 et L/250 | EN 1995-1-1 tableau 7.2 (fourchettes L/300 à L/500, L/250 à L/350) | la norme confirmée ; les limites de l'annexe belge NON TROUVÉES |
| sections standard du bois massif (26) et du lamellé-collé (67, h/b ≤ 6) | liste du lot Q | NON RECOUPÉE sur un tarif de négociant (seules les largeurs de 63 et 75 mm le sont) |

## 5 · Ce que `verifier` rend

Le format complet est en tête de `bois/poutre/moteur.py`. En bref : le matériau, les coefficients, la géométrie, les
charges, les combinaisons, les vérifications — chacune `{cle, nom, groupe, article, enonce, symbole, valeur, limite,
unite, taux, etat, combinaison, detail (une ligne par combinaison), blocs, formules}` —, `taux_max`, `statut`,
`gouvernant`, les blocs du détail du calcul, les hypothèses, les avertissements (`sans_charge`, `poids_propre`,
`kc90_distance`, `kc90_longueur`, `appui_non_verifie`, `contre_fleche`, `systeme_entraxe`), ce qui n'est pas couvert,
les masses.

Une **formule** est `{symbole, expression, gabarit, termes, resultat, unite}` : le gabarit
(« `{γ_G} × {g} + {γ_Q} × {q}` ») se recalcule à partir de ses seuls termes — le test le fait, avec un interpréteur
indépendant — et l'écran y remplace chaque terme par sa valeur : « symbole = expression = valeurs = résultat », comme
une note faite à la main. Une valeur LUE dans un tableau (k_mod, k_def) ou fixée par une condition (k_h = 1,
k_c,90 = 1) n'a pas de gabarit : son expression dit d'où elle vient.

**L'écran n'écrit ni formule ni valeur de norme** : ses phrases d'aide (les ⓘ) sont `moteur.AIDES`, écrites à partir
des constantes et des gabarits du calcul (les 30 mm de l'appui, 0,9·L + 2·h, L/300 à L/500, 1,35…). Un contrôle lit
les textes de `api_web/bois.py` et de `api_web/dessin_bois.py` et rougit si un nombre de norme y est récrit.

## 6 · Ce qui n'est PAS couvert (dit sous chaque résultat et au pied de chaque fenêtre de calcul)

Les charges concentrées et les charges partielles ; les porte-à-faux et les poutres continues ; les entailles d'appui
(k_v, §6.5) ; les vibrations des planchers (§7.3) ; la flexion déviée et la flexion composée ; la résistance au feu
(EN 1995-1-2) ; les perçages et les trémies ; le bois mis en œuvre humide (k_def majoré de 1,0, §3.2(4)).

Et deux hypothèses à connaître : la portée est comptée entre AXES des appuis, la charge portée au-delà de l'axe est
négligée (1 à 2 % de la réaction pour un appui de 100 mm) ; un DÉBORD n'est compté que pour la longueur efficace
d'appui (30 mm au plus) et pour la masse — il ne porte aucune charge et n'est pas vérifié en porte-à-faux.

## 7 · Les preuves

`/usr/bin/python3.12 tests/test_poutre_bois_moteur.py` — 99 contrôles :

- les tables relues contre une transcription indépendante, et les relations de l'EN 384 entre leurs colonnes ;
- **la poutre de l'ouverture déroulée à la main** (49 grandeurs, `math` seul, à 1e-9) : g = 0,66953 kN/m ; G + Q :
  p_d = 2,10386 kN/m, M_d = V_d = 4,20773 ; σ_m,d = 6,649 ≤ 14,769 N/mm² (45,02 %) ; τ_d = 0,558 ≤ 2,462 (22,68 %) ;
  σ_c,90,d = 0,432 ≤ 2,308 (18,70 %) ; w_inst = 6,558 mm (49,19 %) ; w_net,fin = 8,993 mm (56,21 %) ;
- cinq autres exemples à la main (G seule qui gouverne, le déversement, un lamellé-collé, l'appui qui gouverne, la
  classe de service 3), puis 400 poutres tirées au hasard comparées à un calcul de référence écrit dans le test ;
- les unités : la flèche refaite en unités SI par intégration numérique de la charge unité (c'est le contrôle qui
  prend l'erreur « × 1 000 » de l'ancien écran) ; k_h, k_crit à ses trois branches, k_c,90, les limites ;
- chaque formule rendue réévaluée à partir de son gabarit et de ses termes ; les aides de la saisie ;
- le sélecteur, les refus (chaque borne, chaque menu, chaque contradiction), les diagrammes (mêmes maxima que le
  calcul, M = ∫V), la pureté (un sous-processus où numpy, pandas, streamlit… sont refusés : le moteur calcule) ;
- **53 mutations**, chacune rejetée (k_cr retiré, k_mod de la mauvaise durée, G seule oubliée, poids propre oublié,
  5/384 changé, flèche × 1 000, ψ₂ oublié, la flèche finale de l'ancien écran, k_h du lamellé à 150 mm, γ_M, E_0,05,
  les 2·h de l'appui… jusqu'aux textes des formules) ; une mutation témoin (un commentaire) ne l'est pas.

`/usr/bin/python3.12 tests/test_api_web_bois.py` — 112 contrôles : sur 17 états, chaque nombre RELU dans le HTML est
celui du moteur ; les 28 champs reliés ; 27 refus ; le sélecteur ; une note d'avant ; l'échappement ; 2 300 lignes de
formules des fenêtres réévaluées ; les vues ; 24 mutations d'affichage vues rouges.

Le gel : `tests/reference_refonte/bois.json` (33 cas, `tests/reference_refonte/cas.py`).

## 8 · Les écarts avec la conception (assumés, à relire)

1. **Entretoises** : l_ef = min(e_m ; 0,9·L) + 2·h au lieu de e_m + 2·h — une poutre entretoisée ne peut pas être
   plus élancée que la même poutre sans entretoise ; et des entretoises plus espacées que la portée sont refusées.
2. **Appui, côté travée** : la majoration de 30 mm est aussi bornée par l_1/2 (le texte du §6.1.5), en plus de a.
3. **La combinaison G + Q n'existe pas quand q = 0** : elle répéterait G seule avec un k_mod plus favorable.
4. **Le bois qui change de type** (massif ↔ lamellé) garde les dimensions de la section, « sur mesure » quand le
   nouveau type ne la propose pas, et un avis le dit une fois — aucune dimension n'est changée en silence.
5. **La masse** est celle de la pièce d'about à about : L + a + 2 débords.
6. **Les notations** : la bande des efforts écrit M_d, V_d, R_d (les symboles des formules de la conception).

## 9 · Ce que l'ancien écran calculait de travers (et que ce moteur ne reproduit plus)

Flèches avec q × 1 000 (toute poutre sortait à 999 %) ; une seule combinaison, avec le k_mod de la durée choisie à
la main ; cisaillement sans k_cr ; flèche finale (1 + k_def)·w(G + ψ₂·Q) ; compression d'appui sous charges non
pondérées ; ni poids propre, ni déversement, ni flèche d'effort tranchant ; une base de classes mêlant trois éditions
de l'EN 338 ; k_h < 1 au-dessus de 150 mm ; une section proposée à 782 % à l'ouverture. Chacun de ces défauts est une
mutation du test : le remettre fait rougir la suite.

Une note enregistrée par l'ancien module (version 1) se rouvre sur des poutres NEUVES qui gardent leurs noms, et
l'écran le dit une fois : ses données sont à ressaisir.
