# Poutre bois — le moteur selon l'Eurocode 5, et le module du site

*Refait le 10/11/2026 (lot Q, `docs/site/LOT_Q_QUATRE_MODULES.md` § 1). Le moteur calcule selon la norme ; il ne
reproduit plus l'écran Streamlit `modules/poutre_bois.py`, dont les flèches étaient 1 000 fois trop grandes et dont
quatre vérifications étaient non conservatives (le relevé de ses défauts reste dans `docs/site/DEFAUTS_STREAMLIT.md`).*

Ce que le module vérifie : une pièce de bois MASSIF résineux (C14 à C50) ou FEUILLU (D18 à D70, depuis le 24/11/2026),
ou de LAMELLÉ-COLLÉ résineux (GL24h à GL32c), de section rectangulaire, sur DEUX APPUIS SIMPLES, sous charge
UNIFORMÉMENT RÉPARTIE — une gîte, un chevron, un madrier de plateforme, une poutre de coffrage, un linteau en chêne. Il
répond : la section tient-elle, quelle vérification gouverne, et quelle est la section standard la plus légère qui
tient. Cinq vérifications : la flexion avec le déversement, le cisaillement, la compression à l'appui, la flèche
instantanée, la flèche finale (§ 3) ; l'essence et la qualité du bois sont écrites (§ 11) ; la note de calcul PDF sort du
site (§ 10).

## 1 · Ce qui est livré

| Fichier | Rôle |
|---|---|
| `bois/poutre/donnees.py` | ce que les normes fixent : chaque valeur UNE fois, sa norme, son édition et son tableau à côté |
| `bois/poutre/entree.py` | la poutre neuve, la relecture STRICTE de la saisie (`normaliser`), les sections standard |
| `bois/poutre/moteur.py` | le calcul (`verifier`), le sélecteur (`candidats_sections`, `optimal`), les aides de la saisie (`AIDES`) |
| `bois/poutre/diagrammes.py` | V(x), M(x) de la combinaison qui gouverne, w_inst(x), w_fin(x), w_net,fin(x) |
| `api_web/bois.py`, `api_web/dessin_bois.py`, `site/web/css/modules/bois.css` | l'écran (établi compact) et ses trois dessins |
| `bois/poutre/note.py`, `api_web/note_bois.py` | la note de calcul PDF (une planche par poutre) et sa plomberie sur le site (§ 10) |
| `tests/test_poutre_bois_moteur.py`, `tests/test_api_web_bois.py`, `tests/test_note_bois.py` | les preuves (§ 7) |

Bibliothèque standard seule (`math`, `re`). Unités du calcul : N et mm, avec **1 kN/m = 1 N/mm**.

## 2 · La saisie

```
{"nom", "essence" (epicea, sapin, pin_sylvestre, pin_maritime, douglas, meleze, resineux, chene, hetre, chataignier, azobe,
 feuillu, lamelle — § 11), "classe" (C14…C50, D18…D70, GL24h…GL32c), "standard" (bool), "section" (« 75x225 »), "b", "h" (mm), "L" (m),
 "charges": {"mode": "lineaire"|"surfacique", "G", "Q" (kN/m), "g", "q" (kN/m²), "entraxe" (m), "poids_propre" (bool),
             "nature" (A, B, C, D, E, H, neige, chantier, autre), "duree", "psi2"},
 "appui": {"verifier" (bool), "a", "debord" (mm)}, "maintien": {"type": "continu"|"appuis"|"entretoises", "entraxe" (m)},
 "service" (1, 2, 3), "gamma_G", "gamma_Q", "systeme" (bool), "lim_inst", "lim_fin" (diviseurs de L), "contre_fleche" (mm)}
```

`normaliser` REFUSE (ValueError en français, le champ nommé, l'entrée intacte) : une clé inconnue, un type faux, un
nombre hors de ses bornes (`donnees.BORNES` : b 20 à 400 mm, h 40 à 2 500 mm, L 0,5 à 30 m, G et Q 0 à 200 kN/m…), une
classe, une nature ou une limite hors de son menu, une essence inconnue ou d'un autre type que la classe (un chêne en
C24), une section standard qui n'est pas du type du bois, une longueur d'appui qui atteint la portée, des entretoises
plus espacées que la portée. Les valeurs DÉDUITES (b et h d'une section
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
   k_h = min((150/h)^0,2 ; 1,3) sous 150 mm (massif, résineux ou feuillu, §3.2(3) — à condition que ρ_k ≤ 700 kg/m³ :
   D70 et D60 n'ont pas de majoration, la clause le dit) ; min((600/h)^0,1 ; 1,1) sous 600 mm (lamellé, §3.3(3)) ; 1 sinon.
   **Déversement** : maintien « continu » de la fibre comprimée → k_crit = 1 (§6.3.3(5)) ; « aux appuis » →
   l_ef = 0,9·L + 2·h (tableau 6.1 : charge répartie, appliquée sur la fibre comprimée) ; « entretoises » à l'entraxe
   e_m → l_ef = min(e_m ; 0,9·L) + 2·h (moment pris constant entre deux entretoises : côté sûr).
   σ_m,crit = 0,78·b²·E_0,05/(h·l_ef) ; λ_rel,m = √(f_m,k/σ_m,crit) ; k_crit = 1 (λ ≤ 0,75), 1,56 − 0,75·λ
   (0,75 < λ ≤ 1,4), 1/λ² au-delà. Les appuis sont supposés maintenus en torsion : c'est dit à l'écran. La formule
   (6.32) est écrite par la norme pour les résineux (elle suppose G = E/16) ; les classes D de l'EN 338 donnent le même
   rapport, et le module l'applique aux feuillus en le disant dans les hypothèses.
2. **Cisaillement** (§6.1.7) : τ_d = 1,5·V_d/(k_cr·b·h) ≤ f_v,d = k_mod·k_sys·f_v,k/γ_M ; k_cr = 0,67.
3. **Compression à l'appui** (§6.1.5) : σ_c,90,d = V_d/(b·l_ef,a) ≤ k_c,90·f_c,90,d ; f_c,90,d = k_mod·f_c,90,k/γ_M ;
   l_ef,a = a + min(30 ; a ; l_1/2) + min(30 ; a ; débord), où l_1 = L − a est la distance libre entre les appuis ;
   k_c,90 = 1,5 (massif résineux), 1,75 (lamellé, a ≤ 400 mm) si l_1 ≥ 2·h, sinon 1,0 — et un avertissement le dit ;
   1,0 pour un FEUILLU (le §6.1.5(4) n'écrit ses valeurs que pour le résineux massif et le lamellé ; avant le 24/11/2026,
   un feuillu aurait reçu 1,5 : non conservatif).
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
| D18 à D70 (les dix colonnes du tableau 3) | EN 338:2016, tableau 3 | transcrit DE MÉMOIRE le 24/11/2026 et recoupé par les relations de l'EN 384 (f_t,0,k = 0,6·f_m,k ; E_0,05 = 0,84·E_0,mean ; G = E/16 ; ρ_mean = 1,2·ρ_k) et des lectures partielles ; **f_v,k des classes D (3,5 à 5,0) À RECOUPER** sur la norme |
| GL30h, GL30c | EN 14080:2013, tableaux 5 et 4 | transcrits de mémoire ; ρ_k de GL30c (390) à recouper |
| γ_M = 1,30 (massif, résineux ou feuillu), 1,25 (lamellé) | NBN EN 1995-1-1 ANB (fiche Buildwise) ; le tableau 2.3 de l'EN ne distingue pas le feuillu du résineux | confirmé par une source belge (le résineux) |
| k_c,90 = 1,0 pour un feuillu ; k_h sans majoration si ρ_k > 700 kg/m³ | EN 1995-1-1 §6.1.5(4), §3.2(3) | confirmé : le texte des deux clauses |
| classements visuels → classes (§ 11) : NF B 52-001-1 (ST-I, ST-II, ST-III ; chêne D24 et D30 ; hêtre H2, H3), DIN 4074-1 (S13, S10, S7), INSTA 142 (T3, T2, T1), BS 4978 (SS, GS) | EN 1912:2012 | la règle française confirmée ; DIN, INSTA et BS de mémoire ; les qualités belges (STS 04, NBN B 16-520 : S6, S8, S10) NON reliées — l'ingénieur tranche |
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

`/usr/bin/python3.12 tests/test_poutre_bois_moteur.py` — 106 contrôles (99 au 10/11/2026 ; les feuillus, les essences et
les qualités le 24/11/2026, § 11) :

- les tables relues contre une transcription indépendante (C, D et GL), et les relations de l'EN 384 entre leurs
  colonnes ;
- **la poutre de l'ouverture déroulée à la main** (49 grandeurs, `math` seul, à 1e-9) : g = 0,66953 kN/m ; G + Q :
  p_d = 2,10386 kN/m, M_d = V_d = 4,20773 ; σ_m,d = 6,649 ≤ 14,769 N/mm² (45,02 %) ; τ_d = 0,558 ≤ 2,462 (22,68 %) ;
  σ_c,90,d = 0,432 ≤ 2,308 (18,70 %) ; w_inst = 6,558 mm (49,19 %) ; w_net,fin = 8,993 mm (56,21 %) ;
- sept autres exemples à la main (G seule qui gouverne, le déversement, un lamellé-collé, l'appui qui gouverne, la
  classe de service 3, un chêne D30, un azobé D70 aux appuis), puis 400 poutres tirées au hasard — des trois types —
  comparées à un calcul de référence écrit dans le test ;
- les unités : la flèche refaite en unités SI par intégration numérique de la charge unité (c'est le contrôle qui
  prend l'erreur « × 1 000 » de l'ancien écran) ; k_h, k_crit à ses trois branches, k_c,90, les limites ;
- chaque formule rendue réévaluée à partir de son gabarit et de ses termes ; les aides de la saisie ;
- le sélecteur, les refus (chaque borne, chaque menu, chaque contradiction), les diagrammes (mêmes maxima que le
  calcul, M = ∫V), la pureté (un sous-processus où numpy, pandas, streamlit… sont refusés : le moteur calcule) ;
- **58 mutations**, chacune rejetée (k_cr retiré, k_mod de la mauvaise durée, G seule oubliée, poids propre oublié,
  5/384 changé, flèche × 1 000, ψ₂ oublié, la flèche finale de l'ancien écran, k_h du lamellé à 150 mm, k_h d'un
  feuillu lourd, k_c,90 d'un feuillu à 1,5, γ_M d'un feuillu à 1,25, f_c,90,k du D30, une essence d'un autre type
  admise, E_0,05, les 2·h de l'appui… jusqu'aux textes des formules) ; une mutation témoin (un commentaire) ne l'est pas.

`/usr/bin/python3.12 tests/test_api_web_bois.py` — 121 contrôles : sur 19 états, chaque nombre RELU dans le HTML est
celui du moteur ; les 29 champs reliés ; 29 refus ; le sélecteur ; les menus des essences et des classes ; l'en-tête
des vues (deux lignes au plus, la légende au survol) ; une note d'avant ; la note PDF par le routeur ; l'échappement ;
2 300 lignes de formules des fenêtres réévaluées ; les vues ; 24 mutations d'affichage vues rouges.

`/usr/bin/python3.12 tests/test_note_bois.py` — 57 contrôles : la note PDF (§ 10).

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

## 10 · La note de calcul PDF (24/11/2026)

Demande du propriétaire : « dans la note de calcul, il ne faut pas qu'il y ait de blabla excessif ». La note est faite
par `bois/poutre/note.py` sur le kit `ndc_pdf` (la garde du site, puis UNE planche A4 paysage par poutre, comme les
notes Poutre et Dalle) ; `api_web/note_bois.py` est la plomberie du site (`bois.telecharger(etat, "note_pdf", infos)`,
le bouton « Note PDF » de la page : `construire.NOTE_PDF`) ; l'identification (projet, partie, indice, date) est
TOUJOURS celle de la note du site, comme pour les autres modules (`docs/site/MODULES_SITE.md` § 4 quinquies).

**La garde** : une ligne de sommaire par poutre — « P1 — 75 × 225 mm · C24 épicéa · L = 4,00 m », le type de bois, sa
norme, « η max 56,2 % » (le η du verdict de l'écran, au dixième), la pastille VÉRIFIÉ / À LA LIMITE / NON VÉRIFIÉ (les
seuils du site : `acier/garde_corps/membres.statut`).

**La planche** : à gauche, la section (le fil ou les lamelles) et le schéma statique (les appuis, la charge, L, le
maintien), puis quatre blocs de données — POUTRE (section, portée, appui, maintien, classe de service, limites), BOIS
(essence, classe, la première qualité visuelle qui y mène, f_m,k, f_v,k, f_c,90,k, E_0,mean, E_0,05, G_mean, ρ_mean),
CHARGES (G et Q ou g, q et l'entraxe, le poids propre et « non compté » s'il ne l'est pas, la nature de Q, sa durée,
ψ₂), COEFFICIENTS (k_mod des deux durées, k_def, k_h, k_crit, k_c,90, γ_M, γ_G · γ_Q, k_sys sur demande). À droite,
six vérifications en colonnes, chaque formule du moteur écrite « symbole = expression = valeurs = résultat » (les
MÊMES objets que les fenêtres de calcul de l'écran : `verifier(...)["calcul"]`), une valeur lue (k_h, k_crit, k_c,90,
k_def) sur sa ligne avec d'où elle vient : **1** Charges et combinaisons (g_pp, g, le tableau des combinaisons — une
ligne par combinaison, celle qui gouverne l'ELU en gras —, p_d, M_d, V_d de cette combinaison) ; **2** Flexion et
déversement (W, k_h, k_crit et ses termes, f_m,d, σ_m,d ; le verdict) ; **3** Cisaillement (f_v,d, τ_d ; le
verdict) ; **4** Compression à l'appui (l_1, l_ef,a, k_c,90, f_c,90,d, σ_c,90,d ; le verdict — ou « Non vérifiée » quand
elle est retirée) ; **5** Flèches (w_g,m, w_g,v, w_g, w_q, w_inst ; k_def, w_fin, w_net,fin si une contre-flèche est
donnée ; les deux verdicts avec L/n) ; **6** Hypothèses — les SEULES propres à la poutre (le maintien, l'appui, un
débord, un feuillu, l'effet système, une contre-flèche qui dépasse), ce qui n'est pas couvert, les normes. La prose
générale du moteur (la poutre sur deux appuis, les deux combinaisons, V_d à l'axe, les flèches, l'essence écrite) n'est
PAS répétée : la planche la dit par ses blocs (`note.HYPOTHESES_GENERALES`). Une vérification gouvernée par l'autre
combinaison que l'ELU le dit en une ligne. Une planche trop longue se POURSUIT sur une page suivante (« (suite) »), la
garde compte ses pages ; une poutre hors domaine ou une note vide est REFUSÉE avec son message (jamais une note tronquée
ni un chiffre faux). Mesuré le 24/11/2026 : 0,1 à 0,3 s par note en CPython, une page par poutre sur les dix états du
test, le bloc Hypothèses de 650 à 770 caractères.

**Sur le site** : le générateur est dans le paquet du MOTEUR (`bois` est un paquet empaqueté entier) et le kit dans le
paquet de la note — la note du bois n'ajoute AUCUN fichier au paquet de la note (`note_bois.FICHIERS` est vide, prouvé
par le relevé « comme dans le navigateur ») et ne charge pas numpy. La note riche du relevé (`construire.NOTES_RELEVE`) :
une solive C24 en charges surfaciques, un linteau en chêne D30 maintenu aux appuis qui ne tient pas, une poutre GL28c
sur mesure entretoisée (service 2, effet système, appui non vérifié, contre-flèche, L/400 et L/300).

**Preuves** : `/usr/bin/python3.12 tests/test_note_bois.py` — 57 contrôles : sur dix états (dont une note à deux et une
à trois poutres, une note d'avant par le routeur), la PARITÉ avec `moteur.verifier` par pdftotext (le sommaire, l'η max
et l'état, le taux et la comparaison de chaque vérification, le résultat et la ligne de chaque formule imprimée, le
panneau d'après le moteur, le tableau des combinaisons, les hypothèses propres et l'absence de la prose générale — un
budget de caractères et des phrases interdites —, le non couvert, la référence, les pages et le sommaire) ;
l'identification piégée ; la page de suite ; les refus ; la sonde navigateur (sans fontTools ni accélérateurs C, le
Pillow factice, les polices du paquet, le même texte qu'en CPython, aucun fichier à ajouter au paquet) ; huit mutations
vues rouges — dont « les verdicts ELU absents de la planche », le défaut que la suite a trouvé en naissant (le kit ne
dessine un verdict que là où un élément le place : la première version de la planche n'en plaçait que pour les
flèches).

## 11 · Essences et qualités du bois (24/11/2026)

Demande du propriétaire : « que tu mettes les essences de bois et les qualités de bois ». Le calcul de l'Eurocode 5 ne
lit que la CLASSE de résistance (C24, D30, GL28c) ; l'essence et la qualité visuelle sont ce que le bureau ÉCRIT et ce
qu'il commande. Le module les porte donc comme une information liée à la classe, jamais comme une valeur de calcul — et
la planche et les hypothèses le disent : « L'essence (chêne) est écrite, pas calculée : les valeurs sont celles de la
classe D30 ».

- **L'essence** (`donnees.ESSENCES` : clé, libellé, nom latin, TYPE) fixe le type du bois — résineux massif (épicéa,
  sapin, pin sylvestre, pin maritime, douglas, mélèze, « résineux non précisé »), feuillu massif (chêne, hêtre,
  châtaignier, azobé, « feuillu non précisé »), lamellé-collé (épicéa, sapin). Changer d'essence vers un autre type
  ramène la classe par défaut du type (D30, GL24h, C24) et la section suit la règle du § 8.4 ; changer de classe vers un
  autre type ramène l'essence « non précisée » de ce type. Une essence d'un autre type que la classe est REFUSÉE à la
  relecture (« L'essence « Chêne » est un bois massif feuillu : la classe C24 (bois massif résineux) ne lui convient
  pas »).
- **Les feuillus** : les huit classes D18 à D70 de l'EN 338:2016 (tableau 3), γ_M = 1,30, k_h de la clause §3.2(3)
  (pas de majoration au-delà de 700 kg/m³ : D60 et D70), k_c,90 = 1,0 (§6.1.5(4)), le déversement par la formule (6.32)
  en le disant ; les sections standard du massif. Deux exemples déroulés à la main dans la suite du moteur (chêne D30
  100 × 200, L = 5 m ; azobé D70 100 × 120 aux appuis).
- **La qualité visuelle** (`donnees.CLASSEMENTS`, `qualites(essence, classe)`) : pour une essence et une classe, les
  classements visuels qui y mènent d'après l'EN 1912 — la règle française NF B 52-001-1 (ST-I → C30, ST-II → C24,
  ST-III → C18 pour les résineux ; chêne D24 et D30 ; hêtre H2 → D35, H3 → D24), la règle allemande DIN 4074-1 (S13, S10,
  S7), la règle nordique INSTA 142 (T3, T2, T1), la règle britannique BS 4978 (SS, GS). Le menu des classes de l'écran
  les écrit à côté de chaque classe (« C24 · ST-II (NF B 52-001-1) · S10 (DIN 4074-1) · T2 (INSTA 142) · SS (BS 4978) »),
  la rubrique « Poutre » montre l'essence et la qualité, l'en-tête de la vue et la note disent « C24 épicéa », la planche
  imprime la première qualité qui mène à la classe. Les correspondances DIN, INSTA et BS sont écrites de mémoire ; les
  qualités BELGES (STS 04, NBN B 16-520 : S6, S8, S10) ne sont PAS reliées — c'est à l'ingénieur de dire lesquelles le
  bureau commande, et la table est faite pour les recevoir (une ligne par règle).

Preuves : `tests/test_poutre_bois_moteur.py` (les tables D et GL30, les relations de l'EN 384 sur les classes D, les
essences et leur type, les qualités, les refus, les deux exemples à la main, les mutations — k_h d'un feuillu lourd,
k_c,90 d'un feuillu à 1,5, γ_M d'un feuillu à 1,25, une essence d'un autre type admise) ; `tests/test_api_web_bois.py`
(les menus, le changement d'essence et de classe dans les deux sens, l'en-tête, les refus).
