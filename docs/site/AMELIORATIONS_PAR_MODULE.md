# Améliorations retenues, module par module

*Ouvert le 14/11/2026, à la demande du propriétaire : « À la fin, on va retenir toutes les améliorations qu'on doit
encore faire dans nos différents modules. Par module, on aura la liste de ce qu'on doit faire ou pas faire, des choses
qu'on doit améliorer ou pas améliorer. »*

**Ce que cette liste est** : des idées et des constats RETENUS, pas des travaux décidés. Rien ici ne change un calcul ni
une note tant que le propriétaire ne l'a pas tranché. Une ligne garde : ce qu'on a vu, ce que ça changerait, l'avis
donné, et l'état — **à trancher**, **à faire** (décidé), **à ne pas faire** (décidé, avec la raison), **fait** (avec le
lot). Une ligne faite ou refusée reste : on ne repose pas deux fois la même question.

Ce qui est déjà un travail décidé vit dans `SUITE_DU_CHANTIER.md` ; les défauts des anciens écrans Streamlit dans
`DEFAUTS_STREAMLIT.md`.

## Dalles et prédalles (`dalle`)

| # | Constat | Ce que ça changerait | Avis donné | État |
|---|---|---|---|---|
| D1 | **Effort tranchant dans la direction SECONDAIRE au droit du joint des prédalles.** Le module vérifie un seul effort tranchant, sur la section entière (τ = V / (0,75·b·h)). Dans le sens des prédalles le joint ne change rien. À TRAVERS le joint, la peau préfabriquée est coupée : seul le béton coulé en place passe (14 cm au lieu de 20 dans l'exemple du lot 5, d = 12,5 cm) — un effort tranchant dans ce sens serait ≈ 40 % plus sévère. | Un second effort tranchant (V_x) et une vérification au joint ; une ligne de plus à l'écran et dans la note. | Cet effort n'existe vraiment que pour une dalle portant dans les deux sens ou une charge concentrée près d'un joint ; négligeable pour un plancher de prédalles classique (un sens, charge répartie). Le propriétaire préfère ne pas le calculer, pour rester simple. | **à trancher** — décision du 14/11/2026 : ne PAS l'écrire dans la ligne « Non vérifiés » de la note ; le retenir ici |
| D2 | **Cisaillement à l'interface prédalle / béton coulé en place** (EN 1992-1-1 §6.2.5). Déjà dit « non vérifié » dans la note. | Une vérification de plus (rugosité, armatures de couture de l'interface). | C'est, en pratique, le contrôle de cisaillement qui compte le plus pour une prédalle. | **à trancher** |
| D3 | **La prédalle qui vérifie aussi l'effort tranchant tient sur deux pages** (sa planche était déjà pleine sans le joint : 9 pt libres). | Écrire A_s,max une seule fois au lieu de quatre (même formule dans les quatre blocs) libère ≈ 78 pt : une page. | Proposé le 14/11/2026. | **à ne pas faire pour l'instant** — le propriétaire garde les deux pages « comme base de réflexion » (14/11/2026) |
| D4 | **Réglages distincts « coulé en place » / « préfabriqué »** (jeu de pose, arrondi). | Deux jeux d'options au lieu d'un. | Évoqué le 13/11/2026 ; un seul jeu aujourd'hui. | **à trancher** |
| D5 | **Jeu entre deux couches d'une dalle** (renfort posé sur une base). | Une option de plus ; change l'axe de calcul des renforts. | — | **à trancher** |
| D6 | **Page de suite de la note** : les schémas de gauche y sont répétés. | Une colonne de gauche vide (ou réduite) sur la page de suite. | Question posée le 14/11/2026, sans réponse. | **à trancher** |

## Poutres béton armé (`poutre`)

| # | Constat | Ce que ça changerait | Avis donné | État |
|---|---|---|---|---|
| P1 | **Paquets de barres** (EN 1992-1-1 §8.9) : deux barres qui se touchent, l'une sur l'autre. Le jeu entre lits vaut 2,5 cm par défaut depuis le lot 6. | Un jeu nul entre deux lits d'un même paquet ; le diamètre équivalent du paquet pour l'enrobage et l'ancrage. | Le propriétaire : « il faudra quand même vérifier ». | **à trancher** |
| P2 | **Note PDF de la poutre** : à relire par le propriétaire. | — | En attente depuis le 13/11/2026. | **à trancher** |

## Gabarit du bureau (page Paramètres)

| # | Constat | Ce que ça changerait | Avis donné | État |
|---|---|---|---|---|
| G1 | **La géométrie de départ et le taux visé** ne sont pas dans le gabarit. | Des valeurs de départ de plus (b, h, enrobage ; taux visé). | — | **à trancher** |
| G2 | **Le gabarit vit dans le navigateur** de chaque poste. | Un gabarit du BUREAU, commun, avec les comptes (stockage en ligne). | Dépend de la question des comptes (Supabase), posée le 13/11/2026. | **à trancher** |

## Profilés et barres en acier (`profil`, `flambement`)

| # | Constat | Ce que ça changerait | Avis donné | État |
|---|---|---|---|---|
| A1 | **Le module « Flambement »** (compression seule, Eurocode 3) fera double emploi quand « Choix de profilé » vérifiera une barre complète par les deux méthodes (`docs/acier/BARRES_ACIER.md`). | Le fondre dans « Choix de profilé », ou le garder comme outil rapide. | À décider quand le module des barres sera en ligne. | **à trancher** |
| A2 | **Cornières, tés, U** : hors du périmètre de la première version des barres (compression simple de la méthode historique seulement). | Le flambement par flexion-torsion (EN 1993-1-1 §6.3.1.4, NCCI SN001a). | Lot ultérieur. | **à trancher** |
| A3 | **Flèches** des barres en acier : hors de la première version. | Une vérification ELS de plus. | Lot ultérieur. | **à trancher** |
| A4 | **Tube rectangulaire en flexion déviée** (Eurocode 3) : ses parois sont classées en compression uniforme, du côté sûr — la classe peut tomber de 1 à 4 dès qu'un petit moment d'axe faible s'ajoute (RHS 200x100x5 S355). | Classer chaque paroi avec son gradient réel de contrainte. | Résultat prudent aujourd'hui, jamais faux ; à affiner si ces tubes servent souvent en flexion déviée. | **à trancher** |
| A5 | **Tube rond en flexion composée** : critère linéaire n + M/M_pl,Rd ≤ 1 (prudent). | La formule M_N,Rd = M_pl,Rd·(1 − n^1,7) du corrigendum de l'EN 1993-1-1, à lire sur le texte. | Gain de quelques pour cent. | **à trancher** |
| A6 | **Effort tranchant élevé avec effort normal** (§ 6.2.10) : la limite d'élasticité est réduite sur TOUTE la section, du côté sûr. | La réduire sur la seule aire de cisaillement. | Rare (V_Ed > 0,5·V_pl,Rd avec N et M). | **à trancher** |
| A7 | **Tube rectangulaire au déversement** : cas général de l'Eurocode 3, courbe d (« autres sections ») — sévère pour un tube (RHS 300x100x8 sur 8 m : χ_LT = 0,81). | Une courbe plus favorable si l'annexe belge ou la pratique du bureau l'admet. | Rare : il faut un tube étroit, long et très sollicité. | **à trancher** |
| A8 | **Le sélecteur des tubes rectangulaires est lent** : chaque profilé de la famille est vérifié par le moteur (199 RHS ≈ 2,6 s hors navigateur, plus dans le navigateur ; 23 IPE : 0,8 s). | Vérifier d'abord les profilés visibles, les autres ensuite ; ou ne vérifier que ce qu'on fait défiler. | Supportable pour les I et les H ; gênant pour les grandes familles de tubes. | **à trancher** |
| A9 | **Les dessins de la barre au téléphone** : la scène fait 640 de large quel que soit l'écran — ses textes font 6 px sur un téléphone. La note PDF, elle, demande une scène de 400 (`dessin_barre.barre(largeur=…)`). | Une scène plus étroite au téléphone (le même dessin, des textes lisibles). | Simple depuis que la largeur est un paramètre ; il faut que la page dise son format au moteur. | **à trancher** |
| A10 | **La note imprime les méthodes cochées à l'écran.** Pour n'imprimer que la méthode historique d'une barre calculée aux deux, il faut la décocher avant. | Un choix au moment d'imprimer (historique, Eurocode 3, les deux). | Un clic de plus à l'écran aujourd'hui ; la boîte de dialogue n'existe dans aucun module. | **à trancher** |
| A11 | **La planche d'une barre répète son dessin** sur ses pages de suite (la règle des planches du site). | Y mettre la section à l'échelle, puis les taux de toutes les vérifications. | Lisible ainsi ; une page de suite gagnerait à montrer autre chose. | **à trancher** |
| A12 | **La ligne « Profilé » ne dit pas le plus léger de la famille qui tient** (l'ancien écran l'écrivait : « plus léger : HEA 220, 50,5 kg/m ») : il se lit dans le sélecteur (★). Le dire sur la ligne demande de vérifier toute la famille à chaque calcul (0,8 s pour 23 profilés, davantage pour les tubes) — la première version ne le disait qu'après l'ouverture du sélecteur, et le résultat dépendait de ce qui avait été ouvert avant (retiré le 14/11/2026). | Le calculer en tâche de fond après chaque résultat, puis l'écrire sur la ligne. | Un clic de plus aujourd'hui (ouvrir le sélecteur). | **à trancher** |
| A13 | **La nature d'un cas de charge est coupée dans son menu fermé** (« Exploitation A — habit… ») : la lettre de la catégorie se lit, pas toujours son usage. | Un libellé plus court dans le menu fermé (« Exploitation A »), l'usage dans la liste ouverte. | Cosmétique ; la liste ouverte dit tout. | **à trancher** |

## Tout le site

| # | Constat | Ce que ça changerait | Avis donné | État |
|---|---|---|---|---|
| S1 | **Remarques d'amélioration des utilisateurs** (une capture + un mot, une liste que le propriétaire traite tous les un à deux mois). | Un stockage en ligne. | Dépend des comptes. | **à trancher** |
| S2 | **Rangement des moteurs béton** (lot 7 : une lecture des matériaux, un dessin commun dalle / prédalle ; aucun chiffre ne doit bouger). | Moins de code, mêmes résultats. | Décidé, à faire seul. | **à faire** |
