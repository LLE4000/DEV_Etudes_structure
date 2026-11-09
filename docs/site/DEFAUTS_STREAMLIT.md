# Défauts de l'écran Streamlit relevés en portant les modules sur le site

*Relevé du 07/11/2026, à la construction du site « Stab Méthodes » (site/web). Chaque module a été extrait de son écran Streamlit puis relu par un second agent indépendant, qui a comparé ce que le site AFFICHE à ce que l'écran Streamlit affiche, cas par cas.*

La règle du site (docs/site/MODULES_SITE.md §5) : **on extrait, on ne réécrit pas.** Chaque défaut ci-dessous est REPRODUIT sur le site (les chiffres restent ceux de l'écran Streamlit, la parité est prouvée par les tests), SIGNALÉ à l'écran du site (en rouge quand il est non conservatif) et ÉCRIT en tête de la doc du moteur. Rien n'est corrigé en silence : c'est au bureau de trancher, d'abord dans l'écran Streamlit (modules/…), puis le moteur du site suit — ses tests de parité diront exactement ce qui change.

Exception, déjà appliquée : une saisie que Streamlit ramène en silence à une valeur fausse (négatif lu 0, valeur absurde qui fait planter le calcul…) est REFUSÉE sur le site avec un message.

## 1. Non conservatifs — à trancher en priorité

### Poutres bois

Écran : `modules/poutre_bois.py` · doc : `docs/bois/POUTRE_BOIS_MOTEUR.md`

- NON CONSERVATIF — combinaison_G (relevé à la relecture) : γ_G·G seule avec k_mod permanent n'est jamais vérifiée. 63×200, G 3, Q 0,5, moyen terme : 92 % affichés pour 104 % réels en G seule. Avec G 4, Q 1, court terme : 118 % affichés pour 138 %.
- NON CONSERVATIF — cisaillement_kcr : τ est calculé sans k_cr = 0,67, donc sous-estimé d'un facteur 1,49.
- NON CONSERVATIF — fleche_finale : la formule (1 + k_def)·w(G + ψ₂Q) oublie (1 − ψ₂)·w_inst,Q par rapport à l'EC5 §2.2.3.
- NON CONSERVATIF — appui_els : la compression à l'appui est calculée sous G + Q non pondérées, puis comparée à f_c,90,d. En sens inverse, k_c,90 et la longueur efficace sont omis (côté sûr).
- NON CONSERVATIF — poids_propre : le poids propre n'est jamais compté (effet faible).
- NON CONSERVATIF, à confirmer sur le texte d'EN 338:2016 : dans la base des classes, f_c,90,k de C22 à C40 et f_v,k de C35 et C40 sont plus élevés que la norme (relevé de mémoire).

### Choix de profilé

Écran : `modules/choix_profile.py` · doc : `docs/acier/CHOIX_PROFIL_MOTEUR.md`

- D1 NON CONSERVATIF : V_y,Ed n'est jamais vérifié. A_v,y est absent de la forme de catalogue lue, donc η_Vy = 0 (message rouge sur le site).
- D2 NON CONSERVATIF : η_N est pris sans valeur absolue, une traction se RETRANCHE de la flexion (message rouge ; jauge maintenant vide).
- D3 NON CONSERVATIF : f_y n'est pas réduit au-delà de 40 mm d'épaisseur. 39 profilés HD et HL sont touchés, surestimés de 9,3 % en S235 et de 6 % en S355 ; au-delà de 80 mm, on sort du tableau 3.1.
- D4 NON CONSERVATIF : pas d'interaction effort tranchant – moment (§6.2.8).
- D5 NON CONSERVATIF (mesure corrigée par cette relecture) : pas de classement de section. 28 âmes de classe 4 en compression en S235, 41 en S275, 67 en S355 ; en S355, trois HP ont des semelles de classe 4, donc W_el est non conservatif même en flexion simple. Désormais dit par l'avertissement classe_4.
- D6 NON CONSERVATIF : le flambement ignore le moment (pas de §6.3.3) et une seule courbe sert aux deux axes.
- D7 NON CONSERVATIF : le déversement n'est pas calculé (la case et L_LT ne font rien).

### Taux d'armature

Écran : `modules/taux_armature.py` · doc : `docs/beton/TAUX_ARMATURE_MOTEUR.md`

- NON CONSERVATIF — D9 : nombre d'étriers tronqué sur flottants, int(L_zone/s)+1 : L = 1 m, zone 0–30 %, s = 10 cm → 0,3/0,1 = 2,9999999999999996 → 3 étriers au lieu de 4 ; mesuré : 2 012 zones sur 39 690 zones ordinaires (5,1 %) perdent un étrier. Reproduit, avertissement grave, quotient écrit dans la fenêtre de la zone.
- NON CONSERVATIF — D2 : « Rectangulaire + deux talons » calculée sans ses talons (le libellé ne contient ni « gauche » ni « droit ») : volume et poids du béton sous-estimés, taux kg/m³ surestimé. Reproduit, message grave, talons en fantôme sur la coupe.
- NON CONSERVATIF — D11 : la longueur d'étrier ne prend que le contour de l'âme ; les cadres/épingles de talon ne sont jamais comptés. Reproduit, désormais dit par un avertissement.
- NON CONSERVATIF — D5 : la hauteur de retour proposée à l'ajout (h − 2c) est perdue ; un retour ajouté pèse 0 tant qu'on ne saisit pas sa hauteur. Reproduit, avertissement.
- NON CONSERVATIF — D13 : famille « À ≤ De », zone « À ≤ De » ou sans type ignorées sans un mot. Reproduit, avertissements graves.

### Enrobage

Écran : `modules/enrobage.py` · doc : `docs/beton/ENROBAGE_MOTEUR.md`

- D1, NON CONSERVATIF probable : poteaux (tab. 5.2a). La caption annonce μ_fi ≈ 0,7, mais les distances a suivent la colonne μ_fi = 0,2 (R90 : 200/31 au lieu de 350/53 pour μ_fi = 0,7), et R240 porte des variantes absentes du tableau. a est sous-estimé de 15 à 25 mm. Reproduit, avertissement rouge sur le site. Valeurs de la norme citées de mémoire : à confirmer dans les textes et l'ANB.
- D2, NON CONSERVATIF probable : voiles R180 / R240 à 45 / 55 mm, alors que la colonne μ_fi = 0,7 une face semble demander 50 / 60 mm (45 / 55 sont les valeurs μ_fi = 0,35 deux faces). Reproduit, avertissement rouge sur le site.
- D3, NON CONSERVATIF : classe structurale proposée S1 pour X0 et S2 pour XC1, sans les conditions du tableau 4.3N (par exemple XC1 : 10 mm au lieu de 15 mm en S4 normatif). Reproduit, message sur le site.
- D6, NON CONSERVATIF si on oublie de mettre 0 : le Ø étrier de 8 mm par défaut est aussi retiré de a pour une dalle ou un voile. Message sur le site.

### Rigidité du sol

Écran : `modules/rigidite_sol.py` · doc : `docs/sol/RIGIDITE_SOL_MOTEUR.md`

- NON CONSERVATIF n° 1 : le conseil « Prolonge le profil … descendre à 2,5·B m » ne dit pas qu'il se compte sous l'assise. Exemple : D = 3 m, B = 2 m ; l'écran conseille 5 m alors qu'il faut au moins 8 m sous le TN. Qui suit le conseil garde un k SURESTIMÉ ; le message reste affiché, l'erreur n'est donc pas silencieuse. Le site ajoute la précision.
- NON CONSERVATIF n° 2 : l'abaque (mode 5) n'a pas de longueur L, donc pas de correction de forme de Terzaghi. Pour une semelle filante sur sable, k est surestimé jusqu'à 1,5 fois (facteur (1 + 0,5·B/L)/1,5).
- NON CONSERVATIF n° 3 : profil trop court au plafond 6·B ; k surestimé, et le message accuse à tort le profil saisi.

### Maturité du béton

Écran : `modules/age_beton.py` · doc : `docs/beton/MATURITE_MOTEUR.md`

- NON CONSERVATIF — ② seuil à la température MOYENNE du relevé : annoncé trop tôt quand les jours froids viennent d'abord. Exemple : 7 j à 2 °C puis 7 j à 20 °C, seuil 15 MPa ; annoncé « après 4,8 j », réellement atteint au jour 8 (f_ck(5 j) = 11,36 MPa). Le site le signale en grave.
- NON CONSERVATIF — ⑤ la mesure d'éprouvette est lue comme une résistance caractéristique : f_ck,28 surestimé de 8/β_cc. Exemple : 21,6 MPa à 7 j → C30/37 « cohérent », contre 19,7 MPa (C20/25) lu en moyenne. À trancher par le bureau.
- NON CONSERVATIF d'au plus un jour — ② la date du seuil ignore la fraction de jour (7,93 j après le 02/11 → « le 09/11 »).

### Catalogue des profilés

Écran : `modules/tableau_profiles.py` · doc : `docs/acier/CATALOGUE_MOTEUR.md`

- NON CONSERVATIF — I_t d'un PRS = Σ b·t³/3 (parois minces) : surestimé de +2 % (400x200x8x12) à +10 % (200x150x15x30) contre Saint-Venant (grille convergée) → moment critique de déversement surestimé. Reproduit, écrit en tête de la doc et au résultat (D11).
- NON CONSERVATIF avec η = 1,0 — A_v,z d'un PRS par la formule des laminés t_w·(h − t_f) au lieu de η·h_w·t_w (EN 1993-1-1 §6.2.6(3)(d)) : 52,0 contre 44,0 cm² (+18 %) pour un PRS 300x300x20x40 ; en dessous avec η = 1,2 tant que t_f ≤ h/7. À trancher par le bureau (D12).
- NON CONSERVATIF — deux filtres sur la même propriété : seul le dernier compte, sans le dire (« I_y ≥ 10 000 » puis « ≤ 5 000 » ne garde que la seconde borne). Le site refuse ce cas (D1).

### Flambement

Écran : `modules/choix_profile.py (option « Vérifier flambement »)` · doc : `docs/acier/FLAMBEMENT_MOTEUR.md`

- NON CONSERVATIF : une seule courbe, choisie à la main (b par défaut), pour les deux axes au lieu du Tableau 6.2 axe par axe. Exemple : un HEA autour de z-z est calculé en courbe b au lieu de c.
- NON CONSERVATIF : pas de classe 4. L'aire A est utilisée au lieu de A_eff (environ +4 % sur un IPE 400 en S355).
- NON CONSERVATIF : f_y vaut la valeur nominale de la nuance, sans réduction au-delà de 40 mm d'épaisseur.

### Prédalles

Écran : `modules/predalle.py` · doc : `docs/beton/PREDALLE_MOTEUR.md`

- NON CONSERVATIF : les couches croisées sont calculées au même niveau (ou trop près). La base secondaire et les renforts principaux sont posés tous à h_pre + ⌈Ø/2⌉, les bases supérieures X et Y à c + ⌈Ø/2⌉ + j. La couche du dessus a en réalité un d plus petit (renfort 3 Ø12 à 7,0 et base X à 6,5 : l'une est en vrai à au moins 7,6 cm). Reproduit, signalé à chaque calcul, et écrit en tête de la doc.
- NON CONSERVATIF : la peau h_pre est ramenée en silence dans [3 ; min(15 ; h−1)] quand l'épaisseur baisse ou à l'ouverture d'un fichier ; les couches posées sur la peau descendent et leur d grandit. Le site refuse la saisie et signale le fichier.

### Tableau des armatures

Écran : `modules/tableau_armatures.py` · doc : `docs/beton/ARMATURES_MOTEUR.md`

- NON CONSERVATIF (léger), écrit en tête de la doc du moteur : les sections sont arrondies au mm² le plus proche, et 84 des 168 cases sont arrondies vers le haut (≤ 0,5 mm²), soit au plus +0,8 % (2Ø6 = 56,55 écrit 57 ; 1Ø10 = 78,54 écrit 79 ; 2Ø8 = 100,53 écrit 101). Exemple : qui retient 1Ø40 (« 1 257 ») pour 1 257 mm² n'a que 1 256,6 mm². Le site garde les mêmes nombres dans le tableau mais compare toujours la section exacte.
- NON CONSERVATIF (léger) : l'en-tête 100/n est à une décimale. Poser les barres à 16,7 cm donne 0,2 % de section en moins que la case, à 14,3 cm 0,1 % de moins (Ø20 : 1 881 au lieu de 1 885 mm²/m).

### Charges sur plancher

Écran : `modules/charges_plancher.py (moteur plancher/charges)` · doc : `docs/plancher/CHARGES_PLANCHER.md`

- NON CONSERVATIF : part du tranchant P-3 < 1 sans répartition active → (1 − part) × P de chaque charge concentrée n'est porté par aucun élément (étançon de 100 kN, part 0,7, hourdis posés : Σ réactions 70 kN). Vient du moteur (verification._efforts_hourdis) ; inactif avec la valeur par défaut (part = 1). Reproduit, et dit en rouge par le site.

### Dalles béton armé

Écran : `modules/dalle.py` · doc : `docs/beton/DALLE_MOTEUR.md`

- NON CONSERVATIF, écrit en tête de la doc du moteur : les nappes X et Y d'une même face sont calculées AU MÊME NIVEAU. L'axe automatique ne dépend que de l'enrobage et du Ø, donc les deux treillis inférieurs sont tous deux à 3,5 cm. Croisée, la nappe du dessus est au moins 1 cm plus loin : son As,req réel est 6,5 % plus grand et h_min est sous-estimé de 1 cm. Reproduit pour la parité ; le site affiche un avertissement et invite à saisir l'axe. Le bureau doit trancher.

### Cornières

Écran : `modules/corniere.py` · doc : `docs/acier/CORNIERE_MOTEUR.md`

- NON CONSERVATIF : la vérification « longitudinale (entre fixations) » répète la transversale. M_long = V·s·e sur W = s·t²/6, donc s se simplifie et U_long = U_ELU au bit près. La flexion de la cornière entre fixations (q·s²/8, « TODO » du code) et la concentration de l'effort aux fixations ne sont vérifiées nulle part : un entraxe de 10 cm ou de 1,50 m donne le même taux. **Le site ne le reproduit plus depuis le 10/11/2026** : le module Cornières du site est refondu sur la méthode du classeur de l'ingénieur (feuille P8 — `docs/acier/CORNIERE_DECISIONS.md`, `docs/acier/CORNIERE_MOTEUR.md`), le modèle lame et sa vérification longitudinale ont disparu du site (le modèle lame donnait 120,65 % sur le cas de contrôle du classeur, 88,86 %). L'écran Streamlit, lui, n'est pas modifié et garde ce défaut.

### Poutres béton armé

Écran : `modules/poutre.py` · doc : `docs/beton/POUTRE_MOTEUR.md`

- NON CONSERVATIF : le cisaillement prend d_v = h − min(lit 1 inf. ; lit 1 sup.) et non la hauteur utile des aciers tendus ; avec plusieurs lits, s_th et s_max sont surestimés (cas « audit C-1 » : d_v = 55,0 cm contre d = 52,0 cm, s_th + 5,8 %).
- NON CONSERVATIF : saisies ramenées sans avertissement — un moment négatif devient 0 (la face passe « ok »), un pas 0 ou négatif est vérifié à 30 cm alors que le champ affiche 0 (le site refuse ces saisies).

### Le catalogue des profilés partagé (acier/profiles)

- La largeur de paroi des tubes rectangulaires FORMÉS À FROID est écrite b − 2·r_o (largeur plate nominale), plus favorable que le b − 3t de l'EN 1993-1-5 §4.4(2) : un classement de section qui la lit peut donner une classe trop basse (le module Flambement s'en écarte et prend b − 3t : 63 tubes changent de classe). À vérifier pour tout module qui classe des tubes formés à froid (Choix de profilé…), et à dire dans docs/acier/PROFILES.md.

## 2. Autres défauts (côté sûr, affichage, ergonomie)

### Poutres bois

- fleche_unite : q est multiplié par 1 000 alors que 1 kN/m vaut déjà 1 N/mm, donc les flèches sont 1 000 fois trop grandes. C'est du côté sûr, mais aucune poutre réaliste ne passe l'ELS. Le bureau doit trancher en corrigeant modules/poutre_bois.py.
- kh_reduit : k_h < 1 est appliqué au-delà de h = 150 mm (côté sûr).
- deversement : ni déversement ni effort tranchant dans la flèche ; G_mean est dans la base mais inutilisé.
- Ergonomie : la section proposée à l'ouverture est 38×150 (782 % en flexion) ; « 100% » s'affiche en rouge pour 100,4 % ; la métrique de section tronque les décimales (int) ; ψ₂ ne peut pas descendre sous 0,1.

### Choix de profilé

- D8 : T_Ed est saisi mais ignoré.
- D9 : l'« optimal » est le plus chargé et non le plus léger, et il est jugé sur l'utilisation arrondie (η = 1,000004 « tient »).
- D10 à D13 : tableau trié avec les plus lourds en tête ; sans effort, tout tient à 0 % ; le taux de flambement n'est pas affiché ; décocher un effort efface sa valeur.
- Constaté en relecture, contourné sur le site : l'écran déborde sur des saisies extrêmes qu'il accepte (OverflowError pour M_y ≥ 1e303 ou L_cr ≥ 1e150, division par zéro pour L_cr = 1e-200). Ses formules écrivent en kN un résultat en N (« 65.25×10²·235/1.00 = 1533.5 kN »).

### Taux d'armature

- D1 : libellé de synthèse non formaté « Poids béton (≈{CONCRETE_DENSITY_KG_M3:.0f} kg/m³) ».
- D3 : deux talons + « Renfort talon » bloque l'écran sur « Talon gauche non défini. » (refusé avec la raison sur le site).
- D4 : talon de 0 cm (valeur affichée) lu 10 × 15 cm (refusé sur le site).
- D6 : supprimer une famille fait hériter la suivante de ses valeurs (clés de widgets par rang).
- D7 : le menu « Section » demande de choisir deux fois.
- D8 : en mètres, changer la portée garde les mètres s'ils tiennent, sinon le pourcentage (reproduit et dit).
- D10 : zones qui se recouvrent comptées deux fois ; portée sans zone sans étrier (avertissements).
- D12 : un étrier Ø7,5 s'affiche « 7 » dans le tableau (affichage seulement ; le site écrit 7,5).
- D14 : deux types d'étriers de même nom ambigus (refusé).
- D15 : 2c ≥ b ou 2c ≥ h donne une longueur d'étrier négative (refusé).

### Enrobage

- D4, côté sûr : C_MIN_DUR dépasse de 5 mm le tableau 4.4N pour X0 et XC1 en S3 à S6.
- D5 : « Position dans l'ouvrage » n'entre dans aucun calcul.
- D7, non conservatif d'affichage : :.0f arrondit au pair (18,5 s'affiche « 18 », c_nom 28,5 s'affiche « 28 »). Le site affiche la valeur exacte.
- D8 (côté sûr probable) : Δc_dev est ajouté au feu. D9 (côté sûr) : c_min,b = Ø de la barre principale appliqué au nu de l'étrier. D10 : le plancher de 10 mm est sans effet. D11 (ergonomie) : décocher le feu oublie la durée.
- Relevé à la relecture : l'écran n'a pas de maximum sur les Ø (10⁹ mm s'y calcule). Le site refuse au-delà de 60 / 40 mm ; écart voulu, documenté.

### Rigidité du sol

- Le découpage d'un essai plante quand les dernières mesures n'ont pas d'Ic. L'export « CSV des couches » plante dès qu'une couche n'a pas de module.
- « Enregistrer » ne sauve ni les modes 3 à 6 (clés rs3_, rs4_, rs5_B, rs6_) ni les mesures CPT brutes ; « Réinitialiser » ne remet pas les modes 3 à 6 à zéro.
- Le graphique k(B) force la pression nette et intègre par tranches de 10 cm : il ne passe pas par le k affiché. Les courbes CPT utilisent le découpage par défaut, pas les réglages de l'aperçu.
- Mode 4 : B et L ne sont pas échangés si L < B. Deux sondages du même nom : l'écran ne trace que le premier. La nappe proposée à l'import est celle de la fondation même non cochée, et « Remplir » l'active.
- Saisies hors bornes ramenées en silence à la borne. Aucune borne haute : 1e308 donne « inf » ou une exception.
- Motifs d'arrêt écrits à l'anglaise (« 2.00 m », « 20% »).
- « Ouvrir » accepte en silence un fichier mal formé : un ordre de couches « abc » devient trois couches par défaut ; un id illisible lève une exception Python.

### Maturité du béton

- ③ calculé à la température moyenne du relevé (même cause que le premier point).
- Le t_e de « Données météo » garde l'E_a de l'EC2 même ajustée.
- f_ck(t) négatif au très jeune âge, affiché et tracé (hors domaine, signalé).
- ③ « (actuelle) » s'affiche aussi quand la classe minimale est inférieure à celle du coulage.
- Bornes ramenées en silence (T, E_a, délai…) ; aucune borne haute pour le délai, l'âge de l'essai ni les résistances.
- Valeurs par défaut de ③ et de la T de cure de ⑤ figées au premier affichage.

### Catalogue des profilés

- Une dimension démesurée (1e300 mm) fait tomber la page : OverflowError non attrapée par _perso, qui n'attrape que ValueError ; le site refuse au-delà de 10 m (D10).
- L'export CSV réécrit les NOMS : « HD 260x54.1 » sort « HD 260x54,1 » (456 profilés) — reproduit (D2).
- PRS t_w ≥ b calculé sans un mot : c_f/t_f = −4,17 pour 400x200x300x12 ; le site l'écrit en rouge (D3).
- Le titre « Colonnes affichées — n sur 39 » a une exécution de retard (D4).
- « Garder par défaut » et « Rétablir » écrivent le fichier de préférences commun à tous les utilisateurs ; « Rétablir » ne vide pas la recherche (D5).
- La fiche ne revient pas après qu'un filtre l'a chassée ; un profilé comparé sorti du tableau quitte la comparaison pour de bon (D6).
- Les bornes portent sur la valeur exacte, pas sur l'arrondi affiché (« I_y ≤ 80,1 » écarte l'IPE 80 affichée 80,1) (D7).
- Les profilés personnalisés ignorent les familles et les séries courantes, mais pas la recherche ni les bornes (D8).
- Une dimension sous 0,1 mm est remontée en silence au minimum ; le site la refuse (D9).

### Flambement

- modules/flambement.py n'a aucun calcul, donc rien n'y est reproduit. Les défauts ci-dessous viennent de l'option « Vérifier flambement » de modules/choix_profile.py, la seule formule de flambement de l'application ; ils ne sont pas corrigés et l'écran Streamlit reste inchangé.
- γ_M1 est admis dès 0,8, sous la valeur de l'ANB (1,0) ; un axe de L_cr nulle est ignoré sans message (ec3_buckling_chi rend None).

### Prédalles

- La base principale peut sortir de la peau sans contrôle (peau 3 cm, Ø12 ; peau 5 cm, Ø25 : dessus des barres à 5,75 cm). Avertissement grave sur le site.
- Aucune borne contre l'absurde : un moment de 10³⁰⁰ kN·m est calculé.
- Pas de verdict global : acier excessif et d ≤ 0 sont non conformes sous 100 %.
- Documentation contradictoire sur le jeu premier lit ; un fichier du module Dalle ouvert dans l'écran Prédalle donne une note vide sans un mot ; une peau nulle dans un fichier est gardée à 0 mais calculée 6 cm.
- Hérités de l'écran Dalle (moment négatif lu 0, bornes appliquées en silence, enrobage nul lu 3 cm) : refusés à la saisie ou relevés à la reprise d'un fichier.

### Tableau des armatures

- ρ = 7 860 kg/m³ au lieu de 7 850 : les masses sont plus fortes de 0,13 % (côté sûr pour un poids). Reproduit et écrit sur le site.
- Une même colonne pour deux lectures (n barres en mm² / n barres par mètre en mm²/m), sans aucune unité ; les colonnes d'espacement fixe n'ont pas d'unité non plus.
- « poids kg/m » désigne une masse.
- Seulement quatre espacements fixes (15, 30, 35, 40 cm).
- Image servie recodée en JPEG (légers artefacts).
- st.image(use_container_width=True) est déprécié et sera retiré après le 31/12/2025.

### Charges sur plancher

- Hauteur utile d ≥ h acceptée (dalle, prédalle) : M_adm grandit avec d, résultat optimiste. Le site la refuse.
- Aucune borne haute : portée de 1 000 m, 10 000 éléments ou stock géant figent l'écran (balayage tous les 10 cm). Le site refuse au-delà de bornes larges.
- Déjà relevé par la construction, confirmé : « 0:0,5 » dans les parts annexe C lu {0: 0}, puis division par zéro.
- Déjà relevé, confirmé : plateau plus étroit que l'empreinte → ValueError qui arrête l'écran.
- Déjà relevé, confirmé : une seconde proposition de filière reprend l'identifiant F1p0.
- Déjà relevé, confirmé : le total « kN au total » ignore les files d'étais et les stocks.
- Déjà relevé, confirmé : champs sans bornes basses (zéros et négatifs passent au moteur).
- Déjà relevé, confirmé : la dalle par défaut est à 111 % sous son poids propre.

### Dalles béton armé

- Saisies ramenées en silence : moment ou V négatif lu 0 ; axe ou centre de gravité illisible ou nul lu « automatique » ; treillis illisible lu 0 mm²/m ; à l'ouverture d'un fichier, classe, nuance, type ou Ø inconnus remplacés et bornes forcées. Le site refuse ces saisies avec un message, et la reprise d'un fichier les signale.
- Enrobage 0 lu 3 cm : le champ affiche 0, le calcul et la note PDF prennent 3 cm. Reproduit, avec un avertissement.
- Aucune borne haute sur les sollicitations, le jeu, l'axe ou le centre de gravité : 1e300 kN·m donne un taux de 1e150 %, 1e308 donne « inf ». Le site refuse l'absurde.
- Le calcul est écrit deux fois (écran et note PDF) : identiques aujourd'hui, mais exposés à diverger.
- Unités cm² partielles : seul le bandeau d'une nappe change d'unité.
- Gestion des éléments limitée : pas d'« ajouter une dalle », et ni la dalle 1 ni la section 1 ne se suppriment.
- À décider par le bureau : ni l'espacement maximal des barres ni l'armature secondaire minimale (EC2 9.3.1.1) ne sont vérifiés. La direction principale n'entre dans aucune formule.

### Cornières

- Depuis le 10/11/2026 le site ne reproduit plus cet écran (module refondu sur le classeur de l'ingénieur, `docs/acier/CORNIERE_MOTEUR.md`) : les défauts ci-dessous restent ceux de l'écran Streamlit, qui n'est pas modifié.
- Modèle lame : seule t entre dans le calcul, A et B n'y jouent aucun rôle, et e > B (charge au-delà de l'aile) n'est jamais contrôlé.
- τ_lim,ELS = 0,8 N/mm² par défaut, environ cent fois sous un cisaillement d'acier : côté sûr, mais c'est lui qui gouverne (62,5 % sur la cornière neuve) et il écarte toute cornière de t < 6,25 mm pour 5 kN/m.
- Bandeaux verts (st.success) même quand ❌, longitudinal en bleu. Pourcentages arrondis au pair (:.0f) : 100,4 % s'écrit « 100% ❌ », 62,5 % s'écrit « 62% ».
- L est dans la barre latérale ; la distance au bord du mur est demandée et jamais utilisée.
- Liste de 41 cornières triée par ordre alphabétique : elle s'ouvre sur 100x100x10. 10 sections sont absentes du catalogue, les masses sont de +1,7 à +2,5 % trop fortes, et 90x90x10 est fausse de +38 % (18,54 au lieu de 13,45 kg/m).
- Saisie libre t ≥ A acceptée (10×10×20 est calculée) : ce n'est pas une cornière.
- V, e ou s démesurés font tomber l'écran sur OverflowError.
- SyntaxWarning « invalid escape sequence » au chargement sous Python 3.12 (\( \) du LaTeX dans des chaînes non brutes).

### Poutres béton armé

- Enrobage nul lu 3,0 cm ; bandeau τ vert sans le contrôle de d ; « ≤ » écrit même quand τ dépasse τ_adm,IV ; τ_lim de beton_classes.json jamais lu.

### Soudures

- Aucun défaut NON CONSERVATIF trouvé : sur 25 cas, le calcul est identique au bit (même moteur).
- Défauts 1 à 10 déjà écrits par le constructeur et vérifiés : valeur hors menu affichée comme la 1re option, côté ±2 ramené à +1, StreamlitValueBelowMinError, deux cordons libres du même nom fusionnés, changement de type C8 dégénéré, n = 1 en cochant « par tronçons » (C6 impossible), verdict d'un refus qui montre le 1er message, nombres à l'anglaise, « −157,9² », noms non échappés.
- 11. Une saisie de 1e300 (gorge ou effort) arrête l'écran Streamlit sur decimal.InvalidOperation (vérifié avec un faux st).
- 12. Arrondis contradictoires : formules du moteur en demi vers l'extérieur (F_adm 490,5 → 491), tableaux de l'écran au pair (490).
- 13. Un même effort porte deux noms : V_z / M_y / T saisis deviennent V_y / M_x / M_z dans la vue de face et les efforts réduits (C5, C6) ; en C3, V devient −V_y. Rien ne relie les deux à l'écran.
- 14. Les symboles des dessins sont écrits avec un trait bas ou collés (h_p, V_y, b_pl, fu/(βw·γM2)).
- 15. En C8, les étiquettes « cordon 1 · a 5 » et « arc 3 · a 5 » se superposent quand le point gauche de l'arc tombe au milieu d'un segment. Dessin du moteur (schemas.py), partagé avec Streamlit et la note : non corrigé.
- Nombre de tronçons sans maximum : à n = 200, 400 lignes de résultat.

### Assemblages · pied de poteau (et toute configuration dont une pince est non conforme)

*Relevé du 07/11/2026 par l'étude « modifier sur le dessin » (docs/site/DESSIN_CLIQUABLE_ETUDE.md), reproduit en CPython.*

- AFFICHAGE QUI MINIMISE — calcul « à la main », une tige déplacée à 30 mm du bord pour 1,2·d₀ = 31,2 mm (platine 500, tige 1 à x = 220) : le moteur déclare la géométrie non valide et ne calcule plus les résistances qui en dépendent ; son η max tombe à 0,03 %. L'écran Streamlit écrit « PLATINE NON VÉRIFIÉE » avec ce η ; le site écrivait **« 0 % · Non conforme »** en tête, gouvernant « Trous · pression diamétrale 0,2 kN ≤ 535,8 kN », pendant que la ligne « Géométrie Tab. 3.3 » disait 104 %. Le verdict était juste, le chiffre en tête trompait.
- **Corrigé sur le site, à l'affichage seulement** (les nombres du moteur sont intacts, la parité avec l'écran Streamlit l'est aussi hors de la pastille de tête) : `api_web/assemblages.py`, `eta_R` porte le taux affiché au rapport de la pire pince non conforme, et `_cause_geometrie` nomme la pince (« Géométrie · pince des boulons au bord de la platine · 30,0 mm < 31,2 mm (e ≥ 1,2·d0) ») ou, sans pince en cause, l'alerte bloquante (« Géométrie non valide »). Figé par `tests/test_api_web_assemblages_pied.py` §7 bis (mutation : 7 contrôles rougissent).
- Même correction pour « Vérifié sous réserve » : le site l'écrivait aussi sous « Non conforme » ; il ne s'écrit plus que sous un verdict vérifié, comme le titre « – sous réserve des alertes » de l'écran Streamlit (`ecran_resultats.py:48`, `manuel/ecran.py:349`).
- À trancher par le bureau pour l'écran Streamlit : la même tête (η max du moteur) s'y affiche encore.

### Notes PDF du garde-corps, des soudures et des charges sur plancher (générateurs validés, portés sur le site)

*Relevé du 09/11/2026 en portant ces notes sur le site (api_web/note_cas.py) ; mesuré par tests/test_note_garde_corps.py,
test_note_soudures.py, test_note_plancher.py. Les générateurs ne sont pas modifiés : à trancher par le bureau — sauf
celui du garde-corps, qui a reçu sa page de suite le 09/11/2026 (lot PDF-fix, ci-dessous, « MISE EN PAGE »).*

- AFFICHAGE — arrondi d'un taux en bord de bande : les trois notes écrivent `f"{t·100:.0f} %"` (arrondi ordinaire, sans
  tenir la bande de couleur), l'écran du site arrondit DANS la bande (`commun.pct`). Garde-corps, H = 1378 mm : taux
  100,009 % → note « NON CONFORME · η max 100 % » (le verdict est juste, le chiffre se lit admis), écran « 101 % » ;
  H = 1340 mm : 94,57 % → note « 95 % » sur une pastille vert pâle, écran « 94 % ».
- AFFICHAGE — seuil d'attention : les moteurs des soudures et du plancher (et leurs notes) disent « attention » dès
  85 % (`etat.SEUIL_ATTENTION`, `parametres.seuil_attention`), l'écran du site dès 95 % (code du bureau). Entre 85 et
  95 %, la note écrit « CONFORME · ATTENTION », l'écran « Conforme » (soudures : IPE 300 en ELS à 94,48 % ; plancher :
  hourdis, V_Rd = 95 kN, 90,55 %). Remesuré le 09/11/2026 (lot PDF-fix, balayage : soudures C3 de 83 à 99 %, plancher de
  78 à 108 %) : désaccord de 86 à 94 %, accord ailleurs ; le tableau des cordons de la note des soudures colore à 70 %
  (vert jusqu'à 70 %, ocre jusqu'à 100 %) — un troisième seuil. Non corrigé : un seuil ne change pas sans décision de
  l'ingénieur (GC10 pour l'écran : rouge > 100 %, vert foncé 95–100 %, vert pâle < 95 %).
- MISE EN PAGE — la note du garde-corps DÉBORDE de ses colonnes même dans sa taille « serrée » (relecture du
  09/11/2026, grille de 240 états : 5 types × 3 formes × « Cumuler qh et qv » oui/non × 4 fixations × 2 méthodes, le
  reste par défaut) : 86 débordent, jusqu'à 89 pt. Forme L ou U avec « Cumuler qh et qv » (l'option côté sûr) : 64 sur
  80, seule la fixation scellée tient ; forme L ou U à montant soudé, sans cumul : 18 sur 80 ; forme droite avec cumul
  (barreaudage ou lisses, platine ou platine latérale, Eurocode) : 4 sur 40 ; forme droite sans cumul : aucun. Exemple :
  garde-corps en U avec « Cumuler qh et qv », le reste par défaut : 28 pt en page 1, 73 pt en page 2. Soudures : C6
  (tube sur platine) en quatre tronçons, Eurocode : 7 pt (2 états sur 56 d'une grille configurations × méthodes ×
  jonctions × tronçons). Plancher : aucun sur 15 états (3 exemples × 5 engins). L'écran Streamlit rend ces notes
  telles quelles (garde-corps en U avec cumul : le tableau de la bibliothèque passe sous le cartouche de la page 2) ;
  le site les REFUSE (règle de la poutre : jamais une note tronquée) — pour ces états, pas de note PDF sur le site tant
  que la mise en page du générateur n'est pas reprise ou que le bureau n'a pas tranché. Le message de refus conseille
  de raccourcir les textes saisis : ici, rien de saisi n'y change, seule la configuration (forme, cumul, fixation).
  **Corrigé pour le garde-corps le 09/11/2026 (lot PDF-fix)** — mesuré d'abord dans l'écran Streamlit (AppTest, bouton
  « 📄 Note ») : aucune erreur, un PDF de deux pages annoncé « (PDF, 2 pages) », AUCUN mot de l'écran ; le
  récapitulatif posé sur la conclusion (page 1), la bibliothèque sous le cartouche et en partie HORS de la page (U avec
  cumul, platine, Eurocode : 8 mots sous le bas de la page 2, la dernière ligne « CHS 42.4x3.6 3,44 104 % » perdue).
  Le générateur (`acier/garde_corps/note.py`, le même pour Streamlit et le site) pose désormais une **page de suite**
  quand une colonne déborde encore en taille réduite : les blocs du bas de la colonne y passent, la colonne le dit. Sur
  la grille, plus aucun débordement ni refus ; les 154 états qui tenaient ont exactement la même note qu'avant
  (`tests/test_note_garde_corps_grille.py`). Le message de refus qui reste (soudures C6 en quatre tronçons) dit la vraie
  cause : la mise en page du générateur pour la configuration, pas un texte saisi.

## 3. Ce que le site ne reprend pas de l'écran Streamlit

Partout : la note PDF de la plupart des modules (bouton grisé ; elle est sur le site pour la poutre, et — à
l'intégration du 09/11/2026 — pour le garde-corps, les soudures et les charges sur plancher). Le reste, module par module :

### Poutres bois

- Le bouton Accueil de l'écran Streamlit : le site a sa propre navigation.
- Modifier la base des classes et les coefficients : ce sont des constantes du code, dans Streamlit aussi.
- Rien d'autre ne manque : tous les champs, options, résultats, légendes et hypothèses de l'écran sont présents, l'Annexe nationale comprise après la relecture.

### Choix de profilé

- η_N n'a pas sa propre ligne quand N n'est pas saisi (l'écran écrit « η_N = 0.000 ») ; le site le montre seulement dans la somme η_NM = 0,000 + … Aucune information n'est perdue.
- La colonne A_v,y du tableau (toujours vide chez Streamlit) est omise, et c'est dit sous le tableau.
- On ne choisit pas un profilé en cliquant dans les vues Tableau ou Taux / masse ; on passe par le sélecteur. Streamlit n'a qu'un menu déroulant.
- Rien d'autre : chaque champ, option, résultat, avertissement et tableau de modules/choix_profile.py a été relu ligne à ligne et retrouvé sur le site. La consigne parlait de schéma statique, de charges et de flèche, mais l'écran Streamlit n'en a pas (il prend les efforts ELU) : rien d'inventé.

### Taux d'armature

- Ø d'un type d'étrier : l'écran Streamlit accepte tout nombre tapé de 6 à 16 (7, 9…), le site propose un menu 6/8/10/12/14/16 (une valeur hors pas venue d'un .stab reste proposée) — restriction volontaire, écrite dans la doc.
- Élévation au téléphone : lisible au PC et à la tablette, trop petite au téléphone (poutre de plusieurs mètres à l'échelle sur 340 px).
- Au téléphone, les cadres Résultat et Dessin ne sont pas atteignables faute d'onglets sur la page (défaut partagé, pas du module).

### Enrobage

- Aucun manque fonctionnel par rapport à l'écran Streamlit. Relecture ligne à ligne : titre et caption, type, position, Ø, Ø étrier, Δc_dev, classe d'exposition avec son encadré, mécanisme, « quand la choisir » et exemples, classe structurale (index proposé, aide), case feu, durée, largeur, caption μ_fi, A/B/C avec toutes leurs lignes, la caption du tableau feu, « non vérifié » et l'invite, « couverte par la durabilité », « valeur gouvernée par », « proches », le guide complet. Tout est présent sur le site, et la caption du feu est désormais aussi dans les Données.
- Ni l'écran ni le site ne couvrent : les classes XF et XA ; Δc_dur,γ, Δc_dur,st, Δc_dur,add ; les barres en paquet et les granulats de plus de 32 mm ; le béton contre le sol (seulement signalé) ; a_sd des barres d'angle ; les dimensions minimales au feu ; les poutres continues ; les dalles portant dans deux directions ; μ_fi autre que 0,7.

### Rigidité du sol

- Import d'un PDF à courbes vectorielles : PyMuPDF n'existe pas dans le navigateur ; le site renvoie vers Streamlit.
- Note PDF : hors périmètre, bouton grisé.
- Case « Détail des calculs » : le site montre toujours le détail dans la fenêtre de calcul.
- « Réinitialiser » : le site crée une note neuve.
- Enregistrement AU FORMAT raideur_sol.json : le site relit ce fichier mais enregistre son .stab.
- La position de référence SCIA ne se choisit pas dans le cadre Résultat (il n'a pas de champ) mais en section 2.
- Une erreur de lecture d'un fichier d'essai s'affiche en toast (passager), là où Streamlit laisse un st.error dans l'expander.

### Maturité du béton

- Récupération météo Open-Meteo : géocodage du lieu, archive et prévisions, prolongation automatique au-delà de 16 j, détail horaire (T min h / T max h), dates « Coulage » et « Fin (décoffrage) », cache et re-essai. Le relevé se saisit : tableau collé (celui de Streamlit passe tel quel) ou fichier.
- Données préparées pour un futur PDF (ab_rapport) : le bouton Note PDF est grisé.

### Catalogue des profilés

- Téléchargement direct de profiles.csv et de la fiche .json : texte à copier à la place (mêmes cellules, octet pour octet hors BOM).
- Clic dans le tableau (ligne → fiche, en-tête → tri), recherche et plein écran du tableau Streamlit : on passe par le sélecteur « Profilé » et « Trier par ».
- « Garder par défaut » (préférences communes à toutes les notes) : chaque note garde ses réglages ; « Copier » reprend ceux de la recherche ouverte.
- Téléphone : l'établi commun n'affiche que l'onglet Données (pas de passage au tableau ni au résultat).

### Flambement

- Rien de l'écran Streamlit ne manque : modules/flambement.py n'a aucun calcul (« Page en construction »). Tout ce que le cahier des charges demande est présent : deux axes, longueurs par axe (cas types, k ou L_cr saisis, maintiens), S235 à S460, I/H, U, tubes et cornières, sélecteur trié par masse avec ★, dessins barre, déformée, section et courbes, fenêtre substituée, plusieurs barres par note, avertissement « nouveau calcul ».
- Hors du champ, et dit dans la doc et à l'écran : le flambement par torsion et par flexion-torsion (§6.3.1.4), l'excentricité des attaches des cornières (annexe BB.1.2), le déversement et la flexion composée, les tubes ronds de classe 4 (refusés avec leur motif), le moment dû au décalage e_N, les PRS et caissons soudés.

### Prédalles

- Les unités d'affichage cm/mm et mm²/cm² (paramètres avancés de l'écran ; aucun chiffre n'en dépend).
- Ce que l'écran Streamlit ne fait pas, le site non plus, mais c'est maintenant écrit à l'écran : phase provisoire (prédalle seule, étaiement, levage), raidisseurs, cisaillement à l'interface (EC2 §6.2.5), flèche, fissuration, portée unidirectionnelle. Un module à écrire et à valider par le bureau.
- Au téléphone, le Résultat et le dessin sont inatteignables (enveloppe commune, pas le module).

### Tableau des armatures

- Rien de fonctionnel. L'écran Streamlit n'a qu'un titre, le bouton Accueil, l'image et sa légende : tout est repris (titre, légende désormais visible, les 180 nombres en texte ; l'Accueil est remplacé par la barre du site).

### Charges sur plancher

- Télécharger le JSON au format de l'écran Streamlit (il manque la brique commune « fichier à télécharger ») ; le site sait l'ouvrir, et le .stab porte toute la note.
- Le glisser-déposer sur le plan est remplacé par la saisie x, y et rotation, comme demandé ; les « toasts » de l'écran Streamlit n'ont pas d'équivalent.
- Le bouton « ↔ Balayer la position » de la carte Résultat est remplacé par un message qui renvoie à la vue Balayage.
- Les dimensions, f et la source d'un plateau (légende sous son menu dans l'étape Répartiteur de Streamlit) ne sont lisibles que dans le sélecteur du résultat et dans la vue Méthode, pas sur la ligne de l'appui.

### Dalles béton armé

- Le champ « Partie » des informations du projet : le cadre commun du site gère note, chantier et indice.
- Appuis et charges : l'écran Streamlit n'en a pas, seulement des moments et V par bande. Le plan montre donc la bande, les nappes et les directions, sans appuis.

### Cornières

- **Le site ne reproduit plus l'écran Streamlit depuis le 10/11/2026** : le module est refondu sur la méthode du classeur de l'ingénieur (feuille P8 : deux configurations, deux modes, jeux par l'enveloppe — `docs/acier/CORNIERE_DECISIONS.md`, `docs/acier/CORNIERE_MOTEUR.md`). Aucun champ ni résultat du modèle lame (V, e, s, γ_ELU, σ_lim,ELS, τ_lim,ELS, σ_eq,lim,ELU, transversal, longitudinal) n'existe plus sur le site ; les grandeurs sont celles du classeur (R, L, X, Y, jeux, ancrages, nuance, coefficient).
- Une note enregistrée avec l'ancien module (version 1) se rouvre : le nom et la cornière sont gardés, la charge et l'ancrage prennent des défauts posés dans les ailes, un avis le dit (« note d'avant le 10/11/2026, modèle lame : charges et ancrage à ressaisir, les anciens chiffres ne sont plus calculés »).
- Le catalogue acier/profiles (233 cornières, r1 = congé) reste la source des sections, l'orientation d'une cornière inégale est un paramètre ; la liste de 41 sections de l'écran et ses masses ne sont pas reprises.

### Soudures

- L'enregistrement au format JSON de l'écran Streamlit : le site le LIT, mais ne l'écrit pas (il enregistre en .stab). Il faut une brique commune de téléchargement.
- Le téléphone : sous 700 px, seul le cadre Données est visible (pas de barre Données / Dessin / Résultat). Besoin commun.
