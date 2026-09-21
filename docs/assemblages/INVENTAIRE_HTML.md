# Inventaire du HTML de référence — matrice de traçabilité (phase 1)

**Source** : `acier/reference/assemblage_double_corniere_EC3.html` (943 lignes :
`<style>` l. 7–65, corps l. 67–103, moteur `DC` l. 104–526, dessins `DCDraw`
l. 527–680, interface l. 681–940). Lu en entier.

**Convention** : une ligne par élément, colonne « Destination » = fichier Python
qui le porte. Abréviations des destinations :

| Abrév. | Fichier |
|---|---|
| `BIB` | `acier/bibliotheques.py` + `acier/donnees/bibliotheques.json` |
| `RES` | `acier/resultats.py` (structures `Verification`, `Alerte`, `Pince`, `EffortBoulon`) |
| `FMT` | `acier/formats.py` (formatage fidèle au JS : `F`, `f0`, `pct`, `js_str`) |
| `ENT` | `acier/assemblages/poutre_poutre/doubles_cornieres/entrees.py` |
| `MOT` | `…/doubles_cornieres/moteur.py` |
| `BEN` | `…/doubles_cornieres/benchmark.py` |
| `SCH` | `…/doubles_cornieres/schemas.py` |
| `TXT` | `…/doubles_cornieres/texte.py` |
| `RAP` | `…/doubles_cornieres/rapport.py` |
| `UI` | `…/doubles_cornieres/interface.py` (+ sous-modules d'écran) |
| `CMP` | `acier/composants/svg_cliquable/` |
| `PAGE` | `modules/assemblages.py` + `acier/registre.py` |
| `T` | tests `tests/test_assemblages_*.py` |

Les noms de grandeurs du JS sont **conservés tels quels** en Python (clés du JSON
de référence). Un élément sans destination serait une migration incomplète : il
n'y en a aucun.

---

## 1. Données d'entrée — 84 clés (`defaults()`, l. 124–135)

| Groupe du formulaire (l. 696–723) | Clés | Type | Condition d'affichage | Destination |
|---|---|---|---|---|
| Identification | `id_projet id_rep id_red id_date` | texte | toujours | ENT → UI (groupe « Identification »), TXT, RAP (cartouche) |
| Mode de calcul et fixations | `mode_calc` (VÉRIFICATION / PRÉDIMENSIONNEMENT), `fix_P`, `fix_S` (Boulonnée / Soudée) | choix | toujours | ENT → UI |
| Poutre principale | `prof_P` (90 + Personnalisé), `hP_u bP_u twP_u tfP_u rP_u` | choix / nombre mm | dimensions **si `prof_P` = Personnalisé** | ENT → UI |
| | `nu_P` (6 aciers + Personnalisé) | choix | toujours | ENT → UI |
| Poutre secondaire et grugeage | `prof_S`, `hS_u bS_u twS_u tfS_u rS_u`, `nu_S` | idem | dimensions si Personnalisé | ENT → UI |
| | `d_top d_nt d_nb l_n r_n g_h` (mm), `lt_ok` (Oui/Non) | nombre / choix | toujours | ENT → UI ; `r_n` information d'exécution, **non utilisé** dans les résistances (conservé) |
| Cornières | `corn_u` (18 + Personnalisé), `k1_u k2_u kt_u kr_u` | choix / mm | dimensions **si Personnalisé** | ENT → UI |
| | `orient` (2 valeurs), `LC_u`, `z_C`, `nu_C` | choix / mm | toujours | ENT → UI |
| Boulons | `boulon_u` (8), `classe` (4), `trou` (Normal/Surdimensionné), `d0_u`, `filet` (Oui/Non), `cat` (A/B/C) | | toujours | ENT → UI |
| | `mu_s`, `k_s` | nombre | **si `cat` ≠ A** | ENT → UI |
| | `k_ser` | nombre | **si `cat` = B** | ENT → UI |
| Groupe S | `n1S_u`, `n2S_u` (1/2), `p1S_u`, `e1S_u`, `e2b_u` | | **groupe visible si `fix_S` = Boulonnée** | ENT → UI |
| | `p2_S` | mm | si `n2S_u` = 2 | ENT → UI |
| Groupe P | `n1P_u`, `n2P_u` (1/2), `p1P_u`, `e1P_u`, `gA_u` | | **si `fix_P` = Boulonnée** | ENT → UI |
| | `p2_P` | mm | si `n2P_u` = 2 | ENT → UI |
| Soudures | `a_P lh_P` | mm | **si `fix_P` = Soudée** | ENT → UI |
| | `a_S lh_S` | mm | **si `fix_S` = Soudée** | ENT → UI |
| Efforts ELU | `V_Ed N_Ed H_Ed` (kN), `M_Ed` (kNm) | nombre | toujours | ENT → UI |
| Coefficients partiels | `g_M0 g_M2n g_M2 g_M3 g_M3s` | nombre | toujours | ENT → UI ; **regroupés en un seul endroit**, à caler sur l'ANB |
| Options du modèle | `opt_exc` (Oui/Non), `k_rot`, `opt_blf`, `expo` | | toujours | ENT → UI |
| Acier personnalisé | `fy_u fu_u bw_u` | nombre | **si un des `nu_*` = Personnalisé** | ENT → UI |
| Règles du prédimensionnement | `eta_c k_e1 k_p1 k_e2`, `pd_dmin pd_dmax` (8 boulons) | | **si `mode_calc` = PRÉDIMENSIONNEMENT** | ENT → UI |
| Marquage ◆ | `PDK` = `corn_u LC_u boulon_u n1S_u n2S_u p1S_u e1S_u e2b_u n1P_u n2P_u p1P_u e1P_u gA_u` (13) | | en prédimensionnement : champs grisés, remplacés par la proposition | ENT (`PILOTEES_PAR_PREDIM`) → UI, SCH (cotes verrouillées) |
| Libellés, unités, aides (`h`), notes de groupe | 84 libellés, unités `mm / kN / kNm / MPa / -`, 17 textes d'aide, 3 notes de section | | | ENT (repris **mot pour mot**) → UI |
| Normalisation (`compute`, l. 145–148) | 41 clés `NUM` passées par `N()` ; `V_Ed` en valeur absolue ; `Mabs = |M_Ed|` ; `H = |H_Ed|` ; `n2S_u`/`n2P_u` acceptés en chaîne ou nombre | | | MOT |

## 2. Paramètres géométriques (`compute`, l. 161–191)

| Grandeur | Définition | Destination |
|---|---|---|
| Paramètres retenus | `boulon n1_S n2_S p1_S e1_S e2b_S n1_P n2_P p1_P e1_P g_A L_C` (saisie ou proposition selon le mode ; `n1` arrondi `Math.round` ≥ 1, `n2` ∈ {1, 2}) | MOT |
| Cornière retenue | `b_A b_B t_C r_C corn_txt` (orientation, Personnalisé, ou proposition) | MOT |
| Boulon retenu | `d_b d_0 As_b A_cis d_m d_w` (d0 imposé si `d0_u` > 0 ; `A_cis` = As si filetage dans le plan) | MOT |
| Groupe S | `g_B z_S e2a_S e1bot_S e1b_S h_e n_S Ip_S xm_S ym_S` | MOT |
| Groupe P | `p_3 e2a_P e1bot_P zt_P zb_P n_P Ip_P xm_P ym_P e_P` | MOT |
| Cordons | `Lw_S xg_S Iw_S Lw_P xg_P Iw_P zw_S ew_P` | MOT |
| Excentricités | `zeff`, `M_S = V·zeff/1000 + |MEd|`, `M_P` (option), `mod_S`, `mod_P` (libellés du modèle) | MOT → UI (tableau « Paramètres retenus »), TXT, RAP |
| Recommandation | `rec_L` = `L_C ≥ 0,6·h_S` | MOT → UI |
| Grugeage | `cas_g` (0/1/2), `h_T`, `Asec_S`, `A_Tee`, `Av_S`, `hw_T zb_T I_N W_N rho_N Mv_N Mc_2 el_N lim_N ln_max val_N` | MOT |
| Aires cornières | `Ant_B Anv_B Ant_A Anv_A` | MOT |
| Tronçon en T | `e1A_T p1A_T leff_T m_T n_T ew_T Mpl_T SFt_T FT_1 FT_2 FT_3` | MOT |
| Blocs en traction | `Vt1_B Vt2_B Vt1_S Vt2_S` | MOT |
| Âme principale | `et_P eb_P Av_P` | MOT |

## 3. Matériaux (`DB.aciers`, l. 117 ; `compute` l. 155–159)

| Élément | Destination |
|---|---|
| 6 aciers `S235 S275 S355 S355 (fu 490) S355 N/NL S460 N/NL` : `fy fu bw` | BIB (depuis le JSON, jamais recopié) |
| Acier Personnalisé : `fy_u fu_u bw_u` appliqué à P, S ou C selon `nu_*` | MOT |
| `fy_P fu_P bw_P fy_S fu_S bw_S fy_C fu_C bw_C` | MOT |
| Classe de boulon : `f_ub`, `a_v` (= αv si filetage dans le plan, sinon 0,6), `k_trou` (0,8 si surdimensionné) | MOT |
| Note « S355 : fu = 510 (Tab. 3.1) ; ligne fu 490 : autres références ; à caler sur ton ANB » | ENT (aide du champ `nu_P`) → UI |

## 4. Boulons (`DB.boulons`, `DB.classes` ; `compute` l. 268–289 ; `boltTable` l. 391–398)

| Élément | Destination |
|---|---|
| 8 boulons M12…M36 : `d d0 As A dm dw` ; 4 classes : `fyb fub av` | BIB |
| Résistances élémentaires `Fv_Rd Ft_Rd Bp_Rd Fp_C` | `acier/resistances.py` (commun à la famille) appelé par MOT |
| Joint long : `Lj_S Lj_P bLf_S bLf_P` (option `opt_blf`) | MOT |
| Efforts par boulon le plus chargé : `Fz_S Fx_S F_S Fz_P Fx_P F_P Ft_P` | MOT |
| Tableau par boulon (répartition élastique, inertie polaire) `tabS`, `tabP` : `i j x y fx fz f` | MOT (`bolt_table`) → RES `EffortBoulon` → UI (dépliable sous « Boulons »), T |
| Pression diamétrale `bearing()` : `abv k1v abh k1h Fbv Fbh ipd` pour `B A S P` (bords libres, `exV/exH` pour P) | MOT |
| Chaîne `bvals()` (αb, k1, Fb par direction) | MOT (champ `vals`) → UI, TXT |

## 5. Cornières (`DB.cornieres`, `DB.ailes`, `DB.epais`)

| Élément | Destination |
|---|---|
| 18 cornières `L60x60x6 … L150x100x10` : `a1 a2 t r` ; 8 ailes standard `[aile, rayon]` ; 5 épaisseurs `8 9 10 12 15` | BIB |
| Orientation `ORI_P` / `ORI_S` (grande aile sur principale / secondaire) | BIB (constantes) → ENT, MOT |
| Cornière Personnalisée `k1_u k2_u kt_u kr_u` (`a1 = max, a2 = min`) | MOT |
| Résistances `VgC` (1,27), `WC`, sections nettes, blocs, flexion ailes | MOT |

## 6. Profilés (`DB.profils`, l. 110–116)

| Élément | Destination |
|---|---|
| 90 profilés : HEA/HEB/HEM 100–1000 (24 × 3), IPE 80–600 (18) : `h b tw tf r` | BIB ; **test de croisement** avec `profiles_test.json` (31 communs) et contrôle d'aire → T |
| Profilé Personnalisé (`h b tw tf r` saisis) | MOT |
| Aire recalculée `A = 2b·tf + (h − 2tf)·tw + (4 − π)·r²` (`Asec_S`) | MOT |

## 7. Efforts

| Élément | Destination |
|---|---|
| `V_Ed` (valeur absolue), `N_Ed` (traction +), `H_Ed` (valeur absolue `H`), `M_Ed` (valeur absolue `Mabs`) | MOT |
| Efforts dérivés : `M_S`, `M_P`, `V/2` groupe P, `Mabs/2` ailes B, `V·(gh + ln)` grugeage, `V·(gh + e2b + p2)` 2ᵉ file | MOT |
| Cordons : `qz qx qn qw` (S et P), `fuw bww fvw Fw` | MOT |

## 8. Hypothèses

| Élément | Destination |
|---|---|
| `hypLines()` (l. 861–867) : articulation à la face de l'âme porteuse ; groupe S élastique ou cordons ; groupe P centré (facteur k_rot) ou excentricité eP ; poutre grugée ; interaction quadratique ; coefficients partiels | TXT (mot pour mot) → RAP |
| Onglet Méthode (l. 82–102) : modèle mécanique, cheminement, excentricités, trois natures, ce que l'outil ne vérifie pas, données de bibliothèque, sources | UI (onglet « Méthode », texte repris **intégralement**) |
| Longueur efficace des cordons = longueur totale (`Lw = Lc + 2·lh`) | MOT ; affichée comme **hypothèse** (UI, RAP) |
| Notes de champs : `lt_ok` (condition MSB §4.2.5), `k_rot` (0,8 MSB/SCI), `opt_exc`, `opt_blf` (§3.8), `expo` (4t + 40), `trou`, `cat`, `k_ser`, `mode_calc`, `fix_S` soudée 2 côtés | ENT → UI |
| Note « Non revérifiés ici : cisaillement global, flexion et déversement de la principale ; torsion ; flexion hors plan de l'âme sous NEd ; deux poutres en vis-à-vis » (l. 834) | UI (sous le groupe « Poutre principale ») |
| Légende des natures EC / COMP / INT (l. 828) | UI |

## 9. Vérifications — 38 (`ck()`, l. 337–382)

| Groupe | Clés | Nature | Condition `active` | Destination |
|---|---|---|---|---|
| Boulons (6) | `bv_S` | EC + COMP | `bolt_S` | MOT → RES |
| | `bv_P` | EC + COMP | `bolt_P` | |
| | `bt_P` | EC | `bolt_P && Ft_P > 0` | |
| | `bi_P` | EC | `bolt_P && Ft_P > 0` | |
| | `gl_S` | EC | `bolt_S && cat ≠ A` | |
| | `gl_P` | EC | `bolt_P && cat ≠ A` | |
| Cornières (13) | `pdB pdA` | EC + COMP | `bolt_S` / `bolt_P` | |
| | `cgB cgA flB flA` | COMP / COMP / COMP / **INT** | toujours | |
| | `cnB cbB tnB tbB` | COMP / EC / EC / EC | `bolt_S` (tn/tb : et `NEd > 0`) | |
| | `cnA cbA` | COMP / EC | `bolt_P` | |
| | `tsA` | EC + COMP | `bolt_P && NEd > 0` | |
| Poutre secondaire (10) | `pdS vgS vnS vbS` | EC+COMP / EC / COMP / EC | `bolt_S` (vgS : toujours) | |
| | `mN stN stD` | COMP | `cas_g > 0` | |
| | `m2` | COMP | `cas_g > 0 && bolt_S && n2_S = 2 && l_n > e2b + p2` | |
| | `tnS tbS` | EC | `bolt_S && NEd > 0` | |
| Poutre principale (3) | `pdP` | EC | `bolt_P` | |
| | `vlP` | COMP | toujours | |
| | `vnP` | COMP | `bolt_P` | |
| Soudures (6) | `wS waS wlS` | EC+COMP / EC / EC | `!bolt_S` | |
| | `wP waP wlP` | idem | `!bolt_P` | |
| Champs de chaque vérification | `key grp label Ed Rd unit ref nat active formula vals ess` + `eta` (Rd > 0 ? Ed/Rd : Ed > 0 ? 1e9 : 0), `ok` (η ≤ 1 + 1e-9) ; inactives : `eta = ok = null` | RES `Verification` ; UI (barre de taux, statut, détail dépliable) ; TXT ; RAP |
| Essentielles (18) : `bv_S bv_P pdB pdA cgB cnB cbB flB pdS vgS vnS vbS mN stN pdP vlP wS wP` | | | | TXT (version essentielle), RAP |
| Vérifications « sans objet » listées par groupe (l. 835) | | | | UI (ligne « Sans objet dans cette configuration : … ») |

## 10. Formules (chaîne `formula` de chaque `ck`, et formules de calcul)

| Élément | Destination |
|---|---|
| 38 formules générales (texte) : reprises **mot pour mot** | MOT (`formula`) → UI, TXT, RAP |
| Formules de calcul : `Fv_Rd = αv·fub·A/γM2`, `Ft_Rd = 0,9·fub·As/γM2`, `Bp_Rd = 0,6·π·dm·tp·fu/γM2`, `Fp_C = 0,7·fub·As`, `βLf`, répartition élastique (`M·1000·x/Ip`), pression diamétrale (`k1·αb·fu·d·t/γM2 · k_trou`), aires nettes `(n − 0,5)·d0`, blocs `0,5·fu·Ant/γM2n + fy·Anv/(√3·γM0)` et `fu·Ant/γM2n + …`, tronçon en T (modes 1–3), grugeage (`W_N`, `ρ`, `ln_max`), cordons (`q = V/Lw + M·(lh − xg)/Iw`, `fvw = fu/(√3·βw·γM2)`) | MOT (transcription ligne à ligne, mêmes noms) |
| Prédimensionnement : `ceil5`, `firstGE`, `e1 = ceil5(k_e1·d0)`, `p1`, `e2`, `Lc = 2e1 + (n − 1)p1`, `Mz`, `Fx = |N|/n + Mz·6/(p1·n(n + 1))`, `Fbv Fbh FP FbP`, taux `bS pdS bP pdP vt`, `score = n·100 + i + 1`, `etag`, `cleatFor` (7 épaisseurs nécessaires, `treq`, `tC`, `gA`, `legreq`, aile) | MOT (`predim`) |
| Formulaire → rapport : « Fv,Rd = αv·fub·A/γM2 (× 2 plans) ; F,Ed = √(Fz² + Fx²) … » etc. | RAP (mini-LaTeX `ndc_pdf.mathx` pour les essentielles, texte pour les autres) |

## 11. Résultats intermédiaires

| Élément | Destination |
|---|---|
| 190 scalaires de `R` (nombres, chaînes, booléens) — clés du JSON `scalaires` | MOT → T (parité, ≤ 1e-9) |
| Chaîne `vals` de chaque vérification (valeurs introduites et intermédiaires, formatées `F()`) | MOT → UI (détail), TXT, RAP ; T (égalité stricte) |
| Formatage fidèle : `F(x, n)` (= `toFixed` JS, virgule, `—` si non fini), `f0(x)` (0 ou 1 décimale, signe `−`), `pct(e, n)` (`> 999 %` au-delà de 9,99), conversion nombre → chaîne JS (`String(3)` = « 3 ») | FMT ; T (`toFixed` arrondi **demi-supérieur sur la valeur binaire exacte**, différent de Python) |

## 12. Résultats finaux (`compute` l. 383–389 ; `renderStatus` l. 790–796)

| Élément | Destination |
|---|---|
| `gov` (première η maximale parmi les actives), `eta_max`, `all_ok`, `geo_ok`, `dist_ok`, `verified`, `statut` (« ASSEMBLAGE VÉRIFIÉ / NON VÉRIFIÉ »), `reserve` (règle de grugeage NON VALIDE) | MOT → UI (bandeau permanent), TXT, RAP |
| Bandeau : statut + « – sous réserve des alertes » (si vérifié et `reserve` ou alertes), « Taux maximal x % : libellé », « | géométrie non valide », « | pinces ou entraxes non conformes », « Benchmark du module : VALIDÉ / À CONTRÔLER » | UI (bloc de synthèse toujours visible) |
| Prédimensionnement : `pd` = `boulon n e1 p1 e2 LC eta treq tC gA legreq leg rC found hav msg reqs rows pick` | MOT → UI (onglet Prédim) ; T (`predim.lignes` pour `defaut` et `predim_*`) |
| Bande « Mode prédimensionnement : M20, 3 rangées, L…, Lc = … mm. <msg> » | UI |

## 13. Avertissements géométriques (Tableau 3.3, `drow`, l. 192–210 ; informatifs)

| Élément | Destination |
|---|---|
| 11 lignes possibles : `S_e1c S_e2c S_e1b S_he S_e2b S_p1 S_p2 P_e1c P_e2c P_p1 P_p2` — `lab val min (k·d0) max (e : 4t + 40 si exposé, sinon — ; p : min(14t ; 200)) kmin fields dims rule ok alert` | MOT → RES `Pince` → UI (tableau « Pinces et entraxes », ligne NON OK cliquable), TXT, T |
| `dist_ok` | MOT |
| Recommandation `Lc ≥ 0,6·h` (`rec_L`) : mention « (Lc < 0,6·h : recommandation MSB Part 5 §4.1 non respectée) » | MOT → UI |
| Alertes informatives (non bloquantes) `MEd NEd HEd soude2` : message + explication (`why`) | MOT → RES `Alerte` → UI, TXT (« ATTENTION : … »), RAP |
| `val_N` : domaine de validité de la stabilité du grugeage (S235 → limites S275 : interprétation ; > S355 : NON VALIDE ; non maintenue : NON VALIDE) | MOT → UI (dans `vals` de `stN`), TXT |

## 14. Contrôles de configuration impossible (`AL`, l. 217–267)

| Identifiant | Condition | Champs / cotes / éléments | Destination |
|---|---|---|---|
| `S_e1bot`, `P_e1bot` | pince basse < 1,2·d0 | `LC_u n1 p1 e1` / `Lc e1 p1 e1bot` / `cleat bolts` ; `covers` `S_e1c`/`P_e1c` si e1 ≥ 1,2·d0 | MOT → RES `Alerte` |
| `S_e2`, `P_e2` | `e2a < 1,2·d0` | cornière, `g_h e2b n2 p2` / `gA n2 p2` ; `covers` `S_e2c`/`P_e2c` | MOT |
| `zc_top` | `z_C < dnt` ou `< tf + r` | `z_C d_nt` + profilé S / `zc dnt` / `cleat notchT` | MOT |
| `h_dispo` | `z_C + L_C > h − dnb` (ou `− (tf + r)`) | `LC_u z_C [d_nb]` + profilé S / `Lc zc [dnb]` / `cleat` | MOT |
| `dnt_min` | `d_top < tf_P + r_P && d_nt < tf_P + r_P − d_top` | `d_nt d_top` + profilé P / `dnt dtop` / `notchT flPt` | MOT |
| `ln_min` | `d_nt > 0 && l_n < (b_P − tw_P)/2 − g_h` | `l_n g_h` + profilé P / `ln gh` / `notchT flPt` | MOT |
| `gap_neg` | `h_P − d_top − h_S < 0` | profilés, `d_top` / `dtop` / `beamS flPb` | MOT |
| `dnb_min` | `0 ≤ gap < tf_P + r_P && d_nb < tf_P + r_P − gap` | `d_nb d_top l_n` + profilés / `dnb dtop` / `notchB flPb` | MOT |
| `P_web` | boulons P hors partie droite de l'âme principale | `d_top z_C e1P n1P p1P` + profilé P / `ztP e1P p1P zc dtop` / `boltsP flPt flPb` | MOT |
| `fillet` | trou dans le congé de la cornière (`gA − d0/2 < tc + rc` ou `gB − d0/2 < …`) | `gA_u` / `g_h e2b_u`, cornière, `boulon_u` / `gA` / `e2b gh` / `cleat bolts…` | MOT |
| `one_bolt` | `bolt_S && n_S = 1` | `n1S_u n2S_u` / — / `boltsS` | MOT |
| `d0_manq` | trou surdimensionné sans `d0_u` | `trou d0_u` | MOT |
| `cat_cl` | catégorie B/C avec `fub < 800` | `cat classe` | MOT |
| `dist_<id>` | générée pour chaque ligne du Tableau 3.3 NON OK non couverte (`covers`) | `fields dims` de la ligne | MOT |
| Chaque alerte : `id msg block fields dims elems why covers` ; explication chiffrée `why` construite avec `f0()` | MOT → RES `Alerte` → UI (alertes cliquables), TXT (« ALERTE BLOQUANTE : … »), RAP ; T (identifiant, bloquant, message, explication, champs, cotes, éléments) |
| `geo_ok` = aucune alerte bloquante | MOT |
| Cas d'erreur `compute` (profilé inconnu…) → bandeau « Données incomplètes » + message | UI |

## 15. Dessins (`DCDraw.elev`, `DCDraw.plan`, l. 589–677)

| Élément | Destination |
|---|---|
| **Élévation** : semelles P (`flPt`, `flPb`) et âme P ; polygone poutre S avec grugeages (`beamS`) et traits de semelles ; surlignage `notchT` / `notchB` ; cornière (`cleat`) + bande d'épaisseur `co2` ; boulons S (cercles + croix d'axes, `boltsS`) ou cordon S (trait épais avec retours) ; boulons P (tirets, `boltsP`) | SCH (`elevation()`) : même géométrie, même `viewBox`, classes SVG conservées |
| **Plan** : âme P, âme S (`beamS`), 2 cornières en L (`cleat`), boulons P (tirets) ou points de soudure P, points de soudure S, boulons S (tirets) | SCH (`plan()`) |
| Échelle : `fs = max((largeur + 260)/34, 9)` (élévation), `max((largeur + 210)/32, 8)` (plan) ; boîte `bb` | SCH |
| Noms de profilés (`S.name`) : poutre secondaire en haut, principale en bas (« Poutre principale/secondaire » si Personnalisé) | SCH |
| Cartouche (`L`, niveau ≥ 1) : cornières + nuance + Lc ; boulons + classe + d0 + groupes ; soudures ailes B / A ; excentricité z, MS, (eP, MP) ; grugeage + bras de levier | SCH (`cartouche()`), même texte |
| Styles (`<style>` l. 38–41) : `pp ps fl2 co co2 bo bp we wd dm ext dln tx cart dl hit ed calc hot hl cm` | SCH : **feuille de style embarquée dans le SVG**, couleurs alignées sur la charte de l'application (bleu accent, rouge `ko`) — pas de seconde charte |
| Mise en évidence rouge (`hot`) des éléments cités par une alerte | SCH (`hl.elems`) |
| Rendu PDF : même modèle de dessin, primitives ReportLab | RAP (`dessiner_elevation`, `dessiner_plan`) via `ndc_pdf` |

## 16. Cotations (`Sheet`, l. 537–587 ; appels `S.dim` / `S.tag`)

| Élément | Destination |
|---|---|
| Algorithme : rangées par côté `T/B/L/R`, `slot()` (créneau libre avec marge 0,4·fs, 40 rangées), `pos()`, `label()` (fond `hit`, `data-key`, `data-sym`, `tabindex`), `dim()` (libellé « sym = val » / « sym val » / déporté `out: lo|hi` si trop court), `name()`, `tag()` (étiquette pour valeur nulle), `finish()` (étendue, cartouche, `viewBox`, `aria-label`) | SCH (`Feuille`) transcrit **à l'identique** |
| 31 cotes : élévation `e2b p2S e2B lhS gh z ln bB e1S p1S e1botS Lc zc dnt dnb e1b he e1P p1P e1botP ztP dtop` + étiquettes `dnt dnb dtop aS` ; plan `e2b p2S gh bB gA p2P e2A p3 aP bA tc twS` | SCH |
| Niveaux `lvl` 0 / 1 / 2 (Vue simple / Cotations principales / Cotations complètes) ; une cote s'affiche si `lvl ≤ niveau` **ou** si elle est en alerte (`hot`) | SCH + UI (sélecteur de niveau) |
| Cotes modifiables (18, `key`) : `LC_u a_P a_S d_nb d_nt d_top e1P_u e1S_u e2b_u gA_u g_h l_n lh_S p1P_u p1S_u p2_P p2_S z_C` ; calculées (`calc`, violet italique) : `e2B z bB e1botS e1b he e1botP ztP e2A p3 bA tc twS` | SCH (`data-k` sur les modifiables) → CMP, panneau « Cotes » |
| Verrouillage en prédimensionnement (`locked = PDK`) | SCH |
| Niveau de `e1P/p1P` = 2 si identiques au groupe S (`lp`) | SCH |
| Cotes rouges (`hl.dims`) | SCH |

## 17. Interactions avec le dessin (`interface`, l. 743–777, 799–820, 927–928)

| Élément | Destination |
|---|---|
| Sélecteur de niveau (3 boutons) | UI |
| Clic (ou Entrée / Espace) sur une cote `.ed` → `openEditor` : champ de saisie sur place, Entrée valide, Échap annule, perte de focus valide ; écriture dans la source unique, synchronisation du formulaire, recalcul ; message « sym = v mm : géométrie et vérifications recalculées » | CMP (composant bidirectionnel : SVG + saisie sur place → `{clé, valeur}`) → UI (mise à jour de `session_state`, recalcul) ; **panneau « Cotes »** (repli et alternative) |
| `selectAlert(id)` : bascule `hl` (champs, cotes, éléments), rendu rouge, ouverture des groupes, défilement vers le premier champ ou vers le dessin ; chips « champ : valeur unité » → `jumpToField` ; « Voir sur le dessin » | UI (alerte cliquable → `asm_ui_alerte` ; marqueurs sur les champs ; groupes dépliés ; cotes/éléments rouges dans le SVG) |
| Ligne NON OK du Tableau 3.3 cliquable (`data-a`) → `selectAlert` | UI |
| Message « Blocage levé : géométrie valide » / « Ce blocage est levé ; il en reste d'autres » quand l'alerte sélectionnée disparaît | UI |
| `geoAfter` : bandeau « Fais glisser le dessin… », centrage sur la cote rouge | UI (dessin en pleine largeur de colonne ; conteneur défilant) |
| Légendes : « Élévation à l'échelle, cotes en mm. En violet italique : grandeurs calculées, non modifiables. » / « Vue en plan » ; note « Cotes sur fond jaune : touche la valeur… » / mode prédim | UI |
| Tableau « Paramètres retenus et excentricités » (13 lignes, l. 810–812) | UI |

## 18. Export texte (`buildTxt`, `dataLines`, `renderReport`, l. 849–887, 930–933)

| Élément | Destination |
|---|---|
| `dataLines()` : identification (si renseignée), poutre principale, secondaire (+ grugeage), cornières, boulons, soudures, efforts | TXT (`lignes_donnees`) → RAP |
| `hypLines()` | TXT (`lignes_hypotheses`) |
| `buildTxt(fullTxt)` : titre, données, hypothèses, vérifications (essentielles : `!ess && ok` omises ; complètes : toutes actives) numérotées avec formule / valeurs / Ed / Rd / η / OK / Réf. [nature], pinces, alertes, dimensionnante, taux max, conclusion | TXT (`construire_texte(R, complet)`) ; T (5 cas × 2 versions comparés à `exports_texte`, ligne à ligne) |
| Case « toutes les vérifications dans le texte » | UI (bascule essentiel / complet) |
| Téléchargement `.txt` (nom `assemblage_<repère>.txt`, BOM), copie du résumé, zone de texte sélectionnable | UI (`st.download_button`, `st.code`/zone de texte) |
| Rapport imprimable HTML (`renderReport`) + `@media print` A4 | remplacé par **RAP** (note PDF A4 paysage `ndc_pdf`) — même contenu : données, deux dessins, hypothèses, tableau des vérifications (Ed, Rd, η, statut, référence), pinces, alertes, conclusion |

## 19. Autres fonctions

| Élément | Destination |
|---|---|
| Enregistrer `.json` (objet des saisies, nom `assemblage_<repère>.json`) | UI (`💾 Enregistrer`, même barre d'outils que les modules béton) |
| Charger `.json` (fusion sur les défauts, clé par clé ; BOM toléré ; « Fichier illisible ») | UI (`📂 Ouvrir`) |
| Valeurs par défaut (`b-reset`) | UI (`🔄 Réinitialiser`, limité aux clés `asm_`) |
| Onglet **Benchmark** : `runBench()` → 4 exemples publiés (BM1 16 lignes, BM2 4, BM3 6, BM4 11 = 37), seuils 1 % / 3 %, statuts `OK / Écart expliqué / À expliquer / Écart > 3 % documenté / À vérifier`, écarts expliqués `E_ARR E_AIRE E_BLOC`, source `SRC`, `NOCOVER` (4), `VALID` VAL-A (13) + VAL-B (9) avec colonnes manuel / Excel / outil, statut global `VALIDÉ / À CONTRÔLER` ; bouton « Charger ces données dans l'outil » | BEN (`BENCH VALID NOCOVER SRC run_bench`) → UI (onglet « Benchmark ») ; T (37 valeurs publiées + 22 manuelles, statuts identiques) |
| Onglet **Méthode** (texte statique) | UI (onglet « Méthode ») |
| Onglet **Prédim** : solutions (≤ 9, triées : ok puis score, sinon η), carte cliquable (retenue / au-dessus du taux cible), détail de la solution retenue (8 lignes), épaisseur nécessaire par vérification (7 lignes), tableau de toutes les combinaisons géométriquement possibles avec « Appliquer », note de règle de choix | UI (onglet « Prédim ») ; `applySolution` + `solutionLabel` → MOT ; clic → écriture des 13 clés + `mode_calc` = VÉRIFICATION + affichage du schéma |
| Bandeau d'état permanent (statut, taux, dimensionnante, benchmark) et zone d'alertes sous le bandeau | UI (bloc de synthèse en tête de la colonne de droite, toujours visible) |
| Messages éphémères (`flash`) | UI (`st.toast`) |
| Mémoire des détails ouverts (`openCk`, `openSec`) et onglet courant | UI (`asm_ui_*`) |
| Persistance des sélections `n2S_u`/`n2P_u` en chaîne | MOT (`N()`) |
| Mise en page : formulaire à gauche (430 px, collant), droite (dessins, résultats) ; onglets ; version mobile | UI : `st.columns([2, 3])` de l'application, onglets `st.tabs` |
| Impression A4 | remplacée par le PDF `ndc_pdf` (RAP) |

---

## Comptes de contrôle (à retrouver en phase 6)

| Objet | HTML | Retrouvé |
|---|---|---|
| Entrées | 84 | 84 |
| Profilés / aciers / boulons / classes / cornières | 90 / 6 / 8 / 4 / 18 | 90 / 6 / 8 / 4 / 18 |
| Vérifications (EC / EC+COMP / COMP / INT) | 38 (17 / 8 / 12 / 1) | 38 |
| Essentielles | 18 | 18 |
| Lignes possibles du Tableau 3.3 | 11 | 11 |
| Alertes bloquantes nommées / informatives | 15 + `dist_…` / 4 | 15 + `dist_…` / 4 |
| Cotes (modifiables / calculées) | 31 (18 / 13) | 31 |
| Éléments de dessin nommés | 8 (`beamS boltsP boltsS cleat flPb flPt notchB notchT`) | 8 |
| Benchmark : valeurs publiées / manuelles | 37 / 22 | 37 / 22 |
| Exports texte de référence | 5 cas × 2 versions | 10 |
| Onglets | 7 | 7 (Données → colonne de gauche ; les 6 autres en onglets) |
