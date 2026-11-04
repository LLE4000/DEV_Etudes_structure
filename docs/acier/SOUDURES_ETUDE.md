# Soudures — étude de conception avant développement

*Écrite le 04/11/2026 à la demande du bureau : « quelque chose de très ergonomique qui
vérifie toutes les différentes soudures qu'on peut faire, avec un switch entre l'Eurocode
et la méthode ELS du bureau, des schémas où l'on comprend dans quel sens vont les efforts,
puis des feuilles de rapport très bien faites ».*

## 0 · La décision en une page

**Un seul moteur, tous les cas.** Une soudure d'angle ne sait rien du cas d'emploi : elle
reçoit une force par unité de longueur et la porte sur sa gorge. Tout ce qui change d'un cas
à l'autre, c'est la GÉOMÉTRIE du groupe de cordons et la façon dont les efforts de
l'assemblage se répartissent dessus. Le module est donc bâti sur un **groupe de cordons en
trois dimensions** (chaque cordon : deux points, une gorge, la direction de ses deux ailes),
une **répartition élastique** des six composantes d'effort (N, V_y, V_z, M_x, M_y, M_z)
sur ce groupe, et la **décomposition exacte** de la force obtenue en σ⊥, τ⊥, τ∥ sur la
gorge — la géométrie du cordon décide des signes, pas une règle de pouce. Les
configurations courantes (recouvrement, T, console, gousset, platine d'about de profilé,
tube sur platine, bout à bout) ne sont que des **constructeurs** de ce groupe à partir de
quelques dimensions ; un cas « cordons libres » reçoit n'importe quelle géométrie.

**Deux méthodes, un switch.** « Eurocode » = EN 1993-1-8 §4.5.3, efforts de calcul (ELU),
méthode directionnelle par défaut (simplifiée en option) avec γM2 = 1,25 et βw de la nuance.
« Méthode ELS (bureau) » = efforts de service sans coefficient, résultante par unité de
longueur comparée à k·a avec **k = 100 kg par mm de gorge et par cm de cordon** (défaut du
bureau), et, à côté, la valeur que l'Eurocode donnerait au même niveau de charge :
f_vw,d/1,5 = **138,6 N/mm² pour le S235** (141 kg/(mm·cm) avec 1 kg = 9,81 N, 139 avec
1 kg = 10 N), 174,5 pour le S355. Le bureau voit d'un coup d'œil la marge que prend sa règle.

**Des schémas qui montrent le sens des efforts.** Chaque cas est dessiné par le même code
qui calcule (source unique) : les pièces en gris, les cordons en orange, les efforts en
rouge avec leur point d'application et leur sens, un encart « la gorge » qui montre σ⊥,
τ⊥, τ∥ au point le plus chargé, et une carte des taux le long des cordons avec le point
gouvernant.

**Une note de deux pages**, comme les autres modules (kit `ndc_pdf`) : calcul, dessins,
tableaux, conclusion ; méthode et détail par cordon.

## 1 · Ce que l'application a déjà, et ce que le module réutilise

| Existant | Réemploi |
|---|---|
| `acier/resistances.py` : `fvw_d(fu, βw, γM2)`, `fu_bw_cordon` (acier de référence du cordon = la pièce la plus faible) | la méthode simplifiée et le βw du cordon |
| `acier/donnees/bibliotheques.json` : nuances S235 … S460 (fy, fu, βw) | source unique des nuances |
| `acier/donnees/profiles.json` (catalogue) | dimensions des profilés pour la platine d'about, le tube sur platine |
| kit `ndc_pdf` (palette Encre, `Doc`, `Frame`, `chip`, `draw_text`) et `plancher/charges/peintre.py` (SVG → reportlab) | la note PDF |
| `modules/navigation.py`, charte des écrans (barre d'outils, onglets à libellé fixe, carte résultat, clés préfixées) | l'écran |
| `plancher/charges` comme patron (état `cas` multiples, JSON, harnais, captures) | l'architecture |

## 2 · Le modèle mécanique

### 2.1 Le cordon

Un cordon d'angle est un segment P₁P₂ (mm) de gorge a, posé dans le dièdre de deux faces
perpendiculaires. Il porte la direction de son axe **e_a**, et les directions **u_A**, **u_B**
de ses deux ailes (de la racine vers chaque face, perpendiculaires à e_a et entre elles).
Le plan de gorge passe par la racine et coupe l'angle droit en deux : il contient l'axe et la
bissectrice **(u_A + u_B)/√2**. Sa normale est donc **n_t = (u_A − u_B)/√2**, et la direction
dans le plan de gorge perpendiculaire à l'axe est **t_⊥ = (u_A + u_B)/√2**.

Une force **f** par unité de longueur (N/mm) transmise par le cordon donne sur sa gorge,
d'aire a par unité de longueur :

- σ⊥ = (f · n_t)/a ; τ⊥ = (f · t_⊥)/a ; τ∥ = (f · e_a)/a.

C'est la décomposition de la figure 4.5 de l'EN 1993-1-8, sans règle de pouce : une force
normale au plan de joint (arrachement d'un T) et une force dans le plan de joint
perpendiculaire au cordon (cisaillement frontal) donnent toutes deux |σ⊥| = |τ⊥| = f/(a√2),
mais leurs signes se combinent selon la position du cordon — ce qu'un calcul « à la
résultante » ne voit pas.

### 2.2 Le groupe

Les cordons d'un assemblage forment un groupe, traité comme des lignes d'aire a·dl
(méthode élastique classique) :

- A_w = Σ a·l ; centre de gravité G des gorges ; I_x, I_y, I_xy, J = I_x + I_y (par rapport à
  G, axes du plan de joint) ;
- les efforts (N normal au plan de joint, V_x, V_y dans le plan, M_x, M_y de flexion,
  M_z de torsion) sont donnés en un point de référence et transportés en G ;
- force par unité de longueur en un point P d'un cordon de gorge a :
  **f = a·[ (V_x, V_y, N)/A_w + (M_x·(y − y_G) − M_y·(x − x_G))/I … ·e_z + M_z/J·(e_z × GP) ]**
  (flexion : contrainte normale linéaire ; torsion : contrainte tangentielle
  proportionnelle au rayon, perpendiculaire au rayon) ;
- f est évaluée aux deux extrémités et au milieu de chaque cordon (elle y varie
  linéairement : le maximum est à une extrémité).

Pour des gorges différentes dans un même groupe, les inerties sont pondérées par a et la
force par unité de longueur de chaque cordon est proportionnelle à sa gorge : un cordon
deux fois plus épais prend deux fois plus de force, à contrainte égale.

### 2.3 Les critères

**Eurocode, méthode directionnelle (§4.5.3.2)** : √(σ⊥² + 3·(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et
σ⊥ ≤ 0,9·fu/γM2, avec fu et βw de l'acier de référence du cordon (la pièce la plus faible,
Tab. 4.1 : 0,8 / 0,85 / 0,9 / 1,0 / 1,0 pour S235 / S275 / S355 / S420 / S460), γM2 = 1,25
(ANB belge).

**Eurocode, méthode simplifiée (§4.5.3.3)** : |f| ≤ F_w,Rd = f_vw,d·a, f_vw,d = fu/(√3·βw·γM2).
Toujours du côté sûr par rapport à la directionnelle (elle traite toute force comme du
cisaillement) ; proposée en option, parce que les bureaux la connaissent.

**Méthode ELS (bureau)** : efforts de service (coefficients 1), résultante |f| ≤ k·a avec
k en kg/(mm·cm), converti en N/mm² par k·g/10 (g = 9,81 N/kg par défaut ; 10 en option) :
100 kg/(mm·cm) ↔ 98,1 N/mm². À côté, la **valeur équivalente de l'Eurocode** au niveau de
service : f_vw,d/γ_F avec γ_F = 1,5 (rapport usuel ELU/ELS d'une charge de chantier) —
138,6 N/mm² pour le S235, 155,8 pour le S275, 174,5 pour le S355 — et un bouton « prendre la
valeur Eurocode ». La règle du bureau est donc **plus sûre de 28 %** sur du S235 ; sur du
S355 elle laisse 43 % de marge, ce que la page dit en clair.

**Bout à bout** (§4.7) : pleine pénétration → la résistance est celle de la pièce la plus
faible, aucun calcul de cordon (la page le dit et vérifie la pièce : σ = N/(b·t) ≤ fy/γM0) ;
pénétration partielle → cordon d'angle de gorge = profondeur de pénétration.

**Longueurs** : longueur efficace l_eff = l − 2a (§4.5.1, cordon sans retour d'angle ;
option désactivable si les retours sont exécutés) ; l_eff ≥ max(30 mm ; 6a) sinon le cordon
ne compte pas ; gorge a ≥ 3 mm ; assemblage long par recouvrement (§4.11) :
βLw,1 = 1,2 − 0,2·L_j/(150a) ≤ 1 quand L_j > 150a.

### 2.4 Exemple chiffré (console, § 4 cas C3)

Console en plat de 10 mm soudée sur un poteau par deux cordons verticaux de 200 mm, gorge
5 mm, S235 ; V_Ed = 60 kN à 150 mm de la face du poteau.

| Grandeur | Valeur |
|---|---|
| M = V·e | 9,0 kN·m |
| I_w = 2·a·l³/12 | 6,67·10⁶ mm⁴ |
| f_n en tête de cordon = M·(l/2)·a/I_w | 675 N/mm |
| f_a = V/(2l) | 150 N/mm |
| σ⊥ = τ⊥ = f_n/(a√2) ; τ∥ = f_a/a | 95,5 ; 95,5 ; 30,0 N/mm² |
| directionnelle : √(σ⊥² + 3(τ⊥² + τ∥²)) = 197,9 ≤ 360 | **η = 0,55** |
| simplifiée : |f| = 691,5 N/mm ≤ 207,8·5 = 1 039 N/mm | η = 0,67 |
| ELS bureau (V_ser = 40 kN) : 461 N/mm ≤ 98,1·5 = 490 N/mm | η = 0,94 |
| ELS avec la valeur Eurocode (138,6 N/mm²) : 461 ≤ 693 | η = 0,67 |

La même console est « à 55 % » selon l'Eurocode et « à 94 % » selon la règle du bureau :
c'est exactement ce que le switch doit rendre visible.

## 3 · Catalogue des configurations

| Code | Cas | Ce qu'on saisit | Groupe construit | Efforts |
|---|---|---|---|---|
| C1 | Recouvrement | largeurs, épaisseurs, recouvrement, cordons latéraux et frontaux (oui/non), a | 2 cordons latéraux (τ∥) ± 1 ou 2 frontaux (σ⊥ = τ⊥) | N dans le plan ± V transversal |
| C2 | Jonction en T | épaisseur de la pièce rapportée, longueur, une ou deux faces, jonction (angle a / pénétration partielle p ± renfort a_r / pleine) | 1 ou 2 cordons le long de la jonction, ailes sur chaque pièce — ou les trois lignes de gorge d'une pleine pénétration (la section t·l de la pièce) | N (arrachement), V_a (le long), V_p (en travers), M (flexion du T) |
| C3 | Console fléchie | hauteur du plat, épaisseur, excentricité, jonction (angle / partielle / pleine), support (aile d'un poteau du catalogue ou plat) | 2 cordons verticaux (ou la section du plat en pleine pénétration) | V à l'excentricité e → M = V·e + V |
| C4 | Gousset en torsion | plat posé à plat, cordons sur 1 à 4 côtés, a | périmètre choisi | V dans le plan à l'excentricité → torsion + cisaillement |
| C5 | Platine d'about de profilé | profilé du catalogue (IPE, HE…), épaisseur de platine, jonction des semelles et de l'âme (angle a / partielle p ± renfort / pleine), semelles une ou deux faces | semelles (4 cordons, ou leurs lignes de pleine pénétration) + âme (2) — un groupe peut mélanger | N, V_z, M_y (+ V_y, M_z) |
| C6 | Tube sur platine | tube du catalogue (CHS, SHS, RHS), épaisseur de platine, jonction (angle a / partielle / pleine), pourtour continu ou par n tronçons de l mm | périmètre (cercle discrétisé, rectangle à coins) ou ses tronçons (le premier centré en haut) ; pleine : la paroi t sur la ligne moyenne | N, V, M, T |
| C7 | Bout à bout | largeur, épaisseur, pénétration (pleine / partielle p par face) | pleine : une soudure de gorge t vérifiée au métal de base ; partielle : une soudure de gorge p par face ; p₁ + p₂ ≥ t → complète | N, V, M |
| C8 | Cordons libres | liste d'éléments : segments (x₁, y₁, x₂, y₂) ou arcs (centre, rayon, angles — 0 → 360 : cercle), gorge, côté de la pièce rapportée, continus ou par tronçons | tel quel (arcs en cordes de 15°) | les six composantes au point de référence |

**Soudure discontinue (04/11/2026).** Toutes les configurations acceptent « n tronçons de
l mm » sur leurs cordons droits et leurs pourtours : aux deux bouts et régulièrement entre
sur un cordon droit, régulièrement à partir du haut sur un pourtour ; chaque tronçon est
une chaîne ouverte raccourcie de a à chaque bout, soumise à l − 2a ≥ max(30 mm ; 6a) ;
l'espacement est rappelé avec la limite de l'EN 1993-1-8 §4.3.2.2 (≤ min(200 mm ; 12 t en
compression ; 16 t en traction)) et marqué « DÉPASSÉ » le cas échéant (`trace.py`).

Chaque constructeur rend aussi les éléments du dessin (pièces, cotes) : le dessin est une
vue du même objet.

## 4 · Les schémas

Principes, les mêmes pour tous les cas — et, depuis le 05/11/2026, le MÊME KIT que les
dessins d'assemblage (`acier/soudures/schemas.py` sur
`acier.assemblages.poutre_poutre.doubles_cornieres.schemas`) : le premier jeu du 04/11,
dessiné à la main en pixels, « n'était pas assez technique » — des carrés et des
rectangles à côté des attaches poutre–poteau ; il a été refait sur le kit (feuille de
cotation, primitives, sections réelles, hachures, ruptures, symboles) et le peintre
vectoriel de la note est celui des assemblages (`rapport.peindre`).

- **deux ou trois vues** par cas (§3) : la vue du plan de joint montre les cordons tels
  que calculés (bandes ocre à leur taille a·√2, chaîne par chaîne, tronçons, exclus en
  interrompu), une coupe ou une élévation montre la jonction (triangles des cordons
  d'angle, chanfreins remplis d'une pénétration partielle avec son renfort, chanfrein K
  d'une pleine pénétration) ; pièces au trait fort, sections hachurées, lignes de rupture
  là où la pièce continue, arêtes cachées, profilés RÉELS du catalogue (section en I avec
  ses congés, couronne d'un tube, aile d'un HEB en support), noms des pièces, symbole de
  soudure EN 22553 (« a 5 », « s 6 + a 3 », « pleine pénétration ») ;
- **cotes** : la feuille à rangées du kit, posée sur l'étendue RÉELLE du corps (efforts
  compris) — une cote ne tombe jamais sur une flèche, rien n'est coupé
  (`tests/test_soudures_schemas.py`) ;
- **efforts en rouge** (`#9C3341`) : dans le plan de joint, flèches V_x / V_y, ⊙ / ⊗ pour N,
  vecteurs ⇒ pour M_x / M_y, arc pour M_z, les valeurs dans un bloc sous le dessin ; en
  coupe ou en élévation, l'effort porte sa valeur ; le sens du dessin EST le sens positif
  de la saisie (une valeur négative retourne la flèche) ;
- **◉ point gouvernant** avec son taux, G centre des gorges, R point de référence ;
- **encart « la gorge »** (`dessin.py`) : la section du cordon le plus chargé — cordon
  d'angle, pénétration partielle ou pleine pénétration — avec sa gorge, et les trois
  composantes σ⊥ (normale à la gorge, sortante en traction), τ⊥, τ∥ avec leurs valeurs —
  la figure 4.5 de l'EN 1993-1-8 appliquée au cas ;
- **carte des taux** à côté de l'encart, et une **légende** d'une ligne sous le dessin.

## 5 · L'écran

Barre d'outils : 🏠 Accueil · [cas ▾] · ＋ Cas · 📋 · 🗑️ · 💾 Enregistrer · 📂 Ouvrir ·
📄 Note. Sous la barre, le **switch** en segmented control, bien visible :
**« Eurocode (ELU) » | « Méthode ELS (bureau) »** — il change le libellé des efforts
(« Efforts de calcul ELU » / « Efforts de service »), la résistance affichée et le texte de
la carte résultat.

Colonne gauche : le dessin (vue principale + encart + carte des taux), les vues au choix
(Élévation / Plan / Gorge). Colonne droite : la carte **RÉSULTAT** (pastille, η max, cordon
et point gouvernants, critère, méthode, phrase d'une ligne), puis cinq onglets à libellé
fixe : **1 · Configuration** (cas, pièces, nuances, profilé du catalogue), **2 · Cordons**
(gorges, longueurs, longueur efficace, retours, βLw), **3 · Efforts** (six composantes,
point de référence, sens rappelé par le dessin), **4 · Méthode** (directionnelle /
simplifiée, γM2, k du bureau et son équivalent Eurocode, g, γ_F), **5 · Détail** (tableau
par cordon et par point : f, σ⊥, τ⊥, τ∥, σ_c, η). En bas, « Méthode et sources ».

Trois formats (R1) ; clés `sdr_` ; cas multiples ; fichier JSON ; AppTest ; captures.

## 6 · La note (feuille de rapport)

Deux pages A4 paysage, kit `ndc_pdf` : page 1 CALCUL — bandeau (projet, cas, méthode
choisie, pastille, η max), dessin principal et encart de gorge peints depuis le SVG,
hypothèses (méthode, acier de référence, coefficients, longueur efficace), tableau des
cordons (position, longueur, gorge, l_eff), tableau des vérifications (par cordon : f, σ⊥,
τ⊥, τ∥, σ_c, résistance, η), conclusion ; page 2 MÉTHODE ET DÉTAIL — les formules de la
méthode employée, la comparaison Eurocode / bureau, la carte des taux, les références.

## 7 · Architecture et lots

```
acier/soudures/
  cordon.py        Cordon, décomposition sur la gorge (§2.1)
  groupe.py        Groupe : centre, inerties, répartition élastique des six composantes (§2.2)
  criteres.py      Eurocode directionnelle / simplifiée, ELS bureau, bout à bout, longueurs (§2.3)
  configurations.py  C1 … C8 : dimensions → groupe + éléments du dessin ; types de jonction (jonction_T)
  etat.py          cas, défauts, JSON, construction depuis l'état
  schemas.py       les schémas techniques sur le kit des assemblages (deux ou trois vues par cas)
  dessin.py        l'encart « la gorge », la carte des taux
  formules.py      les formules de la méthode choisie (substitutions testables)
  methode.py       textes de méthode (écran et note), références
  note.py          la note PDF (la méthode choisie seulement)
modules/soudures.py          l'écran
tests/test_soudures_moteur.py, test_soudures_schemas.py, test_soudures_formules.py, test_soudures_ecran.py, test_soudures_note.py
tools/captures_soudures.js
docs/acier/SOUDURES.md       document de référence (§9 : couverture)
```

| Lot | Contenu | Preuve |
|---|---|---|
| SW1 · moteur | cordon, groupe, critères, configurations, état | tests analytiques : recouvrement (τ∥ = N/(2·a·l)), frontal (σ⊥ = τ⊥ = N/(a·l·√2)), T en traction, console (3M/l²), gousset (polaire), platine (semelles = M/h), simplifiée ≥ directionnelle, conversions k |
| SW2 · écran | dessins techniques, écran, switch, cas, JSON | harnais AppTest (50), captures trois formats (`tools/captures_soudures.js`), images |
| SW2a · tronçons | soudures discontinues, arcs, éléments libres (`trace.py`) | tests analytiques (78 au total) |
| SW2b · liaison | `modules/liaison.py` : les widgets suivent l'état (Soudures et Charges sur plancher) | contrôle navigateur dans les captures (état « tube »), harnais des deux écrans |
| SW3 · note | note deux pages, peintre SVG partagé (`ndc_pdf/svg.py`), doc `docs/acier/SOUDURES.md` | tests PDF (43), batterie, commit, push |
| SW4a · jonctions | angle / pénétration partielle ± renfort / pleine pénétration (la pièce, métal de base, trois lignes de Simpson) sur C2, C3, C5, C6, C7 ; gorge de pleine résistance ; poteau du catalogue en support ; signe de σ⊥ (traction > 0) | tests analytiques (103 au total : N/(l·t), 6M/(l·t²) exacts, §4.7.3, I exact des semelles, fy/γ_F) |
| SW4b · schémas | `schemas.py` sur le kit des assemblages : vues, profilés réels, symboles EN 22553, cotes hors de tout, efforts, ◉ | `test_soudures_schemas.py` (32 : rien de coupé, contenus, peintre), captures trois formats |
| SW4c · formules | `formules.py` : la vérification au point gouvernant écrite comme à la main, la note n'imprime que la méthode choisie | `test_soudures_formules.py` (27 : chaque substitution redonne le moteur), tests de la note |
| SW4d · couverture | `docs/acier/SOUDURES.md` §9 : couvert / pas couvert, la cornière reconstituée = C2 pleine | — |

## 8 · Points à trancher par le bureau (défauts retenus en attendant)

| | Question | Défaut retenu |
|---|---|---|
| P-1 | k de la méthode ELS | 100 kg/(mm·cm), modifiable ; l'équivalent Eurocode affiché à côté |
| P-2 | conversion kg → N | g = 9,81 N/kg ; option 10 (la pratique qui donne « 138 ») |
| P-3 | rapport ELU/ELS pour l'équivalent Eurocode | γ_F = 1,5, modifiable |
| P-4 | γM2 | 1,25 (ANB belge), modifiable dans Méthode |
| P-5 | longueur efficace | l − 2a par défaut (§4.5.1) ; « retours d'angle exécutés » rend l entière |
| P-6 | méthode Eurocode par défaut | directionnelle (la plus juste) ; simplifiée en option |
| P-7 | la méthode ELS compare la résultante (comme la simplifiée) | oui — c'est la règle du bureau ; la directionnelle reste lisible dans Détail |
| P-8 | fatigue, aciers > S460, soudures sur acier inoxydable | hors V1, dits dans « ce que l'outil ne fait pas » — les cordons discontinus, d'abord exclus, sont entrés le 04/11/2026 à la demande du bureau (tronçons sur toutes les configurations) |
| P-9 | position des tronçons | aux deux bouts et régulièrement entre sur un cordon droit (les bouts doivent être soudés) ; régulièrement à partir du haut sur un pourtour ; l'espacement §4.3.2.2 est rappelé, pas imposé |
| P-10 | l'écran garde-t-il la valeur du navigateur ou celle de l'état ? | celle de l'ÉTAT, toujours : `modules/liaison.py` réécrit les widgets depuis l'état à chaque changement hors widget (défaut trouvé par les captures : C3 → C6 affichait l'ancienne gorge) |
| P-11 | pleine pénétration en méthode ELS : à quoi comparer la pièce ? | fy/γ_F (156,7 N/mm² pour le S235 avec γ_F = 1,5), le pendant de k·a pour les cordons ; à l'ELU fy/γM0 = fy (§4.7.1) |
| P-12 | signe de σ⊥ | σ⊥ > 0 quand la pièce rapportée s'arrache (traction sur la gorge), < 0 quand elle comprime le cordon ; l'encart dit « traction » / « compression » d'après ce signe (la convention du 04/11 était inversée : un arrachement s'affichait « compression ») |
| P-13 | pénétration partielle des deux faces avec p₁ + p₂ ≥ t | en T : traitée comme pleine si c = t − 2p ≤ min(t/5 ; 3 mm) (§4.7.3) ; bout à bout coplanaire : pénétration complète, la pièce gouverne (§4.7.1) |

## 9 · Sources

EN 1993-1-8:2005 §4 (4.3 cordons, 4.5.1 longueur efficace, 4.5.3.2 méthode directionnelle,
4.5.3.3 méthode simplifiée, 4.7 soudures bout à bout, 4.11 assemblages longs, Tab. 4.1 βw) ;
NBN EN 1993-1-8 ANB (γM2 = 1,25) ; pratique du bureau (k = 100 kg/(mm·cm)) ; méthode
élastique des groupes de cordons (traités classiques d'assemblages : répartition linéaire
en flexion, polaire en torsion).
