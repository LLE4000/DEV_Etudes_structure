# Soudures — document de référence

*Livré le 04/11/2026. Page « Soudures » (tuile Acier), moteur `acier/soudures/`,
étude de conception `docs/acier/SOUDURES_ETUDE.md`.*

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `acier/soudures/cordon.py` | le cordon : segment de gorge a dans un dièdre, et la décomposition EXACTE d'une force par unité de longueur sur sa gorge (σ⊥, τ⊥, τ∥, EN 1993-1-8 fig. 4.5) ; cordon d'angle, bout à bout à pénétration partielle, longueur efficace, βLw |
| `acier/soudures/trace.py` | le tracé : chemins, arcs, chaînes (polyligne → segments de calcul), **tronçons** (soudures discontinues), espacement §4.3.2.2 |
| `acier/soudures/groupe.py` | le groupe de cordons : A_w, G, I_x, I_y, I_xy, J des gorges ; répartition élastique des six composantes d'effort |
| `acier/soudures/criteres.py` | les critères : Eurocode directionnelle (§4.5.3.2) et simplifiée (§4.5.3.3), méthode ELS du bureau, pièce à pleine pénétration, k équivalent Eurocode |
| `acier/soudures/configurations.py` | huit configurations C1 … C8 : dimensions → groupe de cordons, efforts au bon point et dans le bon sens, pièces du dessin |
| `acier/soudures/etat.py` | les cas, les défauts, le JSON, la vérification complète (points, cordons, chaînes, comparaison des méthodes, messages) |
| `acier/soudures/dessin.py` | les dessins techniques SVG (vues, coupes, élévations, encart « la gorge », carte des taux) |
| `acier/soudures/methode.py` | les textes de méthode (onglet Méthode et page 2 de la note) |
| `acier/soudures/note.py` | la note de calcul PDF en deux pages |
| `ndc_pdf/svg.py` | le peintre SVG → reportlab partagé par toutes les notes (hachures, arcs, dégradés) |
| `modules/soudures.py` | l'écran ; `modules/liaison.py` : la liaison état → widgets partagée avec « Charges sur plancher » |
| `tests/test_soudures_moteur.py`, `test_soudures_ecran.py`, `test_soudures_note.py` | les tests (78 + 50 + 44) ; `tools/captures_soudures.js` : captures et mesures aux trois formats |

## 2 · Le modèle (une seule règle pour tous les cas)

**Le cordon** est un segment de gorge a posé dans le dièdre de deux faces
perpendiculaires. Il porte l'axe e_a et les directions u_A, u_B de ses deux ailes (de la
racine vers chaque face). Le plan de gorge passe par la racine et coupe l'angle droit en
deux : sa normale est n_t = (u_A − u_B)/√2, la direction en travers dans la gorge est
t_⊥ = (u_A + u_B)/√2. Une force f (N/mm) sur le cordon donne **σ⊥ = f·n_t/a,
τ⊥ = f·t_⊥/a, τ∥ = f·e_a/a** — c'est la figure 4.5 appliquée sans règle de pouce : un
effort en travers dans le plan de joint et un arrachement donnent tous deux
|σ⊥| = |τ⊥| = f/(a√2), mais ne se combinent pas de la même façon.

**Le groupe** traite les cordons comme des lignes d'aire a·dl : A_w, centre G,
I_x, I_y, I_xy, J. Les six composantes données au point de référence sont transportées en
G ; f = a·[F/A_w + σ_flexion·e_z + (M_z/J)·e_z × GP], σ_flexion linéaire (axes
quelconques, I_xy compris). f est évaluée au début, au milieu et à la fin de la longueur
efficace de chaque cordon ; le point le plus chargé gouverne.

**Longueurs.** l_eff = l − 2a sans retours d'angle (§4.5.1) ; un cordon ne compte que si
l_eff ≥ max(30 mm ; 6a) et a ≥ 3 mm (§4.5.2). Un arc ou un pourtour est une CHAÎNE :
découpée en segments de calcul (cordes de 15°), raccourcie de a à chaque bout si elle est
ouverte, entière si elle est fermée. Recouvrement long : βLw,1 (§4.11).

**Soudure discontinue (par tronçons).** Toute configuration accepte « n tronçons de l mm » :
aux deux bouts et régulièrement entre sur un cordon droit, régulièrement à partir du haut
sur un pourtour ou un cercle. Chaque tronçon est une chaîne ouverte (l − 2a) ; l'espacement
est rappelé avec la limite de l'EN 1993-1-8 §4.3.2.2 (≤ min(200 mm ; 12 t en compression ;
16 t en traction)) et marqué « DÉPASSÉ » le cas échéant. On n'est donc pas obligé de
souder partout — un CHS sur platine par quatre tronçons se calcule comme le reste.

**Bout à bout.** Pleine pénétration : la pièce la plus faible gouverne (§4.7.1), vérifiée
élastiquement √(σ² + 3τ²) ≤ fy/γM0. Pénétration partielle : cordon de gorge = pénétration,
gorge perpendiculaire à l'effort normal (ailes virtuelles à 45°).

## 3 · Les deux méthodes et le switch

| | Eurocode (ELU) | Méthode ELS du bureau |
|---|---|---|
| efforts | de calcul (γG·G + γQ·Q) | de service, sans coefficient |
| critère | directionnelle : √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et σ⊥ ≤ 0,9·fu/γM2 ; simplifiée en option : \|f\| ≤ f_vw,d·a | résultante \|f\| ≤ k·a avec k = 100 kg par mm de gorge et par cm de cordon (98,1 N/mm² à 9,81 N/kg) |
| acier | fu et βw de la pièce la plus faible (Tab. 4.1), γM2 = 1,25 (ANB) | — |
| équivalent affiché | le taux selon le bureau avec les mêmes efforts | f_vw,d/γ_F avec γ_F = 1,5 : 138,6 N/mm² = 141 kg/(mm·cm) pour le S235, et un bouton pour prendre cette valeur |

La carte RÉSULTAT donne toujours les quatre taux (directionnelle, simplifiée, bureau,
bureau avec k Eurocode) pour les mêmes efforts, la méthode choisie marquée ◀ : la même
console est « à 61 % » selon l'Eurocode et « à 156 % » selon la règle du bureau avec des
efforts ELU — c'est exactement ce que le switch rend visible. Les points à trancher par le
bureau (k, g, γ_F, γM2, longueur efficace, critère par défaut) sont dans l'étude, §8.

## 4 · Les configurations

| Code | Cas | Vue | Tronçons |
|---|---|---|---|
| C1 | Recouvrement : latéraux ± frontaux | dessus (cordon caché en pointillé) | latéraux |
| C2 | Jonction en T : un ou deux cordons | coupe A-A hachurée + plan | oui |
| C3 | Console fléchie : V à e, N | élévation + vue de dessus ×2,5 des cordons | oui |
| C4 | Gousset en torsion : 1 à 4 côtés, point d'application | face (plat en porte-à-faux sur l'appui) | oui |
| C5 | Platine d'about de profilé I/H du catalogue : semelles (une ou deux faces), âme | face (profilé coupé, hachuré) + élévation | semelles, âme |
| C6 | Tube CHS / SHS / RHS du catalogue sur platine | face + élévation | pourtour |
| C7 | Bout à bout : pleine ou partielle | plan + coupe de la soudure (×3) | — |
| C8 | Cordons libres : segments et arcs, continus ou par tronçons, six composantes au point R | plan de joint | par élément |

## 5 · Les dessins (style dessin technique)

Pièces au trait fort, sections hachurées (deux sens pour deux pièces, le cordon coupé en
orange hachuré), cotes fines à lignes d'attache et pointes, axes en trait mixte, trait de
coupe A-A sur le plan, arête cachée en interrompu. Les cordons sont colorés par leur taux
(vert ≤ 70 %, ocre jusqu'à 100 %, rouge au-delà), le point gouvernant marqué ◉, les
cordons exclus en pointillé gris. Les efforts sont en rouge dans le sens saisi : flèche
droite, flèche courbe (moment dans le plan), ⇒ vecteur moment, ⊙ / ⊗ pour l'effort normal
au dessin ; « tend le haut » nomme la semelle tendue. L'encart « LA GORGE » montre la
section du cordon gouvernant, le plan de gorge en interrompu, σ⊥ (sortante en traction,
entrante en compression), τ⊥, τ∥ et les valeurs. Le SVG porte un viewBox et s'étire à sa
colonne ; la note le peint tel quel (`ndc_pdf/svg.py`, hachures réelles).

## 6 · L'écran

Barre d'outils (Accueil, cas, ＋, ⧉, 🗑️, Enregistrer, Ouvrir, 📄 Note, nom du cas) ; le
SWITCH « Eurocode (ELU) | Méthode ELS (bureau) » avec une ligne qui dit ce que chaque
position change ; à gauche le choix de configuration, le dessin, l'encart de la gorge et la
carte des taux ; à droite la carte RÉSULTAT puis cinq onglets à libellé fixe :
1 · Configuration (dimensions, profilé du catalogue, tronçons, nuances ; en C8 un tableau
éditable d'éléments), 2 · Cordons (chaînes, l_eff, propriétés du groupe), 3 · Efforts (sens
du dessin, nature selon la méthode), 4 · Méthode (critère, γM2, k, g, γ_F, bouton
« prendre la valeur Eurocode », identification, textes de méthode), 5 · Détail (par point :
f, σ⊥, τ⊥, τ∥, σ_c, η selon chaque méthode). Trois formats (R1), clés `sdr_`, JSON.

**Liaison état → widgets (`modules/liaison.py`).** Un widget à clé garde son identité d'un
rendu à l'autre et le navigateur garde sa dernière valeur tant qu'on ne la lui réassigne
pas : supprimer la clé ne suffit pas (défaut trouvé par les captures du 04/11/2026 : après
C3 → C6 la gorge affichait 5 alors que le calcul prenait 4 ; sur « Charges sur plancher »,
après « ＋ Cas », la portée affichait celle du cas précédent). Chaque widget est enregistré
avec son chemin dans le cas ; à tout changement d'état hors widget, les clés sont RÉÉCRITES
depuis l'état (dans un callback directement, depuis le corps du script au rendu suivant).
Le harnais AppTest ne voit pas ce défaut (pas de navigateur) : c'est le contrôle Playwright
de `tools/captures_soudures.js` (état « tube ») qui le garde.

## 7 · La note

Deux pages A4 paysage (`acier/soudures/note.py`, kit `ndc_pdf`). Page 1 CALCUL : bandeau
(statut, η max, cordon et point gouvernants, critère, les quatre méthodes), configuration,
efforts, méthode ; le dessin et l'encart de la gorge peints depuis le même SVG, hypothèses
(messages du moteur) ; tableaux des cordons, des efforts (avec le sens), des propriétés du
groupe, des vérifications par point (le pire point de chaque cordon quand ils ne tiennent
pas tous), la comparaison des méthodes, la conclusion. Page 2 MÉTHODE ET DÉTAIL : les
textes de `methode.py`, la carte des taux, le détail complet, le cartouche (cas,
configuration, méthode, acier, date · indice, échelle du dessin).

## 8 · Les preuves

- `tests/test_soudures_moteur.py` (78) : décomposition contre la figure 4.5, console de
  l'étude (σ⊥ = τ⊥ = 95,5, τ∥ = 30, η 0,550 / 0,665 / 0,94), torsion d'un cordon seul,
  recouvrement, T, platine (M/(h·b)), tube (N/périmètre, Bredt), bout à bout, βLw,
  conversions k ; chemins, chaînes, tronçons (positions, exclusion, refus), arcs ; CHS et
  SHS par tronçons, console par tronçons contre l'inertie à la main, espacement dépassé,
  cercle et demi-cercle libres, ancien fichier converti.
- `tests/test_soudures_ecran.py` (50) : navigation, switch, source unique, huit
  configurations, cordons libres, cas multiples, JSON, Note, étanchéité des clés.
- `tests/test_soudures_note.py` (44) : treize notes sans débordement, contenus, dessin
  peint, page 2, méthode ELS, peintre partagé.
- `tools/captures_soudures.js` : cinq états × quatre largeurs, aucune exception ni
  débordement, dessin et encart visibles, pastilles attendues, gorge qui suit l'état.

![Écran PC](images/soudures_pc.png)

![Tablette, tube par tronçons](images/soudures_tablette_tube.png)

![Note, page 1](images/soudures_note_p1.png)

*Téléphone : `images/soudures_telephone_els.png` ; note page 2 : `images/soudures_note_p2.png`.*
