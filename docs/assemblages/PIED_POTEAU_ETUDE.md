# Pieds de poteaux sur béton sous N + Mx + My — étude de méthode

*Étude du 30/10/2026, avant tout développement. Elle répond aux questions A à G du
bureau : méthode sans éléments finis, résistance de la platine, profilés, équations
et références, risques non conservatifs, statut normatif, périmètre d'une première
version. Les nombres de l'exemple viennent d'un script jetable (hors dépôt) ; ils
illustrent, ils ne valident rien.*

**Convention de citation.** Les numéros d'articles et d'équations cités sont ceux
dont je suis sûr. Là où je ne le suis pas, je cite la section par son titre, et la
relecture sur le texte imprimé fait partie du travail de V1 (§ 8). Aucun numéro
n'est reconstitué de mémoire pour « faire propre ».

---

## 0 · La décision en une page

**Faisable, sans maillage, avec une note justifiable — à condition d'accepter une
méthode à DEUX NIVEAUX et un périmètre V1 net.**

1. **La répartition sous la platine** (pressions, résultante, efforts d'ancrage) se
   calcule par une *analyse de section* : platine rigide, lit de béton linéaire
   **sans traction**, ancrages tendus seulement, le tout posé sur l'**aire efficace**
   d'EN 1993-1-8 §6.2.5 (bande de largeur *c* autour du profilé) et non sur la
   platine entière. Trois inconnues (le plan de pénétration), trois équations
   (N, Mx, My), intégrales **exactes** sur des polygones. C'est la méthode
   qu'EN 1992-4 impose pour dériver les efforts dans les ancrages (platine rigide,
   théorie de l'élasticité) : elle est **codifiée pour cet usage**.
2. **La résistance du pied** (ELU) se vérifie par un modèle **plastique de borne
   inférieure** : bloc de contrainte *f*jd côté comprimé, ancrages plafonnés par
   leur résistance de tronçon en T. Le domaine de résistance (N, Mx, My) est
   **convexe**, ce qui donne un résultat exploitable immédiatement : l'interaction
   **linéaire** des capacités uniaxiales calculées « à la Table 6.7 » d'EN 1993-1-8
   est une borne **rigoureusement conservative** — c'est le **niveau 1**, celui de
   la note de calcul V1. Le domaine exact (**niveau 2**) est affiché à titre
   d'information et sera promu après campagne de validation.
3. **La platine** se vérifie côté comprimé par la bande *c* (élastique, EC3) et
   côté tendu par les tronçons en T (Table 6.2, longueurs efficaces Table 6.6,
   effet de levier selon *L*b\*). Le **boulon de coin au-delà de l'about d'aile**
   d'un profilé en I n'a pas de longueur efficace codifiée : modèle de lignes de
   rupture + repli conservatif + validation.
4. **Profilés V1** : HEA/HEB/HEM/IPE (couverts par EC3), RHS/SHS (composants EC3,
   assemblage par analogie « paroi = aile », appuyé sur CIDECT/AISC ; statut
   « modèle d'ingénieur »). CHS en V1.1 (aire annulaire, lignes de rupture
   circulaires). Barres pleines en V2. Cornières, T, U : hors périmètre.
5. **Ce que la note V1 peut affirmer « conforme »** : N seul ; N + M uniaxial ;
   N + Mx + My par interaction linéaire ; ancrages selon EN 1992-4 ; soudures
   selon la méthode directionnelle ; cisaillement par frottement + ancrages.
   **Ce qu'elle doit présenter comme modèle d'ingénieur** : le domaine biaxial
   exact, les boulons de coin, les profils creux.

---

## 1 · Ce que les normes applicables en Belgique couvrent — et ce qu'elles ne couvrent pas

### 1.1 Textes

| Texte | Ce qu'on y prend |
|---|---|
| **NBN EN 1993-1-8:2005 + AC:2009 + ANB** | §6.2.5 tronçon en T comprimé : *F*C,Rd = *f*jd·*b*eff·*l*eff, *c* = *t*·√(*f*y/(3·*f*jd·γM0)), *f*jd = βj·*F*Rdu/(*b*eff·*l*eff), βj = 2/3 (éq. (6.4) à (6.6)) · §6.2.6.9 à 6.2.6.12 (platine fléchie en compression / en traction, boulons d'ancrage tendus, béton comprimé y compris scellement) · §6.2.8 pieds de poteaux : 6.2.8.2 (N seul, trois tronçons en T sans recouvrement), 6.2.8.3 (N + M, bras de levier *z*C / *z*T, Table 6.7) · §6.3.4 rigidité en rotation (coefficients *k*13, *k*15, *k*16 de la Table 6.11, Table 6.12) · §5.2.2.5 classification des pieds par rigidité · §6.2.2 cisaillement d'un pied : frottement *F*f,Rd = *C*f,d·*N*c,Ed avec *C*f,d = 0,20 (mortier sable-ciment), tiges *F*2,vb,Rd = αb·*f*ub·*A*s/γM2, αb = 0,44 − 0,0003·*f*yb · §3 boulons d'ancrage (nuances, *f*yb ≤ 640 N/mm² s'ils reprennent du cisaillement) · Table 6.2 (modes 1-2-3 du tronçon en T, mode 1 « sans effet de levier » si *L*b > *L*b\* = 8,8·*m*³·*A*s/(Σ*l*eff,1·*t*f³)) · Tables 6.4 à 6.6 (longueurs efficaces) · §4.5.3.2 méthode directionnelle des soudures, βw Table 4.1. γM0 = 1,00 (ANB). |
| **NBN EN 1992-1-1 + ANB** | §6.7 charges concentrées : *F*Rdu = *A*c0·*f*cd·√(*A*c1/*A*c0) ≤ 3,0·*f*cd·*A*c0 (éq. (6.63)) ; *f*cd = αcc·*f*ck/γC — **αcc à lire dans l'ANB du bureau** (0,85 est l'hypothèse de l'exemple ; 1,0 donnerait *f*jd 18 % plus haut). |
| **NBN EN 1992-4:2018 + ANB** | Section 6 « dérivation des efforts dans les fixations » : platine **rigide**, théorie de l'élasticité, béton sans traction — c'est le modèle élastique du § 2.3 · §7.2.1 traction (acier, arrachement, **cône de béton** avec ψec,N pour l'excentrement dans les deux directions et ψM,N pour l'effet favorable de la compression de platine, fendage, éclatement latéral pour les tiges à tête près d'un bord) · §7.2.2 cisaillement (acier avec ou sans bras de levier, rupture de bord, pry-out) · §7.2.3 interaction traction–cisaillement · coefficients γM (Table 4.1). Ancrages post-installés : valeurs de l'**ETA** du produit. |
| **EN 10025-2, EN ISO 898-1, EN 1090-2** | Nuances (fy selon l'épaisseur : 345 N/mm² pour S355 entre 16 et 40 mm), classes de boulons, transmission par contact si l'about est dressé (à spécifier à l'exécution). |

### 1.2 Ce que ces textes **ne disent pas**

- **Flexion biaxiale** : la Table 6.7 est écrite pour un moment dans un plan. Rien
  n'interdit de l'appliquer deux fois, mais la **combinaison** n'est pas écrite.
- **Boulons au-delà de l'about d'aile** (les quatre coins d'une platine large sur un
  profilé en I) : les longueurs efficaces des Tables 6.4 à 6.6 supposent un boulon
  **adjacent à une aile** (ou entre aile et raidisseur). Le coin n'est pas décrit.
- **Profils creux** : les composants (béton comprimé, platine fléchie, tiges) sont
  génériques, mais §6.2.8 est écrit pour les profilés en I. L'assemblage pour RHS
  et CHS n'est pas décrit.
- **« Platine suffisamment rigide »** : EN 1992-4 l'exige pour sa dérivation
  élastique des efforts d'ancrage, sans donner le critère. Il faut en choisir un et
  l'écrire (§ 2.6).
- **Deuxième génération** : EN 1993-1-8:2024 étend la méthode des composants ; à ma
  connaissance les pieds restent traités en flexion plane. Sans ANB belge publiée,
  elle n'est pas applicable au bureau ; à suivre, pas à coder.

### 1.3 Littérature de référence (reconnue, non normative)

- **HERON vol. 53 (2008) n° 1/2**, numéro consacré aux pieds de poteaux, documents
  de base d'EN 1993-1-8 : Wald, Sokol, Jaspart, *Base plate in bending and anchor
  bolts in tension* ; Steenhuis, Wald, Sokol, Stark, *Concrete in compression and
  base plate in bending* ; Gresnigt, Romeijn, Wald, Steenhuis, *Column bases in
  shear and normal force* ; Wald, Sokol, Steenhuis, Jaspart, *Component method for
  steel column bases*.
- Jaspart & Weynand, *Design of Joints in Steel and Composite Structures*, ECCS
  Eurocode Design Manual (2016) — exemples uniaxiaux.
- SCI P398, *Joints in Steel Construction — Moment-Resisting Joints to Eurocode 3*
  (2013) — pieds de poteaux encastrés, exemple traité.
- Access Steel, NCCI SN037 (pieds articulés) et SN043 (pieds encastrés) — à relire.
- CIDECT Design Guide 9 (Kurobane, Packer, Wardenier, Yeomans, 2004), chapitre
  « column base plates » pour RHS et CHS ; AISC Design Guide 24 (2010) pour les
  platines de pied de profils creux ; AISC Design Guide 1 (Fisher & Kloiber, 2006)
  pour la méthode des porte-à-faux et le bloc rectangulaire en flexion plane.
- Lignes de rupture de platines sur profils creux : Packer, Bruno, Birkemoe,
  *Limit analysis of bolted RHS flange plate joints*, J. Struct. Eng. 115(9), 1989 ;
  Wheeler, Clarke, Hancock, Murray, *Design model for bolted moment end plate
  connections joining rectangular hollow sections*, J. Struct. Eng. 124(2), 1998 ;
  modèle d'Igarashi pour les brides circulaires de CHS (repris dans CIDECT DG1).
- Interaction biaxiale : Bresler (1960), contour de charge (*M*x/*M*Rx)^α +
  (*M*y/*M*Ry)^α ≤ 1 ; EN 1992-1-1 §5.8.9 éq. (5.39) en est la forme codifiée pour
  le béton armé. Avec α = 1 c'est la borne convexe (§ 2.4).

---

## 2 · Réponse A — la méthode pour N + Mx + My sans éléments finis

### 2.1 Le principe : deux modèles, comme en béton armé

Une section de béton armé se calcule avec un modèle **élastique** pour les
contraintes de service et un modèle **plastique** (bloc rectangulaire) pour la
résistance ultime. Le pied de poteau se traite de la même façon, et c'est aussi
le partage que font les deux normes concernées :

| Modèle | Ce qu'il donne | Qui l'impose |
|---|---|---|
| **Élastique** — platine rigide, lit sans traction, ancrages élastiques | la répartition des pressions, la position de la résultante, l'effort dans **chaque** ancrage, le soulèvement, la rigidité *S*j | **EN 1992-4 section 6** pour les ancrages ; EN 1993-1-8 §6.3.4 pour *S*j |
| **Plastique** — bloc *f*jd, tronçons en T à leur résistance | la résistance du pied sous (N, Mx, My) | **EN 1993-1-8 §6.2.8.3** (Table 6.7) en flexion plane ; extension biaxiale par le théorème statique |

Les deux reposent sur la **même géométrie** : l'aire efficace.

### 2.2 L'aire efficace (EN 1993-1-8 §6.2.5)

La platine ne porte pas sur toute sa surface : seule la bande de largeur *c* autour
des parois du profilé travaille, parce qu'au-delà la platine est trop souple pour
mobiliser le béton :

> *c* = *t*p · √( *f*y / (3 · *f*jd · γM0) )      (éq. 6.5)

> *f*jd = βj · *F*Rdu / (*b*eff · *l*eff), βj = 2/3 si le mortier a *f*c ≥ 0,2 × celle
> du massif et une épaisseur ≤ 0,2 × la plus petite dimension de la platine ;
> *F*Rdu d'EN 1992-1-1 éq. (6.63), *A*c0 étant l'aire du tronçon en T.

Le « 3 » vient du moment **élastique** *t*p²·*f*y/6 : la bande comprimée reste
élastique sous *f*jd. L'aire efficace *A*eff est l'union des rectangles
« paroi + *c* de chaque côté », découpée par le contour de la platine. Pour un
profilé en I : deux rectangles d'aile et un d'âme ; pour un RHS : les quatre
parois ; pour un CHS : un anneau. En compression pure, *N*Rd = *f*jd·*A*eff
(§6.2.8.2, tronçons sans recouvrement).

### 2.3 Modèle élastique : la répartition (questions 1, 2, 3 du bureau)

**Hypothèses.** Platine rigide (plan indéformable) ; béton = lit de ressorts
indépendants de raideur *k*c, **sans traction**, actif sur *A*eff seulement ;
chaque ancrage = ressort de raideur *K*b, **en traction seulement** ; aucune
précontrainte.

**Inconnues.** Le plan de pénétration *w*(x, y) = *a* + *b*x·x + *b*y·y
(pénétration > 0 = compression du béton).

**Équations.** Avec *p* = *k*c·*w* là où *w* > 0 et *T*i = *K*b·(−*w*(x\_i, y\_i)) là
où *w* < 0 :

> ∫ *p* dA − Σ *T*i = *N*
> ∫ *p*·y dA − Σ *T*i·y\_i = *M*x
> ∫ *p*·x dA − Σ *T*i·x\_i = *M*y

Trois équations non linéaires (le domaine d'intégration dépend des inconnues),
**mais** homogènes de degré 1 : seule la direction de (*a*, *b*x, *b*y) change les
zones, l'amplitude est linéaire. L'axe neutre est la droite *w* = 0.

**Sans maillage.** La zone comprimée est l'intersection de chaque rectangle de
*A*eff avec le demi-plan *w* ≥ 0 : un polygone, obtenu par un découpage
Sutherland–Hodgman à un seul bord. Les intégrales ∫1, ∫x, ∫y, ∫x², ∫xy, ∫y² sur
un polygone sont **exactes** (formules de Green sur les sommets). Newton converge
en quelques itérations ; toute la résolution tient en quelques dizaines de
lignes et ne dépend d'aucune bibliothèque.

**Raideurs.** On prend celles qu'EN 1993-1-8 §6.3.4 utilise pour *S*j, pour rester
sur une base normative : béton *k*13 = *E*c·√(*b*eff·*l*eff)/(1,275·*E*) d'où
*k*c = *E*·*k*13/(*b*eff·*l*eff) ; tige *k*16 = 2,0·*A*s/*L*b (1,6 si effet de
levier), *K*b = *E*·*k*16, avec *L*b = 8·d + scellement + *t*p + rondelle + ½ écrou
(Table 6.2). La position de l'axe neutre dépend du rapport *k*c/*K*b — comme en
béton armé du rapport *E*s/*E*c — et la campagne de validation doit en mesurer la
sensibilité (§ 8).

**Ce que le modèle sort.** La résultante *C* et sa position (centre de gravité de
la pression), la pression maximale (au sommet le plus éloigné de l'axe neutre),
la zone de contact, l'effort de chaque ancrage, et pour les ancrages les
grandeurs qu'EN 1992-4 demande ensuite : la résultante de traction du groupe et
son excentrement dans les deux directions (ψec,N).

**Statut.** Pour les ancrages, c'est la méthode d'EN 1992-4 section 6 : codifiée.
Pour la vérification du béton comprimé et de la platine, c'est un modèle élastique
— conservatif pour le béton (la pointe de pression est comparée à *f*jd), et
exigé pour vérifier que la platine reste élastique sous la répartition réelle
(§ 3.1).

### 2.4 Modèle plastique : la résistance

**Hypothèses.** Béton à *f*jd sur la partie comprimée de *A*eff (bloc
rectangulaire), 0 ailleurs ; ancrages en traction avec 0 ≤ *T*i ≤ *F*T,Rd,i (la
résistance du tronçon en T de la rangée ou du boulon, § 3.2) ; équilibre.

**Théorème statique.** Tout champ (pressions, efforts d'ancrage) qui équilibre
(N, Mx, My) et respecte ces bornes est une **borne inférieure** de la charge
ultime, pourvu que les composants soient ductiles. L'ensemble des charges
équilibrables est **convexe** (contraintes bornées dans un intervalle, équilibre
linéaire). Deux conséquences exploitables :

1. **Interaction linéaire rigoureuse.** À *N* donné, si *M*x,Rd(*N*) et
   *M*y,Rd(*N*) sont deux points admissibles du domaine (les capacités uniaxiales
   calculées avec le modèle de la Table 6.7, résultante au centre de l'aile
   comprimée et tirants à leur résistance), alors tout point du segment qui les
   joint est admissible : **Mx,Ed/Mx,Rd + My,Ed/My,Rd ≤ 1 est une vérification
   conservative, démontrée**, et chacune de ses deux capacités est codifiée.
2. **Domaine exact par balayage.** La frontière du domaine s'obtient en faisant
   tourner la normale de l'axe neutre (angle θ) : pour chaque θ, la profondeur
   du bloc est fixée par l'équation de *N* (bissection, fonction monotone), et
   l'on lit le couple (Mx, My) résistant. Une centaine de valeurs de θ dessinent
   la frontière ; le taux de travail est le rapport des normes de (Mx,Ed, My,Ed)
   et du point de frontière dans la même direction. Pas de maillage non plus.

**Répartition côté tendu.** EN 1993-1-8 §6.2.8.3 n'a qu'une rangée par côté ;
avec deux ou trois tiges du côté tendu à des distances différentes de l'axe
neutre, deux choix : plastique (toutes à *F*T,Rd) ou linéaire (proportionnelles à
leur distance, la plus éloignée à *F*T,Rd). **Je retiens la répartition linéaire
plafonnée** : les modes de rupture du béton d'ancrage (cône, arrachement) sont
fragiles et n'autorisent pas la redistribution plastique entre tiges ; c'est
aussi l'hypothèse d'EN 1992-4. Elle est une borne inférieure comme l'autre.

**Statut.** En flexion plane, ce modèle contient celui de la Table 6.7 : il le
reproduit quand on bloque la résultante au centre de l'aile, et le dépasse
légèrement quand il laisse le bloc se placer au bord de l'aire efficace (+ 11 %
sur l'axe fort, + 22 % sur l'axe faible dans l'exemple du § 10 — l'écart est
entièrement expliqué par la position de la résultante, et il va dans le sens que
les documents de base d'EC3 décrivent). En flexion biaxiale, c'est une
**extension cohérente de la méthode des composants, non codifiée** : c'est le
**niveau 2**.

### 2.5 Les deux niveaux de verdict

| | Niveau 1 — « conforme » | Niveau 2 — « modèle d'ingénieur » |
|---|---|---|
| Résistance en flexion | interaction linéaire des capacités uniaxiales à la Table 6.7 | domaine exact par balayage |
| Efforts d'ancrage, répartition | modèle élastique (EN 1992-4) | idem |
| Platine, soudures, béton, ancrages | composants codifiés | idem |
| Dans la note V1 | **le verdict** | une ligne d'information : « marge réelle estimée » |
| Promotion | — | après la campagne du § 8 |

Le coût du niveau 1 est mesuré dans l'exemple : taux 0,48 au lieu de 0,34. Il
est le plus lourd quand les deux moments sont comparables ; pour une platine
dominée par un axe, il est faible. Pour un **CHS** il serait absurde (la
symétrie de révolution rend le domaine quasi circulaire, l'interaction linéaire
perd 30 %) : pour les CHS le niveau 1 vérifie le moment résultant √(Mx² + My²)
contre la capacité minimale sur l'angle, calculée par le balayage — c'est
rigoureux parce que le balayage ne fait là que tourner le cas uniaxial.

### 2.6 Ce que fait la flexibilité de la platine (question 5)

- **Côté comprimé.** Une platine souple concentre la pression sous les parties
  raides (ailes, âme) : la résultante se rapproche du poteau, le bras de levier
  diminue, l'effort d'ancrage augmente. Calculer sur la platine **entière** supposée
  rigide — ce que font certains logiciels d'ancrage — surestime le bras de levier
  dès que le débord dépasse *c* : **non conservatif pour les ancrages**. L'aire
  efficace règle le problème : elle ne compte que ce que la platine peut
  mobiliser en restant élastique. Dans l'exemple, le débord est de 105 mm pour
  *c* = 46 mm ; le lit est donc posé sur *A*eff.
- **Côté tendu.** Une platine mince plie, s'appuie sur le béton au-delà du boulon
  et crée un **effet de levier** *Q* qui s'ajoute à la traction de la tige :
  c'est le mode 2 du tronçon en T. Avec des tiges d'ancrage longues (*L*b >
  *L*b\*), la platine n'arrive pas à mobiliser le levier : mode 1 « sans effet de
  levier », *F*T,1-2,Rd = 2·*M*pl,1,Rd/*m*. EN 1993-1-8 Table 6.2 couvre les deux.
- **Critère de rigidité pour EN 1992-4.** Je propose d'écrire le critère que la
  norme laisse ouvert : la platine est « suffisamment rigide » si, sous la
  répartition élastique calculée, son moment de porte-à-faux reste **élastique**
  (*m*Ed ≤ *t*p²·*f*y/(6·γM0)) sur chaque bande comprimée, et si la rangée tendue
  est en mode 3 ou en mode 1 sans effet de levier. Sinon l'outil refuse de
  conclure sur les ancrages et propose d'épaissir.
- **Rigidité globale.** *S*j (§6.3.4) baisse avec *t*p ; un pied déclaré encastré
  dans le modèle global doit satisfaire §5.2.2.5 (*S*j,ini ≥ 30·*E*·*I*c/*L*c
  hors contreventement). L'outil le calcule et l'écrit dans la note.

---

## 3 · Réponse B — la résistance de la platine elle-même

### 3.1 Côté comprimé

Par construction de *c*, la bande travaille au moment élastique sous *f*jd. Deux
vérifications complémentaires, toutes deux analytiques :

- sous la **répartition élastique réelle** (§ 2.3), moment de porte-à-faux par
  unité de longueur au droit de chaque paroi, *m*Ed = ∫ *p*·s dA / *l* (intégrale
  exacte sur le polygone au-delà de la paroi) ≤ *t*p²·*f*y/(6·γM0) ;
- **coin comprimé** (au-delà de l'aile **et** de l'about) : EC3 inclut le carré
  *c* × *c* du coin dans l'aire efficace ; on le garde, c'est codifié, et l'on
  vérifie en plus la bande diagonale du coin sous la pression élastique.

### 3.2 Côté tendu — tronçon en T (EN 1993-1-8 §6.2.4, Table 6.2)

> *M*pl,Rd = 0,25 · Σ*l*eff · *t*p² · *f*y / γM0
> mode 1 : *F*T,1,Rd = 4·*M*pl,1,Rd/*m* (avec levier) ou 2·*M*pl,1,Rd/*m* (sans levier, *L*b > *L*b\*)
> mode 2 : *F*T,2,Rd = (2·*M*pl,2,Rd + *n*·Σ*F*t,Rd)/(*m* + *n*), *n* = min(*e*, 1,25·*m*)
> mode 3 : *F*T,3,Rd = Σ*F*t,Rd, *F*t,Rd = 0,9·*f*ub·*A*s/γM2 (Table 3.4)
> *m* = distance boulon–paroi − 0,8·*a*·√2 (cordon d'angle)

Longueurs efficaces selon la position du boulon :

| Position du boulon | Modèle | Statut |
|---|---|---|
| entre les ailes, près de l'âme | tronçon en T adjacent à l'âme, motifs Table 6.6 (rangée adjacente à un raidisseur : l'aile joue le raidisseur) | codifié |
| à l'extérieur de l'aile, **dans la largeur d'aile** | motifs « rangée extérieure à l'aile tendue » (2π*m*x, π*m*x + *w*, 4*m*x + 1,25*e*x, …) | codifié, c'est le cas de §6.2.6.10 et des documents HERON |
| **au-delà de l'about d'aile** (coin) | lignes de rupture : charnière le long de la face d'aile sur une longueur ℓ vers l'intérieur + ligne inclinée issue du coin d'aile ; capacité minimale sur (ℓ, angle) | **non codifié** — § 3.3 |

### 3.3 Le boulon de coin d'un profilé en I

C'est le cas de l'exemple du bureau (tiges à 200 mm de l'axe, ailes à 150 et
145 mm). Le plat autour du boulon n'est tenu que par la face extérieure d'aile
(jusqu'à l'about) et par l'épaisseur de l'about lui-même, 14 mm : il n'y a pas de
« T ». Ce que je propose, dans l'ordre :

1. **Relire HERON 2008 (Wald, Sokol, Jaspart)** : les longueurs efficaces des
   platines de pied y sont établies pour les boulons extérieurs à l'aile ; si le
   coin y est traité, on le code tel quel avec sa référence.
2. Sinon, **mécanisme de lignes de rupture** à deux volets — un volet tournant
   autour de la face d'aile (charnière de longueur ℓ), un volet de coin articulé
   sur le premier le long du prolongement de l'about — minimisé sur ℓ et sur
   l'inclinaison ; **repli conservatif** si le minimum est douteux : tronçon en T
   contre la face d'aile avec Σ*l*eff plafonnée à la longueur d'aile disponible
   entre le boulon et l'âme.
3. **Validation** : quelques géométries en éléments finis **hors outil** (une fois,
   par un logiciel que le bureau possède) ou CBFEM publié ; la fiche de validation
   est versée au dépôt. Tant qu'elle n'existe pas, la note porte la mention
   « modèle d'ingénieur ».

Et une recommandation de conception que l'outil affichera : pour un profilé en
I, placer les tiges **dans la largeur d'aile** (cas codifié) ou raidir le coin.
Pour un **RHS/SHS**, au contraire, le coin est le bon endroit : le boulon y est
adjacent à **deux** parois perpendiculaires, et le mécanisme est celui de
Packer (1989) / Wheeler (1998), bien documenté.

### 3.4 Les autres mécanismes à vérifier (question 6)

| Composant | Vérification | Référence | V1 |
|---|---|---|---|
| Béton sous la platine | *f*jd sur l'aire efficace comprimée (bloc) ; pointe élastique ≤ *f*jd | EN 1993-1-8 §6.2.5, EN 1992-1-1 §6.7 | ✔ |
| Mortier de scellement | βj = 2/3 : *f*c,mortier ≥ 0,2·*f*c,massif, épaisseur ≤ 0,2·min(*b*p, *l*p) | §6.2.5 | ✔ (contrôle des hypothèses) |
| Platine comprimée | bande *c* ; moment élastique sous répartition réelle | §6.2.5, § 3.1 | ✔ |
| Platine tendue | tronçon en T modes 1-2-3, *L*b\* | §6.2.6.10, Table 6.2 | ✔ |
| Tiges — acier | traction *F*t,Rd, cisaillement *F*2,vb,Rd (αb réduit), interaction | §3.6.1 Table 3.4, §6.2.2 | ✔ |
| Tiges — béton | cône, arrachement, fendage, éclatement, bord, pry-out, interaction | EN 1992-4 §7.2.1–7.2.3 + ETA | ✔ |
| Aile et âme du poteau comprimées | *F*c,fc,Rd = *M*c,Rd/(*h* − *t*f) par analogie avec §6.2.6.7 | EN 1993-1-8 | ✔ |
| Soudures poteau–platine | méthode directionnelle : √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ *f*u/(βw·γM2), σ⊥ ≤ 0,9·*f*u/γM2 ; groupe de cordons traité « en ligne » (aire de gorge, moments quadratiques des lignes) → N, Vx, Vy, Mx, My, Mz sans maillage | §4.5.3.2, Table 4.1 | ✔ |
| Cisaillement du pied | frottement 0,20·*N*c,Ed + tiges ; mortier ≤ d/2 et ≥ 30 N/mm² sinon flexion de tige avec bras de levier ; répartition polaire pour Mz | §6.2.2, EN 1992-4 §6 et §7.2.2 | ✔ (Mz : polaire, conservatif) |
| Bêche | hors V1 | — | ✘ |
| Rigidité, classification | *S*j, Table 6.12, §5.2.2.5 | §6.3.4 | ✔ |

---

## 4 · Réponse C — profilés réellement compatibles

| Profilé | Aire efficace | Côté tendu | Statut | Version |
|---|---|---|---|---|
| **HEA, HEB, HEM, IPE** | 3 rectangles (ailes, âme) + *c* | tronçons en T codifiés ; coin = § 3.3 | **codifié** (hors coin) | V1 |
| **RHS, SHS** | 4 parois + *c*, intérieur et extérieur | paroi = aile ; coin appuyé sur deux parois (Packer, Wheeler) | composants EC3 + assemblage par analogie (CIDECT DG9, AISC DG24) : **modèle d'ingénieur documenté** | V1 |
| **CHS** | anneau de largeur 2*c* + *t* | lignes de rupture circulaires (Igarashi / CIDECT DG1) ; balayage angulaire exact | idem | V1.1 |
| **Barres pleines** (rond, carré, rectangle) | section + bande *c* | comme CHS/SHS, soudure à pleine pénétration à prévoir | idem, peu de littérature | V2 |
| **L, T, U** | union de rectangles (la méthode marche) | asymétrie forte, torsion | rare en poteau | hors périmètre |

Ce que le bureau demandait — « plusieurs méthodes appropriées plutôt qu'une
méthode universelle » — se traduit ainsi : **un seul solveur** (aire efficace
polygonale + analyse de section), **plusieurs bibliothèques de lignes de
rupture** côté tendu, une par famille de parois.

---

## 5 · Réponse D — équations fondamentales et références (récapitulatif)

| N° | Équation | Référence | Statut |
|---|---|---|---|
| 1 | *c* = *t*p·√(*f*y/(3·*f*jd·γM0)) | EN 1993-1-8 §6.2.5 éq. (6.5) | codifié |
| 2 | *f*jd = βj·*F*Rdu/(*b*eff·*l*eff), *F*Rdu = *A*c0·*f*cd·√(*A*c1/*A*c0) ≤ 3·*f*cd·*A*c0 | §6.2.5 éq. (6.6) ; EN 1992-1-1 éq. (6.63) | codifié |
| 3 | équilibre plan : ∫*p* dA − Σ*T* = N ; ∫*p*·y − Σ*T*·y = Mx ; ∫*p*·x − Σ*T*·x = My, *p* = *k*c·*w*⁺, *T* = *K*b·(−*w*)⁺ | EN 1992-4 section 6 (platine rigide, élasticité) | codifié pour les ancrages |
| 4 | *k*c = *E*·*k*13/(*b*eff·*l*eff), *k*13 = *E*c·√(*b*eff·*l*eff)/(1,275·*E*) ; *K*b = *E*·*k*16, *k*16 = 2,0·*A*s/*L*b | EN 1993-1-8 Table 6.11 | codifié (usage *S*j) |
| 5 | domaine plastique : bloc *f*jd sur *A*eff ∩ demi-plan, 0 ≤ *T*i ≤ *F*T,Rd,i, convexité | théorème statique ; Table 6.7 en flexion plane | extension non codifiée |
| 6 | Mx,Ed/Mx,Rd(N) + My,Ed/My,Rd(N) ≤ 1 | convexité (5) ; forme Bresler α = 1, EN 1992-1-1 (5.39) | borne démontrée sur composants codifiés |
| 7 | uniaxial : *T* = (M − N·*z*C)/(*z*T + *z*C) ≤ *F*T,Rd ; *C* = N + *T* ≤ *F*C,Rd | modèle de la Table 6.7 (§6.2.8.3) | codifié |
| 8 | tronçon en T modes 1-2-3, *L*b\* = 8,8·*m*³·*A*s/(Σ*l*eff,1·*t*f³) | Table 6.2, Table 6.6 | codifié |
| 9 | *F*t,Rd = 0,9·*f*ub·*A*s/γM2 ; *F*2,vb,Rd = αb·*f*ub·*A*s/γM2, αb = 0,44 − 0,0003·*f*yb | Table 3.4 ; §6.2.2 | codifié |
| 10 | *F*f,Rd = 0,20·*N*c,Ed | §6.2.2 | codifié |
| 11 | *N*Rk,c = *N*⁰Rk,c·(*A*c,N/*A*⁰c,N)·ψs,N·ψre,N·ψec,N·ψM,N, *N*⁰Rk,c = *k*1·√*f*ck·*h*ef^1,5 ; *N*Rk,p = *k*2·*A*h·*f*ck ; *V*Rk,c, *V*Rk,cp = *k*8·*N*Rk,c ; interaction §7.2.3 | EN 1992-4 §7.2 (+ ETA) | codifié |
| 12 | soudures : √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ *f*u/(βw·γM2) | §4.5.3.2 | codifié |
| 13 | *S*j = *E*·*z*²/Σ(1/*k*i) sur *k*13, *k*15, *k*16 ; classification | §6.3.4 Table 6.12 ; §5.2.2.5 | codifié |
| 14 | lignes de rupture de coin (I) et de parois (RHS, CHS) : *T*Rd = *m*pl·Σ(ℓi/di) minimisé | Packer 1989, Wheeler 1998, Igarashi ; HERON 2008 | modèle d'ingénieur |

---

## 6 · Réponse E — où l'on peut se tromper du mauvais côté

1. **Platine entière prise rigide** (débord > *c*) : bras de levier surestimé,
   ancrages sous-estimés. Parade : lit sur *A*eff, critère de rigidité § 2.6.
2. **Oublier EN 1992-4.** EN 1993-1-8 ne vérifie que la tige et la platine ; le
   cône de béton, l'arrachement, le bord gouvernent souvent sur un massif étroit.
   Parade : EN 1992-4 obligatoire dans le moteur, avec les dimensions du massif,
   l'état fissuré/non fissuré et l'ETA en entrées explicites.
3. ***f*jd trop optimiste** : √(*A*c1/*A*c0) suppose un massif large et une
   profondeur suffisante ; un mortier épais ou faible invalide βj = 2/3. Parade :
   *A*c1 calculée depuis les bords du massif, contrôle des hypothèses du mortier,
   αcc de l'ANB.
4. **Redistribution plastique entre tiges** alors que le béton d'ancrage est
   fragile. Parade : répartition linéaire plafonnée (§ 2.4).
5. **Boulons de coin** avec une longueur efficace de Table 6.6 qui ne leur
   revient pas. Parade : § 3.3, mention « modèle d'ingénieur », validation.
6. **Cisaillement par les tiges** avec trous surdimensionnés : seules les tiges en
   contact reprennent ; EC3 répond par αb réduit ; EN 1992-4 compte un bras de
   levier au-delà de d/2 de mortier. Parade : les deux calculs, le minimum.
7. **Frottement** compté avec de la traction ou sous combinaison accidentelle de
   soulèvement. Parade : *N*c,Ed = part comprimée de la combinaison considérée.
8. **Moment plastique côté comprimé** : *c* est élastique par choix de la norme ;
   prendre *t*p²/4 là serait non conservatif.
9. **Rigidité surestimée** → pied « encastré » dans le modèle global sans l'être.
   Parade : *S*j et classification dans la note.
10. **Combinaisons** : vérifier chaque combinaison (ELU fondamental, accidentel,
    *N* minimal avec *M* maximal — c'est souvent le soulèvement qui gouverne les
    tiges), pas une enveloppe.
11. **Sensibilité à *k*c/*K*b** dans le modèle élastique : l'effort d'ancrage peut
    varier de quelques dizaines de pour cent selon la raideur retenue. Parade :
    mesurer (§ 8) et, si nécessaire, prendre l'enveloppe de deux rapports.

---

## 7 · Réponse F — ce qu'une note Eurocode peut affirmer

| Configuration | Statut dans la note | Validation complémentaire |
|---|---|---|
| I/H, N seul | conforme (§6.2.8.2) | exemples publiés |
| I/H, N + M plane, tiges dans la largeur d'aile | conforme (§6.2.8.3, Table 6.7) | exemples publiés (SCI P398, ECCS, HERON) |
| I/H, N + Mx + My, interaction linéaire | conforme par démonstration (composants codifiés + convexité) | reproduction des cas uniaxiaux par le même code |
| I/H, N + Mx + My, domaine exact | modèle d'ingénieur | EF hors outil / CBFEM publié, puis promotion |
| I/H, tiges de coin | modèle d'ingénieur | EF hors outil, quelques géométries |
| RHS/SHS | modèle d'ingénieur documenté (CIDECT, AISC, Packer, Wheeler) | exemples CIDECT/AISC |
| CHS | modèle d'ingénieur | exemples CIDECT/AISC |
| Ancrages, tous cas | conforme (EN 1992-4 + ETA) | exemples publiés de fabricants (PROFIS-type) |
| Soudures, cisaillement, rigidité | conforme | calculs manuels |

---

## 8 · Réponse G — périmètre d'une première version

**V1 (opérationnelle).**
- Profilés : HEA, HEB, HEM, IPE, RHS, SHS (base de profilés existante).
- Platine rectangulaire centrée, S235/S275/S355, épaisseur selon EN 10025-2.
- Ancrages : 4 tiges symétriques (2 × 2), puis 6 et 8 en rangées ; tiges à tête
  ou à crosse coulées en place, et post-installées par les valeurs de l'ETA
  saisies ; béton fissuré/non fissuré ; dimensions du massif.
- Sollicitations : N (±), Mx, My, Vx, Vy, Mz, par combinaison (ELU fondamental,
  accidentel) ; SLS pour le soulèvement et *S*j.
- Vérifications : toutes celles du § 3.4 sauf la bêche.
- Verdict : niveau 1 ; niveau 2 affiché.
- Sorties : même présentation que les assemblages existants — synthèse, dessin
  de la zone de contact avec axe neutre, résultante et efforts d'ancrage, plan de
  principe ; export JSON ; place dans le projet multi-assemblages.

**Campagne de validation (RV, comme pour les assemblages existants).**
1. **Uniaxial** : le solveur, bloqué en flexion plane, reproduit à la main la
   Table 6.7 sur l'exemple SCI P398 et sur les exemples ECCS/HERON — tolérance
   déclarée, fiche figée.
2. **Convexité** : pour cent géométries tirées au sort, l'interaction linéaire est
   toujours ≤ le domaine exact — c'est un test, pas une démonstration de plus.
3. **Élastique** : équilibre vérifié à 10⁻¹⁰ sur chaque cas ; sensibilité à
   *k*c/*K*b mesurée et écrite.
4. **Biaxial et coins** : trois à cinq cas en EF hors outil, écarts consignés ;
   promotion du niveau 2 si l'écart est dans la tolérance et du bon côté.
5. **Ancrages** : comparaison à un exemple de fabricant (EN 1992-4) par mode.

**V1.1** : CHS. **V2** : barres pleines, bêche, raidisseurs, tiges non
symétriques, platine excentrée.

---

## 9 · Noyau commun réutilisable — oui, et voici lequel

Le solveur du § 2 ne sait rien du béton : il résout **« une plaque plane posée
sur des zones unilatérales et retenue par des tirants »**. Ce qui change d'un
assemblage à l'autre, c'est la nature des zones et des tirants :

| Assemblage | Zones comprimées | Tirants | Spécifique |
|---|---|---|---|
| Pied de poteau | aire efficace sur béton, *f*jd, *k*13 | tiges d'ancrage (T-stub + EN 1992-4) | EN 1992-4, frottement, mortier |
| Platine d'about boulonnée acier–acier | aile/âme comprimée du support (*F*c,fc,Rd, *F*c,wc,Rd), largeur de T-stub | rangées de boulons (T-stub platine + T-stub aile de poteau, le plus faible) | effets de groupe des rangées (§6.2.7.2) |
| Console sur mur béton | comme pied de poteau, platine verticale | ancrages | cisaillement vertical dominant |
| Console sur poteau acier | aile comprimée | boulons | idem platine d'about |

Composants partagés (certains existent déjà dans `acier/` : tronçon en T,
boulons, soudures, cisaillement de bloc, base de profilés, kit `ndc_pdf`) :
polygones + intégrales exactes ; analyse de section élastique et domaine
plastique ; bibliothèque de tronçons en T et de lignes de rupture ; boulons ;
ancrages EN 1992-4 ; soudures « en ligne » ; béton comprimé ; rigidité. Le
pied de poteau est le **premier client** du noyau, pas un module à part : c'est
ce qui garantit que les suivants coûteront moins et ne divergeront pas.

En flexion plane sur une platine d'about acier–acier, EN 1993-1-8 §6.2.7.2 a sa
propre répartition plastique entre rangées (avec la limite 1,9·*F*t,Rd) : le noyau
la reproduit comme un cas particulier du domaine plastique, et c'est le test de
non-régression de cette famille.

---

## 10 · Exemple illustratif chiffré

*HEA 300 (h 290, b 300, tw 8,5, tf 14), platine 500 × 500 × 25 en S355
(fy = 345 N/mm²), 4 tiges M24 8.8 à ± 200 mm (pinces 50), C30/37 avec
αcc = 0,85, massif large (√(Ac1/Ac0) = 3), mortier conforme, cordons a = 8,
Lb = 260 mm. N = 300 kN en compression, Mx = 50 kNm (comprime y > 0),
My = 30 kNm (comprime x > 0). Données illustratives, non validées.*

**Données dérivées.** *f*cd = 17,0 ; *f*jd = 34,0 N/mm² ; *c* = 46 mm ;
*A*eff = 1 001 cm² (deux ailes de 415 cm², âme) ; *N*Rd en compression pure
= 3 405 kN. *k*c = 127 N/mm³ ; *K*b = 569 kN/mm ; *F*t,Rd = 203 kN par tige.

**(a) Répartition élastique.** Axe neutre incliné (normale à 48,6° de l'axe x),
passant à 46 mm de l'axe du poteau côté tendu ; zone de contact 670 cm² sur
1 001 ; pression maximale **14,9 N/mm²** (44 % de *f*jd) ; résultante
**C = 350 kN en (57 ; 115) mm**, c'est-à-dire sous l'aile comprimée, décalée
vers le côté x > 0 ; **une seule tige tendue, 49,6 kN** (celle du coin opposé),
les trois autres à zéro. Équilibre retrouvé à 10⁻¹⁰. Moment de porte-à-faux de
la platine au-delà de l'aile comprimée : 8,8 kNm/m contre 35,9 élastiques
(24 %) — la platine est rigide au sens du § 2.6.

**(b) Résistance à N = 300 kN.** Capacités uniaxiales exactes : Mx,Rd = 198,
My,Rd = 185 kNm ; capacités « Table 6.7 » (résultante au centre de l'aile,
*z*C = 138 mm ; demi-ailes, *z*C = 98 mm) : **179 et 151 kNm**, toutes deux
gouvernées par la rangée tendue (2 × 203 kN). Dans la direction appliquée
(31°), la frontière exacte est à 174 kNm pour 58 kNm appliqués : **taux 0,34**.
Interaction linéaire des capacités Table 6.7 : **0,28 + 0,20 = 0,48** ; des
capacités exactes : 0,42.

**Lecture.** Le pied est loin de sa capacité ; la tige tendue est à 25 % de sa
résistance acier — ce sont le **cône de béton** (EN 1992-4, selon *h*ef et le
massif) et la **longueur efficace du coin** (§ 3.3) qui décideront, pas EC3-1-8.
Le niveau 1 coûte 0,14 de taux ici ; il reste très loin de 1. La figure jointe
montre la zone de contact, l'axe neutre, la résultante, les efforts de tige, et
le domaine (Mx, My) avec les deux niveaux.

---

## 11 · Proposition, domaine d'application, décision

**Méthode retenue.**
1. Aire efficace EC3 (bande *c*) sur le contour du profilé, découpée par la platine.
2. Analyse de section élastique (platine rigide, lit sans traction sur *A*eff,
   tirants élastiques, raideurs *k*13/*k*16) → pressions, résultante, efforts de
   chaque tige, soulèvement, *S*j ; c'est l'entrée d'EN 1992-4.
3. Domaine plastique convexe (bloc *f*jd, tirants linéaires plafonnés) ; verdict
   V1 par interaction linéaire des capacités uniaxiales à la Table 6.7 ; domaine
   exact en information.
4. Platine : bande *c* élastique + contrôle sous répartition réelle ; tronçons en
   T codifiés ; lignes de rupture pour les coins et les profils creux.
5. Tiges : EC3 (acier) + EN 1992-4 (béton) ; cisaillement : frottement + tiges,
   minimum EC3/EN 1992-4 ; soudures : méthode directionnelle sur le groupe « en
   ligne » ; rigidité et classification.

**Domaine d'application V1.** Pieds sur béton, platine rectangulaire centrée,
profilés en I/H et RHS/SHS, 4 à 8 tiges symétriques, mortier conforme
(βj = 2/3), tiges avec ETA ou coulées en place, charges statiques, ELU
fondamental et accidentel. Hors domaine : CHS (V1.1), barres pleines, bêches,
raidisseurs, platines excentrées, fatigue, séisme, tiges sans ETA.

**Ce qui n'est pas négociable.** EN 1992-4 dans le moteur dès V1 ; la mention
« modèle d'ingénieur » imprimée dans la note là où elle s'applique ; la
campagne RV figée avant de promouvoir le niveau 2 ; aucun réglage propre à un
cas pour rattraper un écart.

**Décision.** Faisable, sans éléments finis, avec un résultat honnête dès V1 :
la répartition est exacte dans son modèle, la résistance est une borne
démontrée sur des composants codifiés, et la marge supplémentaire du modèle
biaxial exact est mesurée plutôt que promise. Le risque principal n'est pas
mathématique, il est de **périmètre** : les coins de platine sur profilé en I et
le béton d'ancrage sur massif étroit sont les deux endroits où une note peut
mentir ; la méthode les nomme, les borne et les valide avant de les affirmer.
