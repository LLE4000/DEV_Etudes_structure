# -*- coding: utf-8 -*-
"""L'onglet Méthode du calcul manuel : le modèle, dans l'ordre d'une note à
la main (docs/assemblages/PLATINE_MANUELLE.md)."""

METHODE = """
#### Le calcul manuel de la platine — ce que l'outil fait, dans l'ordre

**1. Les efforts des tiges** — platine rigide (EN 1992-4 section 6, le modèle des logiciels
d'ancrage) : lit de ressorts sans traction sur l'aire d'appui (bande c de l'EN 1993-1-8 §6.2.5
autour de chaque paroi et de chaque raidisseur, ou toute la platine), tiges tendues seulement ;
N, Mx, My appliqués au point (xN ; yN) — un élément excentré crée son moment N·e tout seul.
On lit l'effort T de chaque boulon, la résultante C, la pression maximale.

**2. Côté tendu, boulon par boulon** — l'appui le plus proche (paroi, raidisseur, cercle), le bras
m jusqu'à lui (cordon déduit ; la diagonale jusqu'à l'angle pour un boulon d'angle), le moment pris
M = T·m, la longueur efficace des lignes de rupture (Tableau 6.6, par boulon) plafonnée à ce que la
face offre (L/2 + m ; m dans un angle), puis le tronçon en T du Tableau 6.2 : avec effet de levier
quand Lb ≤ Lb* (platine mince sur tiges longues — la platine s'appuie sur le béton au-delà du boulon
et le boulon reprend l'effort de levier Q, d'où FT,Rd = 4·Mpl/m), sans effet de levier sinon
(FT,Rd = 2·Mpl/m).

**3. Côté tendu, face par face** — la section au nu de la paroi sous tous les boulons qui tirent
devant elle (les deux faces de l'angle pour un boulon d'angle) : M = ΣT·m⊥ ≤ Mpl = 0,25·b·tp²·fy/γM0
avec b = min(platine ; L + 2·m) (diffusion à 45°). C'est le calcul classique, et c'est lui qui
explose quand l'élément est étroit (un rond fin à l'axe, un boulon au-delà du bout d'un raidisseur).

**4. Côté comprimé, bande par bande** — au nu de chaque face, une bande de largeur unité en
console porte la pression de la répartition : mEd = ∫p·t dt ≤ tp²·fy/(6·γM0) (EN 1993-1-8 §6.2.5).
Chaque paroi reprend la pression qui lui est plus proche qu'à toute autre (entre deux ailes,
chacune porte sa moitié) ; on garde la bande la plus sollicitée. Autour d'un rond la bande est
radiale et s'élargit.

**5. Les trous** — pression diamétrale du Tableau 3.4, boulon par boulon, dans la direction de son
cisaillement (V partagé, torsion autour du centre du groupe, frottement Cf,d·N déduit) ; pinces et
entraxes du Tableau 3.3.

**6. L'épaisseur requise** — pour chaque critère, la plus petite épaisseur qui le satisfait, la
répartition étant RECALCULÉE à chaque épaisseur essayée (bande c, raideurs, efforts des tiges,
pressions) ; la plus grande des quatre est l'épaisseur requise, et le verdict est « t_req ≤ tp ».

**Les deux critères** — EN 1993-1-8 (défaut) : tronçon en T plastique côté tendu, console élastique
côté comprimé. Élastique : partout σ de Von Mises ≤ fy/γM0 sur la bande, sans effet de levier —
√1,5 fois l'épaisseur plastique sans effet de levier, jusqu'à √3 fois avec.

**Les deux aires d'appui** — bande c (référence, sûre) ou toute la platine (platine rigide : les
efforts des tiges se comparent alors à ceux des logiciels d'ancrage ; moins de pression, plus de bras).

**Hors verdict** — les tiges (acier) et l'appui (fjd) sont donnés en information ; le béton
d'ancrage relève du logiciel d'ancrage ou de la vérification complète ; cordons, raidisseurs
eux-mêmes et poteau ne sont pas vérifiés ici.

**Vérification complète** — pour une grille centrée autour d'un seul profilé en I ou RHS sans
raidisseur, le mode « Vérification complète EN 1993-1-8 + EN 1992-4 » reprend la disposition :
Table 6.7, tronçons en T par rangée, ancrages dans le béton.
"""
