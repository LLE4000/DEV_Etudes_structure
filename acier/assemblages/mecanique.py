# -*- coding: utf-8 -*-
"""Mécanique PARTAGÉE des attaches articulées — briques pures, sans écran.

Ce fichier rassemble les briques de calcul déjà écrites DEUX ou TROIS fois
dans les moteurs des trois attaches poutre–poutre (`plat_ame`,
`doubles_cornieres`, `platine_about`). Il est destiné aux attaches
poutre–poteau, qui les réutilisent au lieu de les recopier une quatrième
fois — voir `docs/assemblages/POUTRE_POTEAU.md` §10.1 et §10.2.

**Les trois moteurs existants ne sont PAS modifiés** : ils gardent leurs
copies locales, et leurs 56 cas figés restent opposables. La règle appliquée
est donc celle des copies NOMMÉES : une information a une source unique ;
quand elle ne peut pas en avoir, les copies sont nommées et **un test les
lie**. Ce test est `tests/test_assemblages_mecanique.py` : pour chacun des
56 cas figés du dépôt (12 de `tests/ref_plat_ame.json`, 13 de
`tests/ref_platine_about.json`, 31 de
`acier/reference/reference_double_corniere.json`), il recalcule chaque
grandeur par ce fichier et la compare à celle du moteur d'origine. Si une
formule change d'un côté, il rougit et nomme le cas divergent.

Chaque fonction porte, dans sa docstring, sa RÉFÉRENCE normative et le
MODULE D'ORIGINE de la copie.

PURETÉ : bibliothèque standard seulement (`math`). Aucun import de
Streamlit, d'interface, de dessin ni de base de profilés — ce fichier se
charge et se calcule hors application. Les deux imports internes sont
eux-mêmes purs et volontaires, pour ne pas créer de seconde vérité :

- `acier.js` : la sémantique numérique du moteur de référence JavaScript
  (`dv`, `mn`, `mx`, `hyp`, `INF`). La recopier ici en ferait une seconde
  vérité — c'est précisément ce que ce fichier combat ;
- `acier.resultats.EffortBoulon` : l'enregistrement d'un effort par boulon
  (dataclass), déjà partagé par les quatre modules.

Unités : mm, mm², mm³, mm⁴, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.js import INF, dv, hyp, mn, mx
from acier.resultats import EffortBoulon

S3 = math.sqrt(3)
E_BS = 205000.0          # module d'Young de BS 5950-1 (courbe de la Table 17)
K_ROT = 0.8              # groupe gouverné par le cisaillement des boulons (MSB P5 §2.2.4)


# =========================================================================
#  1. Pression diamétrale (Tableau 3.4) et interaction quadratique
# =========================================================================
def interaction_pd(Fz, Fx, Fbv, Fbh):
    """Interaction quadratique de la pression diamétrale dans les deux
    directions : η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²].

    Référence : EN 1993-1-8 Tableau 3.4 ; interaction : MSB Part 5 §3.2
    (fin plate) et §4.2.1.1 (cornières).
    Origine de la copie : les trois moteurs (`ipd_*`)."""
    return hyp(dv(Fz, Fbv), dv(Fx, Fbh))


def bearing(t, fu, fub, d, d0, e1, p1, n1, e2, p2, n2, gM2, ktrou=1.0,
            exV=None, exH=None, Fz=None, Fx=None):
    """Pression diamétrale d'un boulon sur une pièce d'épaisseur ``t`` :
    les DEUX résistances directionnelles et leurs coefficients.

    Rend ``{abv, k1v, abh, k1h, Fbv, Fbh}`` (kN), plus ``ipd`` (le taux
    d'interaction) si ``Fz`` et ``Fx`` sont donnés.

    - ``e1`` / ``e2`` à ``None`` : AUCUN bord libre dans cette direction (le
      terme de pince disparaît). C'est le cas de l'âme d'une poutre porteuse
      et, pour les attaches poutre–poteau, de l'aile comme de l'âme du
      poteau : le poteau continue au-dessus et en dessous de l'attache ;
    - ``exV`` / ``exH`` : termes supplémentaires du « gauge » quand l'entraxe
      des deux files ne vient pas de ``p2`` (groupe P des cornières :
      ``exV = 1,4·p3/d0 − 1,7``, ``exH = p3/(3·d0) − 1/4``) ;
    - ``ktrou`` : 0,8 pour un trou surdimensionné, 1 pour un trou normal.

    Référence : EN 1993-1-8 Tableau 3.4.
    Origine de la copie : les trois moteurs (fonction interne ``bearing`` de
    `plat_ame` et de `doubles_cornieres`, forme la plus générale retenue)."""
    abv = mn(INF if e1 is None else e1 / (3 * d0),
             p1 / (3 * d0) - 0.25 if n1 > 1 else 9, dv(fub, fu), 1)
    k1v = mn(INF if e2 is None else 2.8 * e2 / d0 - 1.7,
             INF if exV is None else exV,
             1.4 * p2 / d0 - 1.7 if n2 > 1 else 9, 2.5)
    abh = mn(INF if e2 is None else e2 / (3 * d0),
             INF if exH is None else exH,
             p2 / (3 * d0) - 0.25 if n2 > 1 else 9, dv(fub, fu), 1)
    k1h = mn(INF if e1 is None else 2.8 * e1 / d0 - 1.7,
             1.4 * p1 / d0 - 1.7 if n1 > 1 else 9, 2.5)
    out = dict(abv=abv, k1v=k1v, abh=abh, k1h=k1h,
               Fbv=dv(k1v * abv * fu * d * t, gM2) / 1000 * ktrou,
               Fbh=dv(k1h * abh * fu * d * t, gM2) / 1000 * ktrou)
    if Fz is not None and Fx is not None:
        out["ipd"] = interaction_pd(Fz, Fx, out["Fbv"], out["Fbh"])
    return out


def portance_bout_interieur(t, fu, fub, d, d0, e1, e2, p1, p3, gM2, ktrou=1.0):
    """Pression diamétrale d'un groupe à DEUX files symétriques : boulons
    d'EXTRÉMITÉ (pince ``e1``) et boulons INTÉRIEURS (entraxe ``p1``), avec
    le ``k1`` de rive commun aux deux files (``e2`` et ``p3``).

    Rend ``{k1, ab_bout, ab_int, Fb_bout, Fb_int}`` (kN). ``e1`` / ``e2`` à
    ``None`` : aucun bord libre (élément support).

    Référence : EN 1993-1-8 Tableau 3.4.
    Origine de la copie : `platine_about` (fonction interne ``portance``)."""
    k1 = mn(INF if e2 is None else 2.8 * e2 / d0 - 1.7, 1.4 * p3 / d0 - 1.7, 2.5)
    ab_b = mn(INF if e1 is None else e1 / (3 * d0), dv(fub, fu), 1)
    ab_i = mn(p1 / (3 * d0) - 0.25, dv(fub, fu), 1)

    def fb(ab):
        return dv(k1 * ab * fu * d * t, gM2) / 1000 * ktrou

    return dict(k1=k1, ab_bout=ab_b, ab_int=ab_i, Fb_bout=fb(ab_b), Fb_int=fb(ab_i))


def groupe_boulons(Fv, Fb_bout, Fb_int, n_bout, n_int):
    """Résistance d'un GROUPE de fixations en cisaillement : ``(FRd, branche)``.

    - (Fb,Rd)max ≤ Fv,Rd              → FRd = ΣFb,Rd ;
    - (Fb,Rd)min < Fv,Rd < (Fb,Rd)max → FRd = ns·(Fb,Rd)min ;
    - Fv,Rd ≤ (Fb,Rd)min              → FRd = 0,8·ns·Fv,Rd (traction parasite
      due à la rotation : le facteur ne s'applique qu'à cette branche).

    Référence : MSB Part 5 §2.2.4 sur EN 1993-1-8 §3.7.
    Origine de la copie : `platine_about`."""
    ns = n_bout + n_int
    fbs = [Fb_bout] * (1 if n_bout else 0) + [Fb_int] * (1 if n_int else 0)
    fb_max, fb_min = max(fbs), min(fbs)
    if fb_max <= Fv:
        return n_bout * Fb_bout + n_int * Fb_int, "ΣFb,Rd (pression diamétrale)"
    if fb_min < Fv:
        return ns * fb_min, "ns·(Fb,Rd)min"
    return K_ROT * ns * Fv, "0,8·ns·Fv,Rd (cisaillement des boulons)"


# =========================================================================
#  2. Groupe de boulons excentré (répartition élastique)
# =========================================================================
def groupe_excentre(n1, n2, p1, p2, V, N, M):
    """Groupe de ``n1 × n2`` boulons reprenant ``V`` (vertical), ``N``
    (horizontal) et le moment ``M`` (kNm) par répartition élastique sur
    l'inertie polaire.

    Rend ``{n, Ip, xm, ym, Fz, Fx, F}`` : Ip = n·[p1²(n1² − 1) +
    p2²(n2² − 1)]/12 (mm²), les bras extrêmes, les composantes et la
    résultante du boulon le plus sollicité (kN). ``N`` est pris en valeur
    absolue (les moteurs d'origine le font ou le fournissent déjà positif).

    Référence : MSB Part 5 §3.2 (fin plate) et §4.2.1.1 (cornières).
    Origine de la copie : `plat_ame` et `doubles_cornieres`."""
    n = n1 * n2
    Ip = n * (p1 * p1 * (n1 * n1 - 1) + p2 * p2 * (n2 * n2 - 1)) / 12
    xm = (n2 - 1) * p2 / 2
    ym = (n1 - 1) * p1 / 2
    Fz = dv(V, n) + (dv(M * 1000 * xm, Ip) if Ip > 0 else 0)
    Fx = dv(abs(N), n) + (dv(M * 1000 * ym, Ip) if Ip > 0 else 0)
    return dict(n=n, Ip=Ip, xm=xm, ym=ym, Fz=Fz, Fx=Fx, F=hyp(Fz, Fx))


def bolt_table(n1, n2, p1, p2, V, Hh, M, Ip, horizontal_signe=False):
    """Efforts boulon par boulon (répartition élastique, inertie polaire) :
    rangée ``i`` (haut → bas), file ``j``, position, composantes et
    résultante. Liste de ``EffortBoulon``.

    ``horizontal_signe`` : les deux copies existantes divergent sur ce seul
    point, et la divergence est nommée ici. `plat_ame` prend ``|Hh|``, comme
    la vérification du boulon le plus sollicité (défaut, forme cohérente) ;
    la table du groupe S de `doubles_cornieres` garde le SIGNE de NEd — ce
    qui ne change que l'AFFICHAGE d'un cas de compression, sa vérification
    utilisant ``|NEd|``. Les attaches poutre–poteau prennent le défaut.

    Référence : MSB Part 5 §3.2 (fin plate) et §4.2.1.1 (cornières).
    Origine de la copie : `plat_ame` et `doubles_cornieres`."""
    t = []
    n = n1 * n2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            x = (j - (n2 + 1) / 2) * p2
            y = ((n1 + 1) / 2 - i) * p1
            fx = (dv(Hh, n) if horizontal_signe else dv(abs(Hh), n)) \
                - (dv(M * 1000 * y, Ip) if Ip > 0 else 0)
            fz = dv(V, n) + (dv(M * 1000 * x, Ip) if Ip > 0 else 0)
            t.append(EffortBoulon(i=i, j=j, x=x, y=y, fx=fx, fz=fz, f=hyp(fx, fz)))
    return t


# =========================================================================
#  3. Poutre GRUGÉE : section réduite, module, réduction, stabilité locale
# =========================================================================
def section_grugee(profil, fy, dnt, dnb, gM0, V=0.0, maintenue=True,
                   renvoi_non_maintenue=""):
    """Tout le bloc de la poutre grugée, pour un profil en I ou H.

    ``profil`` : indexable par ``h``, ``b``, ``tw``, ``tf``, ``r`` (mm).
    ``dnt`` / ``dnb`` : profondeurs de grugeage en haut / en bas (0 = aucun).
    ``V`` (kN) sert à la réduction ``ρ`` ; ``maintenue`` dit si la poutre est
    maintenue au déversement (domaine de validité de la règle du guide) ;
    ``renvoi_non_maintenue`` est le renvoi propre au module, ajouté à la
    phrase de domaine (`doubles_cornieres` cite « SCI P358 Check 7 »).

    Rend ``{cas_g, Asec, h_T, A_Tee, Av, VRd_g, hw_T, zb_T, I_N, W_N, rho_N,
    Mv_N, el_N, lim_N, ln_max, domaine, val_N}`` :

    - ``cas_g`` : 0 non grugée, 1 une semelle, 2 deux semelles ;
    - ``Av`` : aire de cisaillement, EN 1993-1-1 §6.2.6(3) hors grugeage,
      section du Té résiduel sinon ;
    - ``W_N``, ``rho_N``, ``Mv_N`` : flexion + cisaillement à l'extrémité du
      grugeage, MSB Part 5 §4.2.4 ;
    - ``ln_max``, ``lim_N`` : stabilité locale en longueur, MSB P5 §4.2.5
      (h/tw ≤ 54,3 en S275, 48 en S355 ; sinon K·h/(h/tw)³ avec
      K = 160 000 / 110 000) ;
    - ``domaine`` : code du domaine de validité (``"sans_objet"``,
      ``"non_maintenue"``, ``"nuance_haute"``, ``"s235"``, ``"valide"``) ;
      ``val_N`` en est la phrase.

    Référence : MSB Part 5 §4.2.4 et §4.2.5 ; EN 1993-1-1 §6.2.6.
    Origine de la copie : les TROIS moteurs, à l'identique (le plus gros
    bloc dupliqué du dépôt)."""
    h = profil["h"]; b = profil["b"]; tw = profil["tw"]
    tf = profil["tf"]; r = profil["r"]
    cas_g = 0 if (dnt == 0 and dnb == 0) else (2 if (dnt > 0 and dnb > 0) else 1)
    Asec = 2 * b * tf + (h - 2 * tf) * tw + (4 - math.pi) * r * r
    h_T = h - dnt - dnb
    A_Tee = b * tf + tw * (h_T - tf)
    if cas_g == 0:
        Av = mx(Asec - 2 * b * tf + (tw + 2 * r) * tf, (h - 2 * tf) * tw)
    elif cas_g == 1:
        Av = A_Tee - b * tf + (tw + 2 * r) * tf / 2
    else:
        Av = tw * h_T
    VRd_g = dv(Av * fy / S3, gM0) / 1000
    hw_T = h_T - tf
    zb_T = dv(b * tf * tf / 2 + tw * hw_T * (tf + hw_T / 2), A_Tee)
    I_N = (b * tf ** 3 / 12 + b * tf * (zb_T - tf / 2) ** 2
           + tw * hw_T ** 3 / 12 + tw * hw_T * (tf + hw_T / 2 - zb_T) ** 2)
    W_N = tw * h_T * h_T / 6 if cas_g == 2 else dv(I_N, mx(zb_T, h_T - zb_T))
    rho_N = 1 if V <= 0.5 * VRd_g else mx(0, 1 - (2 * dv(V, VRd_g) - 1) ** 2)
    Mv_N = dv(W_N * fy, gM0) / 1e6 * rho_N
    el_N = dv(h, tw)
    lim_N = 54.3 if fy <= 275 else 48
    ln_max = h if el_N <= lim_N else dv((160000 if fy <= 275 else 110000) * h, el_N ** 3)
    if cas_g == 0:
        domaine = "sans_objet"
        val_N = "Sans objet (poutre non grugée)"
    elif not maintenue:
        domaine = "non_maintenue"
        val_N = ("NON VALIDE : poutre non maintenue au déversement – stabilité "
                 "d'ensemble de la poutre grugée à justifier" + renvoi_non_maintenue)
    elif fy > 355:
        domaine = "nuance_haute"
        val_N = "NON VALIDE : nuance > S355 – règle non publiée pour cette nuance"
    elif fy < 275:
        domaine = "s235"
        val_N = "S235 : limites S275 appliquées (côté sécurité) – interprétation"
    else:
        domaine = "valide"
        val_N = "Valide"
    return dict(cas_g=cas_g, Asec=Asec, h_T=h_T, A_Tee=A_Tee, Av=Av, VRd_g=VRd_g,
                hw_T=hw_T, zb_T=zb_T, I_N=I_N, W_N=W_N, rho_N=rho_N, Mv_N=Mv_N,
                el_N=el_N, lim_N=lim_N, ln_max=ln_max, domaine=domaine, val_N=val_N)


# =========================================================================
#  4. Déversement d'un plat LONG
# =========================================================================
def lam_LT(z, hp, tp):
    """Élancement de déversement d'un plat d'âme LONG :
    λLT = 2,8·√(z·hp/(1,5·tp²)) ; ``INF`` si ``tp`` ≤ 0.

    Référence : SCI P358 (plats longs).
    Origine de la copie : `plat_ame` (``R.lam_LT``)."""
    return 2.8 * math.sqrt(dv(z * hp, 1.5 * tp * tp)) if tp > 0 else INF


def pb_bs5950(lam, py):
    """Contrainte de déversement pb (MPa) — BS 5950-1 Annexe B.2.1, la
    formule qui engendre la Table 17 : Perry-Robertson avec αb = 0,007,
    λL0 = 0,4·√(π²E/py), E = 205 000 MPa. pb = py pour λ ≤ λL0.

    Référence : BS 5950-1:2000 Annexe B.2.1 (courbe de la Table 17).
    Origine de la copie : `plat_ame`."""
    if py <= 0:
        return 0.0
    lam0 = 0.4 * math.sqrt(math.pi * math.pi * E_BS / py)
    if lam <= lam0:
        return py
    eta = 0.007 * (lam - lam0)
    pE = math.pi * math.pi * E_BS / (lam * lam)
    phi = (py + (eta + 1) * pE) / 2
    rac = phi * phi - pE * py
    return dv(pE * py, phi + math.sqrt(rac)) if rac > 0 else py


# =========================================================================
#  5. Règles de détail et de ductilité
# =========================================================================
def tp_ductile(d, fub, fy_p):
    """Épaisseur maximale d'une platine pour la ductilité de l'articulation
    (la rotation vient de la déformation de la platine, pas des boulons) :
    t ≤ 0,36·d·√(fub/fy). mm.

    Référence : EN 1993-1-8 §6.4.2(2) ; règle reprise par ECCS n°126 pour
    les platines d'about (d/tp ≥ 2,8·√(fy/fub)).
    Origine de la copie : `platine_about`."""
    return 0.36 * d * math.sqrt(dv(fub, fy_p)) if fy_p > 0 else INF


def coef_pleine_resistance(fy, fu, bw, gM2, gM0=1.0):
    """a/tw de la PLEINE RÉSISTANCE d'un double cordon sur une âme tendue :
    βw·γM2·fy/(√2·fu·γM0) — redonne les 0,46 / 0,48 / 0,55 du guide en
    S235 / S275 / S355.

    Référence : MSB Part 5 §2.2 (dérivation d'EN 1993-1-8 §4.5.3.3).
    Origine de la copie : `platine_about`."""
    return dv(bw * gM2 * fy, math.sqrt(2) * fu * gM0)


def a_pleine_resistance_plat(tp, fy_p, bw, fu_w, gM2, gM0=1.0):
    """Gorge de PLEINE RÉSISTANCE d'un double cordon soudant un PLAT sur
    chant : a ≥ tp·fy·βw·γM2/(2·fu·γM0). mm.

    Référence : dérivation d'EN 1993-1-8 §4.5.3.3 (2·a·fvw,d ≥ tp·fy/(√3·γM0)).
    Origine de la copie : `plat_ame` (``R.a_plein``)."""
    return dv(tp * fy_p * bw * gM2, 2 * fu_w * gM0)


# =========================================================================
#  6. Tronçon en T équivalent (EN 1993-1-8 §6.2.4, Tableau 6.2)
# =========================================================================
def troncon_T(leff1, leff2, t, m, n, ew, f, gamma, SFt):
    """Tronçon en T équivalent : modes 1 (MÉTHODE 2), 2 et 3, GÉNÉRIQUES —
    aucune géométrie en dur, les longueurs efficaces sont fournies.

    - ``leff1`` / ``leff2`` : Σleff des modes 1 et 2 (mm), rendues par
      ``leff_aile_poteau`` ou ``leff_platine_bords_libres`` ;
    - ``t`` : épaisseur de l'aile fléchie (mm) ; ``m``, ``n``, ``ew`` : les
      bras du Tableau 6.2 (mm) ;
    - ``f`` / ``gamma`` : fy/γM0 à l'ELU, fu/γMu en tying ;
    - ``SFt`` : ΣFt,Rd des boulons du tronçon (kN).

    Rend ``{Mpl1, Mpl2, FT1, FT2, FT3, FT_Rd}`` (N·mm et kN) :
    FT,1 = (8n − 2ew)·Mpl,1/[2mn − ew(m + n)] — et la MÉTHODE 1
    (4·Mpl,1/m) si le dénominateur dégénère ;
    FT,2 = (2·Mpl,2 + n·ΣFt,Rd)/(m + n) ; FT,3 = ΣFt,Rd.

    Référence : EN 1993-1-8 §6.2.4, Tableau 6.2 (méthode 2) ; MSB Part 5
    §2.3 (tying).
    Origine de la copie : `platine_about` (``tronçon_en_T``, lié à la
    platine) et `doubles_cornieres` (lié aux ailes A des cornières)."""
    Mpl1 = 0.25 * leff1 * t * t * dv(f, gamma)              # N·mm
    Mpl2 = 0.25 * leff2 * t * t * dv(f, gamma)
    FT1 = dv((8 * n - 2 * ew) * Mpl1, 2 * m * n - ew * (m + n)) / 1000
    if 2 * m * n - ew * (m + n) <= 0:
        FT1 = 4 * Mpl1 / m / 1000                           # méthode 1 si la 2 dégénère
    FT2 = dv(2 * Mpl2 + n * SFt * 1000, m + n) / 1000
    FT3 = SFt
    return dict(Mpl1=Mpl1, Mpl2=Mpl2, FT1=FT1, FT2=FT2, FT3=FT3,
                FT_Rd=mn(FT1, FT2, FT3))


def leff_aile_poteau(n1, p1, m, e):
    """Σleff d'une AILE DE POTEAU NON RAIDIE — Tableau 6.4.

    Un poteau continue au-dessus et en dessous de l'attache : il n'y a
    AUCUNE rangée d'extrémité, **toutes les rangées sont INTÉRIEURES**.

    - rangées isolées  : leff,cp = 2·π·m      leff,nc = 4·m + 1,25·e
    - rangées en groupe : leff,cp = 2·p       leff,nc = p
    - Σleff,1 = min[ min(Σcp ; Σnc) isolées ; min(Σcp ; Σnc) en groupe ]
    - Σleff,2 = min( Σnc isolées ; Σnc en groupe )

    Deux conséquences utiles : en groupe, ``nc = p`` est TOUJOURS sous
    ``cp = 2·p``, donc la branche « en groupe » vaut ``n1·p1`` ; et avec UNE
    seule rangée il n'y a pas de groupe — seules les formules isolées
    s'appliquent (``p1`` n'a alors aucun sens).

    Rend ``(leff1, leff2)`` en mm.

    Référence : EN 1993-1-8 Tableau 6.4 (aile de poteau non raidie). Le
    NUMÉRO du tableau est à recontrôler sur le texte normatif (point P-1 de
    `docs/assemblages/POUTRE_POTEAU.md` §9 : la consigne citait le Tableau
    6.6) ; les FORMULES, elles, sont bien celles de l'aile non raidie, et
    ce sont celles que `platine_about` applique déjà.
    Origine de la copie : AUCUNE — composant NEUF."""
    cp_i = n1 * (2 * math.pi * m)
    nc_i = n1 * (4 * m + 1.25 * e)
    if n1 < 2:
        return mn(cp_i, nc_i), nc_i
    cp_g = n1 * (2 * p1)
    nc_g = n1 * p1
    return mn(mn(cp_i, nc_i), mn(cp_g, nc_g)), mn(nc_i, nc_g)


def leff_platine_bords_libres(n1, p1, e1h, e1b, m, e):
    """Σleff d'une PLATINE à bords libres en haut et en bas : les deux
    rangées d'extrémité sont à ``e1h`` et ``e1b`` du bord, les autres sont
    intérieures. Rend ``(leff1, leff2)`` en mm.

    Pour les détails courants, Σleff retombe sur ``hp``, la valeur du guide.
    Comme pour l'aile de poteau, le terme Σcp « en groupe » n'est jamais
    déterminant pour Σleff,1 (en groupe, ``nc`` reste sous ``cp``) : il est
    INERTE, dans cette copie comme dans l'originale.

    Référence : EN 1993-1-8 Tableau 6.4 (formules), appliqué à la platine —
    interprétation déclarée comme telle par `platine_about` ; MSB Part 5
    §2.3.
    Origine de la copie : `platine_about` (``tronçon_en_T``)."""
    pi = math.pi
    p = p1

    def bout(e1):
        """Rangée d'extrémité près d'un bord libre (distance e1)."""
        return dict(cp_seul=mn(2 * pi * m, pi * m + 2 * e1),
                    nc_seul=mn(4 * m + 1.25 * e, 2 * m + 0.625 * e + e1),
                    cp_grp=mn(pi * m + p, 2 * e1 + p),
                    nc_grp=mn(2 * m + 0.625 * e + 0.5 * p, e1 + 0.5 * p))

    if n1 == 1:
        cp = mn(2 * pi * m, pi * m + e1h + e1b)
        nc = mn(4 * m + 1.25 * e, 2 * m + 0.625 * e + 0.5 * (e1h + e1b))
        return mn(cp, nc), nc
    bh, bb = bout(e1h), bout(e1b)
    cp_g = bh["cp_grp"] + bb["cp_grp"] + (n1 - 2) * 2 * p
    nc_g = bh["nc_grp"] + bb["nc_grp"] + (n1 - 2) * p
    cp_i = bh["cp_seul"] + bb["cp_seul"] + (n1 - 2) * 2 * pi * m
    nc_i = bh["nc_seul"] + bb["nc_seul"] + (n1 - 2) * (4 * m + 1.25 * e)
    return mn(mn(cp_g, nc_g), mn(cp_i, nc_i)), mn(nc_g, nc_i)


# =========================================================================
#  7. Géométrie du POTEAU porteur (famille poutre – poteau)
# =========================================================================
def geometrie_poteau(profil, j=0.0, gh=0.0, bS=0.0, dw=0.0):
    """Les largeurs utiles d'un poteau en I ou H qui porte une attache
    articulée, dans les DEUX cas d'appui — étude §3.2.2, §4.3 et §5.3.

    ``profil`` : indexable par ``h``, ``b``, ``tw``, ``tf``, ``r`` (mm) — le
    POTEAU. ``j`` : jeu latéral de montage ; ``gh`` : distance de la face
    d'appui à l'about de la poutre portée (c'est le jeu ``gh`` pour un plat
    d'âme ou des cornières, et l'épaisseur ``tp`` de la platine pour une
    platine d'about — l'about est alors à ``tp`` de la face d'appui) ;
    ``bS`` : largeur de semelle de la poutre portée ; ``dw`` : diamètre
    d'appui de la rondelle ou de l'écrou.

    Rend ``{libre_droit, demi_droit, libre_ailes, p3_congeP, p3_maxP, gh_b1,
    rec_min, passe_S}`` :

    - ``libre_droit`` : largeur DROITE de l'âme entre congés,
      ``hP − 2·tfP − 2·rP`` — ce qui reste utilisable en cas (b) ;
    - ``demi_droit`` : la demi-largeur droite, ``(hP − 2·tfP)/2 − rP``,
      mesurée depuis l'axe de l'âme (une file de boulons, un talon de
      cornière ou un demi-gauge ne peut pas la dépasser en cas b) ;
    - ``libre_ailes`` : passage de la semelle portée entre les ailes,
      ``hP − 2·tfP − 2·j`` ;
    - ``p3_congeP`` : gauge MINIMAL d'une file sur l'aile (cas a),
      ``twP + 2·rP + dw`` — l'écrou est du côté intérieur de l'aile, au
      droit du congé : c'est la règle qui refuse le ``p3 = 90`` usuel des
      âmes de poutre sur l'aile d'un HEB 300 (point P-7 de l'étude, nature
      INT : dégagement géométrique, pas une clause normative) ;
    - ``p3_maxP`` : gauge MAXIMAL en cas (b), ``2·(demi_droit − dw/2)`` —
      les deux écrous dégagent les deux congés ;
    - ``gh_b1`` : ``gh`` à partir duquel l'about de la poutre passe au-delà
      des bouts d'aile du poteau, ``(bP − twP)/2 + j`` (variante b1) ;
    - ``rec_min`` : recoupe minimale des semelles depuis l'about si la
      poutre ne passe pas, ``max(0 ; gh_b1 − gh)``. La recoupe en LARGEUR
      n'est PAS un grugeage (point P-3) ;
    - ``passe_S`` : la semelle portée passe entre les ailes sans recoupe.

    Les quatre derniers arguments ne servent qu'aux grandeurs qui en
    dépendent : un appelant qui ne lit que ``libre_droit`` ou ``demi_droit``
    (le prédimensionnement, qui tourne avant ``compute``) peut les omettre.

    Référence : géométrie de profilés laminés ; MSB Part 5 §2 à §4 pour les
    cas d'appui. Origine de la copie : les TROIS moteurs poutre–poteau, à
    l'identique — ``tests/test_assemblages_mecanique.py`` les lie."""
    hP = profil["h"]; bP = profil["b"]; twP = profil["tw"]
    tfP = profil["tf"]; rP = profil["r"]
    libre_droit = hP - 2 * tfP - 2 * rP
    demi_droit = (hP - 2 * tfP) / 2 - rP
    libre_ailes = hP - 2 * tfP - 2 * j
    gh_b1 = (bP - twP) / 2 + j
    return dict(libre_droit=libre_droit, demi_droit=demi_droit,
                libre_ailes=libre_ailes, p3_congeP=twP + 2 * rP + dw,
                p3_maxP=2 * (demi_droit - dw / 2), gh_b1=gh_b1,
                rec_min=mx(0.0, gh_b1 - gh), passe_S=bool(bS <= libre_ailes + 1e-9))
