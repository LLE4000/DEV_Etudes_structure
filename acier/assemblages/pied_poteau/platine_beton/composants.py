# -*- coding: utf-8 -*-
"""Composants du pied de poteau — un composant, une fonction, ses références.

- béton comprimé et bande ``c`` : EN 1993-1-8 §6.2.5 (équations (6.4) à (6.6)),
  EN 1992-1-1 §6.7 (équation (6.63)) ;
- tronçon en T de la platine tendue : EN 1993-1-8 §6.2.4, Tableau 6.2 (modes
  1-2-3, Lb*), longueurs efficaces dérivées des motifs des Tableaux 6.4 et
  6.6 PAR TIGE (convention « demi-longueur » des tableaux : la somme sur une
  rangée de deux tiges redonne exactement les valeurs de rangée du Tableau
  6.6) ; au-delà de l'about d'aile, longueur plafonnée (modèle d'ingénieur) ;
- tiges, acier : EN 1993-1-8 Tableau 3.4 et §6.2.2 (αb), EN 1992-4 §7.2.1.3
  et §7.2.2.3 ; tiges, béton : EN 1992-4 §7.2.1 (cône, arrachement) et §7.2.2
  (bord, pry-out), interaction §7.2.3 ; adhérence d'une crosse : EN 1993-1-8
  §6.2.6.12 sur EN 1992-1-1 §8.4.2 ;
- cordons : EN 1993-1-8 §4.5.3.2 (méthode directionnelle), groupe traité
  « en ligne » ;
- rigidité : EN 1993-1-8 §6.3.4 (k13, k15, k16), classification §5.2.2.5.

Unités : N, mm, MPa (N/mm²), N·mm. Les valeurs k des formules EN 1992-4
sont celles des tiges coulées en place (tête ou plaque d'ancrage) ; une tige
post-installée relèvera de son ETA (V1.1).
"""
import math

INF = float("inf")
E_ACIER = 210000.0


# --------------------------------------------------------- béton comprimé
def E_cm(fck):
    """Module sécant du béton, EN 1992-1-1 Tableau 3.1 : 22·((fck + 8)/10)^0,3 GPa."""
    return 22000.0 * ((fck + 8.0) / 10.0) ** 0.3


def beton_comprime(fck, a_cc, g_c, bp, lp, bc, lc, hc, t_g, f_g, beta_j=2.0 / 3.0):
    """Résistance de calcul à l'écrasement du joint fjd = βj·FRdu/(beff·leff),
    avec FRdu = Ac0·fcd·√(Ac1/Ac0) ≤ 3·fcd·Ac0 (EN 1992-1-1 §6.7). Ac0 est
    pris égal à la PLATINE (simplification conservative : √(Ac1/Ac0) ne
    dépasse pas celui d'un tronçon en T plus petit) ; Ac1 est l'aire
    homothétique concentrique dans le massif, limitée par ses bords, par
    3·Ac0 et par la hauteur (diffusion 2:1 : b2 ≤ b1 + h)."""
    fcd = a_cc * fck / g_c
    Ac0 = bp * lp
    cx, cy = (bc - bp) / 2.0, (lc - lp) / 2.0
    b2 = min(3 * bp, bp + 2 * max(0.0, cx), bp + max(0.0, hc))
    l2 = min(3 * lp, lp + 2 * max(0.0, cy), lp + max(0.0, hc))
    Ac1 = b2 * l2
    ratio = min(3.0, math.sqrt(Ac1 / Ac0)) if Ac0 > 0 else 1.0
    fjd = beta_j * fcd * ratio
    mortier_ok = (t_g <= 0.2 * min(bp, lp)) and (f_g >= 0.2 * fck)
    return dict(fcd=fcd, Ac0=Ac0, Ac1=Ac1, b2=b2, l2=l2, ratio=ratio, fjd=fjd, beta_j=beta_j,
                mortier_ok=mortier_ok)


def c_bande(tp, fy_p, fjd, g_M0):
    """Largeur de bande c = tp·√(fy/(3·fjd·γM0)) — EN 1993-1-8 éq. (6.5)."""
    return tp * math.sqrt(fy_p / (3.0 * fjd * g_M0)) if fjd > 0 else 0.0


# ------------------------------------------------- tronçon en T (traction)
def tstub_rangee(tiges_rangee, m, e_x, tp, fy_p, g_M0, Lb, As, Ft_anc, cap=None):
    """Résistance d'une rangée de tiges tendues contre une ligne d'appui.

    ``tiges_rangee`` : liste de dicts ``e_lat`` (pince latérale, le long de
    la ligne d'appui) et ``s_min`` (entraxe minimal avec une voisine de la
    rangée, ∞ si seule) ; ``m`` : distance à l'appui, cordon déduit ; ``e_x``
    : pince perpendiculaire (au-delà de la tige) ; ``Lb`` : longueur
    d'allongement de la tige (Tableau 6.2 : 8d + mortier + platine + rondelle
    + ½ écrou) ; ``As`` : section résistante d'une tige ; ``Ft_anc`` :
    résistance en traction d'une tige (acier) ; ``cap`` : plafond de la
    longueur efficace PAR TIGE (convention demi-longueur), None sinon.

    Longueurs efficaces par tige (demi-longueur) :
      circulaire     : min(π·m ; π·m/2 + s/2 ; π·m/2 + e_lat)
      non circulaire : min(2·m + 0,625·e_x ; e_lat/2 + m + 0,3125·e_x ;
                           s/4 + m + 0,3125·e_x)
    — la somme sur deux tiges redonne le Tableau 6.6 (rangée extérieure à
    l'aile : 2πmx ; πmx + w ; πmx + 2e ; 4mx + 1,25ex ; e + 2mx + 0,625ex ;
    0,5w + 2mx + 0,625ex).
    Modes du Tableau 6.2 (méthode 1) ; sans effet de levier si Lb > Lb*."""
    m = max(m, 1.0)
    n_b = len(tiges_rangee)
    cp = nc = 0.0
    for t in tiges_rangee:
        e_lat, s = max(t["e_lat"], 0.0), t.get("s_min", INF)
        cpi = min(math.pi * m, 0.5 * math.pi * m + 0.5 * s, 0.5 * math.pi * m + e_lat)
        nci = min(2 * m + 0.625 * e_x, 0.5 * e_lat + m + 0.3125 * e_x, 0.25 * s + m + 0.3125 * e_x)
        if cap is not None:
            cpi, nci = min(cpi, cap), min(nci, cap)
        cp += cpi
        nc += nci
    leff1, leff2 = min(cp, nc), nc
    Mpl1 = 0.25 * leff1 * tp ** 2 * fy_p / g_M0
    Mpl2 = 0.25 * leff2 * tp ** 2 * fy_p / g_M0
    Lb_star = 8.8 * m ** 3 * As / (leff1 * tp ** 3) if leff1 > 0 else INF
    n = min(e_x, 1.25 * m)
    levier = Lb <= Lb_star
    FT3 = n_b * Ft_anc
    if levier:
        FT1 = 4 * Mpl1 / m
        FT2 = (2 * Mpl2 + n * FT3) / (m + n)
    else:
        FT1 = 2 * Mpl1 / m
        FT2 = INF
    FT = min(FT1, FT2, FT3)
    mode = "1" if FT == FT1 else ("2" if FT == FT2 else "3")
    if not levier and mode == "1":
        mode = "1-2 (sans effet de levier)"
    return dict(n=n_b, m=m, e_x=e_x, n_lev=n, leff_cp=cp, leff_nc=nc, leff1=leff1, leff2=leff2,
                Mpl1=Mpl1, Mpl2=Mpl2, Lb=Lb, Lb_star=Lb_star, levier=levier,
                FT1=FT1, FT2=FT2, FT3=FT3, FT_Rd=FT, mode=mode, FT_tige=FT / n_b if n_b else 0.0)


# --------------------------------------------------------- tiges : acier
def tige_traction_EC3(f_ub, As, g_M2):
    """Ft,Rd = 0,9·fub·As/γM2 — EN 1993-1-8 Tableau 3.4."""
    return 0.9 * f_ub * As / g_M2


def tige_traction_EN1992_4(As, f_uk, f_yk):
    """NRk,s = As·fuk ; γMs,N = 1,2·fuk/fyk ≥ 1,4 — EN 1992-4 §7.2.1.3 et Tableau 4.1."""
    g = max(1.4, 1.2 * f_uk / f_yk) if f_yk > 0 else 1.5
    return As * f_uk / g, g


def tige_cisaillement_EC3(f_yb, f_ub, As, g_M2):
    """F2,vb,Rd = αb·fub·As/γM2, αb = 0,44 − 0,0003·fyb — EN 1993-1-8 §6.2.2."""
    ab = 0.44 - 0.0003 * f_yb
    return ab * f_ub * As / g_M2, ab


def tige_cisaillement_EN1992_4(As, f_uk, f_yk, d, t_g, f_g, N_Ed_tige, N_Rd_s):
    """VRk,s = k6·As·fuk (k6 = 0,6 si fuk ≤ 500, 0,5 au-delà), γMs,V = fuk/fyk ≥ 1,25 —
    EN 1992-4 §7.2.2.3. Si le mortier dépasse d/2 ou fait moins de 30 N/mm², le
    cisaillement agit avec un BRAS DE LEVIER l = d/2 + tg : VRk,s,M = αM·MRk,s/l,
    MRk,s = 1,2·Wel·fuk·(1 − NEd/NRd,s), αM = 2 (platine encastrant la tige)."""
    k6 = 0.6 if f_uk <= 500 else 0.5
    g = max(1.25, f_uk / f_yk) if f_yk > 0 else 1.25
    VRd = k6 * As * f_uk / g
    levier = (t_g > 0.5 * d) or (f_g < 30.0)
    VRd_M = INF
    if levier:
        d_s = math.sqrt(4 * As / math.pi)
        Wel = math.pi * d_s ** 3 / 32.0
        MRk = 1.2 * Wel * f_uk * max(0.0, 1.0 - (N_Ed_tige / N_Rd_s if N_Rd_s > 0 else 0.0))
        l = 0.5 * d + t_g
        VRd_M = 2.0 * MRk / l / g
    return min(VRd, VRd_M), VRd, VRd_M, levier, g


# --------------------------------------------------------- tiges : béton
def cone_beton(hef, fck, fiss, x_t, y_t, bc, lc, e_x, e_y, g_Mc):
    """Cône de béton du groupe TENDU — EN 1992-4 §7.2.1.4 :
    NRk,c = N⁰Rk,c·(Ac,N/A⁰c,N)·ψs,N·ψre,N·ψec,N·ψM,N, N⁰Rk,c = k1·√fck·hef^1,5
    (k1 = 7,7 fissuré, 11,0 non fissuré, tiges coulées en place), scr,N = 3·hef,
    ccr,N = 1,5·hef. ``x_t``, ``y_t`` : coordonnées des tiges tendues (massif
    centré bc × lc) ; ``e_x``, ``e_y`` : excentrement de la résultante de
    traction par rapport au centre du groupe tendu. ψM,N = 1 (conservatif)."""
    if not x_t:
        return dict(NRk=0.0, NRd=0.0, actif=False)
    k1 = 7.7 if fiss else 11.0
    N0 = k1 * math.sqrt(fck) * hef ** 1.5
    scr, ccr = 3.0 * hef, 1.5 * hef
    xs, ys = sorted(set(x_t)), sorted(set(y_t))
    cx1, cx2 = bc / 2 + xs[0], bc / 2 - xs[-1]
    cy1, cy2 = lc / 2 + ys[0], lc / 2 - ys[-1]
    wx = min(ccr, cx1) + sum(min(scr, b - a) for a, b in zip(xs, xs[1:])) + min(ccr, cx2)
    wy = min(ccr, cy1) + sum(min(scr, b - a) for a, b in zip(ys, ys[1:])) + min(ccr, cy2)
    Ac0 = scr ** 2
    Ac = wx * wy
    cmin = min(cx1, cx2, cy1, cy2)
    psi_s = min(1.0, 0.7 + 0.3 * cmin / ccr)
    psi_re = min(1.0, 0.5 + hef / 200.0)
    psi_ec = 1.0 / (1.0 + 2.0 * abs(e_x) / scr) / (1.0 + 2.0 * abs(e_y) / scr)
    psi_M = 1.0
    NRk = N0 * Ac / Ac0 * psi_s * psi_re * psi_ec * psi_M
    return dict(NRk=NRk, NRd=NRk / g_Mc, actif=True, N0=N0, Ac=Ac, Ac0=Ac0, psi_s=psi_s, psi_re=psi_re,
                psi_ec=psi_ec, psi_M=psi_M, cmin=cmin, scr=scr, ccr=ccr, k1=k1)


def arrachement_tete(d, d_h, fck, fiss, g_Mc):
    """NRk,p = k2·Ah·fck (k2 = 7,5 fissuré, 10,5 non fissuré), Ah = π(dh² − d²)/4 —
    EN 1992-4 §7.2.1.5 (tiges à tête)."""
    k2 = 7.5 if fiss else 10.5
    Ah = math.pi / 4.0 * max(0.0, d_h ** 2 - d ** 2)
    NRk = k2 * Ah * fck
    return dict(NRk=NRk, NRd=NRk / g_Mc, Ah=Ah, k2=k2)


def adherence_crosse(d, l_a, fck, g_c, eta1=0.7):
    """Tige à crosse : Ft,bond,Rd = π·d·la·fbd — EN 1993-1-8 §6.2.6.12 sur
    EN 1992-1-1 §8.4.2 : fbd = 2,25·η1·η2·fctd, fctd = fctk,0,05/γc,
    fctk,0,05 = 0,7·fctm, fctm = 0,30·fck^(2/3). η1 = 0,7 (conditions
    d'adhérence médiocres, tige verticale coulée en place), η2 = 1 (d ≤ 32).
    La crosse n'est pas comptée (sécurité)."""
    fctm = 0.30 * fck ** (2.0 / 3.0)
    fctd = 0.7 * fctm / g_c
    eta2 = 1.0 if d <= 32 else (132.0 - d) / 100.0
    fbd = 2.25 * eta1 * eta2 * fctd
    return dict(F=math.pi * d * l_a * fbd, fbd=fbd, fctd=fctd)


def bord_beton(d, hef, fck, fiss, c1, c2a, c2b, s_perp, h_c, g_Mc):
    """Rupture du bord du massif en cisaillement — EN 1992-4 §7.2.2.5, pour la
    RANGÉE de tiges la plus proche du bord chargé (trous normaux : seules ces
    tiges transmettent le cisaillement au béton) :
    V⁰Rk,c = k9·d^α·lf^β·√fck·c1^1,5, k9 = 1,7 fissuré / 2,4 non fissuré,
    α = 0,1·(lf/c1)^0,5, β = 0,1·(d/c1)^0,2, lf = min(hef ; 12d) (d ≤ 24),
    A⁰c,V = 4,5·c1², Ac,V = (min(1,5c1 ; c2a) + Σmin(s ; 3c1) + min(1,5c1 ; c2b))·min(1,5c1 ; h),
    ψs,V = 0,7 + 0,3·c2/(1,5c1) ≤ 1, ψh,V = (1,5c1/h)^0,5 ≥ 1, ψec = ψα = ψre = 1."""
    if c1 <= 0:
        return dict(VRk=0.0, VRd=0.0)
    k9 = 1.7 if fiss else 2.4
    lf = min(hef, 12.0 * d) if d <= 24 else min(hef, max(8.0 * d, 300.0))
    alpha = 0.1 * (lf / c1) ** 0.5
    beta = 0.1 * (d / c1) ** 0.2
    V0 = k9 * d ** alpha * lf ** beta * math.sqrt(fck) * c1 ** 1.5
    Ac0 = 4.5 * c1 ** 2
    larg = min(1.5 * c1, c2a) + sum(min(3.0 * c1, s) for s in s_perp) + min(1.5 * c1, c2b)
    Ac = larg * min(1.5 * c1, h_c)
    psi_s = min(1.0, 0.7 + 0.3 * min(c2a, c2b) / (1.5 * c1))
    psi_h = max(1.0, math.sqrt(1.5 * c1 / h_c)) if h_c > 0 else 1.0
    VRk = V0 * Ac / Ac0 * psi_s * psi_h
    return dict(VRk=VRk, VRd=VRk / g_Mc, V0=V0, Ac=Ac, Ac0=Ac0, psi_s=psi_s, psi_h=psi_h, lf=lf, c1=c1)


def pryout(NRk_c_groupe, hef, g_Mc):
    """VRk,cp = k8·NRk,c (groupe entier), k8 = 1,0 si hef < 60 mm, 2,0 sinon —
    EN 1992-4 §7.2.2.4."""
    k8 = 1.0 if hef < 60 else 2.0
    return dict(VRk=k8 * NRk_c_groupe, VRd=k8 * NRk_c_groupe / g_Mc, k8=k8)


# ---------------------------------------------------------------- cordons
def segments_cordons(forme, dims):
    """Les cordons poteau / platine, tous en cordon d'angle : profilé en I —
    faces extérieures et intérieures des ailes, deux faces de l'âme ; profil
    creux — périmètre extérieur. Segments ``(x1, y1, x2, y2)``."""
    h, b = dims["h"], dims["b"]
    if forme == "I":
        tw, tf = dims["tw"], dims["tf"]
        hi = h / 2 - tf
        return [(-b / 2, h / 2, b / 2, h / 2), (-b / 2, -h / 2, b / 2, -h / 2),
                (-b / 2, hi, -tw / 2, hi), (tw / 2, hi, b / 2, hi),
                (-b / 2, -hi, -tw / 2, -hi), (tw / 2, -hi, b / 2, -hi),
                (-tw / 2, -hi, -tw / 2, hi), (tw / 2, -hi, tw / 2, hi)]
    return [(-b / 2, h / 2, b / 2, h / 2), (-b / 2, -h / 2, b / 2, -h / 2),
            (b / 2, -h / 2, b / 2, h / 2), (-b / 2, -h / 2, -b / 2, h / 2)]


def cordons(forme, dims, a, fu, bw, g_M2, N, Mx, My, Vx, Vy, T):
    """Groupe de cordons traité « en ligne » (gorge a) : aire, inerties des
    lignes, contraintes au point le plus sollicité de l'aile (coin) et de
    l'âme, méthode directionnelle EN 1993-1-8 §4.5.3.2 :
    σeq = √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ fu/(βw·γM2), σ⊥ ≤ 0,9·fu/γM2, avec, pour
    une force normale au cordon dans le plan de la platine, σ⊥ = τ⊥ = σn/√2.
    N en valeur absolue (cordons supposés transmettre aussi la compression)."""
    segs = segments_cordons(forme, dims)
    A = Ix = Iy = 0.0
    A_h = A_v = 0.0
    for (x1, y1, x2, y2) in segs:
        L = math.hypot(x2 - x1, y2 - y1)
        xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        vertical = abs(x2 - x1) < 1e-9
        A += a * L
        Ix += a * (L * yc ** 2 + (L ** 3 / 12.0 if vertical else 0.0))
        Iy += a * (L * xc ** 2 + (0.0 if vertical else L ** 3 / 12.0))
        if vertical:
            A_v += a * L
        else:
            A_h += a * L
    Ip = Ix + Iy
    h, b = dims["h"], dims["b"]
    xm, ym = b / 2, h / 2
    if forme == "I":
        tw, tf = dims["tw"], dims["tf"]
        pts = [("aile (coin)", xm, ym, False), ("âme (extrémité)", tw / 2, h / 2 - tf, True)]
    else:
        pts = [("paroi (coin)", xm, ym, False), ("paroi latérale (coin)", xm, ym, True)]
    pire = None
    for nom, x, y, vertical in pts:
        sig_n = abs(N) / A + abs(Mx) * abs(y) / Ix + abs(My) * abs(x) / Iy
        if vertical:
            sig_n += abs(T) * abs(y) / Ip
            tau_p = abs(Vy) / A_v + abs(T) * abs(x) / Ip
        else:
            sig_n += abs(T) * abs(x) / Ip
            tau_p = abs(Vx) / A_h + abs(T) * abs(y) / Ip
        s_perp = sig_n / math.sqrt(2.0)
        s_eq = math.sqrt(2.0 * sig_n ** 2 + 3.0 * tau_p ** 2)
        cand = dict(point=nom, sig_n=sig_n, tau_par=tau_p, s_perp=s_perp, s_eq=s_eq)
        if pire is None or s_eq > pire["s_eq"]:
            pire = cand
    pire.update(A=A, Ix=Ix, Iy=Iy, Ip=Ip, L=A / a if a else 0.0, f_eq=fu / (bw * g_M2), f_perp=0.9 * fu / g_M2)
    return pire


# --------------------------------------------------------------- rigidité
def rigidite(Ec, A_tstub, leff, m, tp, n_b, As, Lb, levier, z, Ic, Lc, contrev):
    """Rigidité initiale Sj,ini = E·z²/(1/k13 + 1/k15 + 1/k16) — EN 1993-1-8
    §6.3.4, Tableau 6.11 : k13 = Ec·√(beff·leff)/(1,275·E), k15 = 0,85·leff·tp³/m³
    (0,425 sans effet de levier), k16 = 1,6·As/Lb (2,0 sans effet de levier),
    As = aire totale des tiges de la rangée. Seuil « rigide » §5.2.2.5 :
    30·E·Ic/Lc (ossature non contreventée), 48·E·Ic/Lc (contreventée, cas
    enveloppe)."""
    E = E_ACIER
    k13 = Ec * math.sqrt(max(A_tstub, 0.0)) / (1.275 * E)
    k15 = (0.85 if levier else 0.425) * leff * tp ** 3 / max(m, 1.0) ** 3
    k16 = (1.6 if levier else 2.0) * n_b * As / max(Lb, 1.0)
    inv = sum(1.0 / k for k in (k13, k15, k16) if k > 0)
    Sj = E * z ** 2 / inv if inv > 0 else 0.0
    seuil = (48.0 if contrev else 30.0) * E * Ic / Lc if Lc > 0 else INF
    return dict(k13=k13, k15=k15, k16=k16, Sj=Sj, seuil=seuil, rigide=Sj >= seuil, z=z)


def inertie_wpl(forme, dims):
    """(Iy, Wpl,y) du profilé autour de l'axe fort (congés négligés)."""
    h, b = dims["h"], dims["b"]
    if forme == "I":
        tw, tf = dims["tw"], dims["tf"]
        I = (b * h ** 3 - (b - tw) * (h - 2 * tf) ** 3) / 12.0
        Wpl = b * tf * (h - tf) + tw * (h - 2 * tf) ** 2 / 4.0
    else:
        t = dims["t"]
        I = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12.0
        Wpl = b * t * (h - t) + 2 * t * (h - 2 * t) ** 2 / 4.0
    return I, Wpl
