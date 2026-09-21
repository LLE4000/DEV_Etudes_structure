# -*- coding: utf-8 -*-
"""Formules et substitutions numériques des vérifications — écran (onglet
Vérifications) et note d'une page.

Chaque vérification est écrite en deux lignes au plus : *grandeur = formule
= substitution numérique = résultat*, puis le taux. Les nombres sont ceux
du moteur (``R``) ; la formule est un gabarit Python (``template``) dont les
identifiants sont remplacés par le symbole (forme littérale) ou par la
valeur formatée (forme numérique). C'est ce qui rend chaque substitution
TESTABLE : ``verifier(sub)`` évalue le gabarit avec les nombres tels
qu'ils sont imprimés et doit retrouver le résultat du moteur (à l'arrondi
d'affichage près) — la leçon de l'audit béton du 21/09/2026.

Rien n'est recalculé pour l'affichage : le résultat de chaque substitution
est une grandeur nommée par le moteur.
"""
import math
import re
from dataclasses import dataclass, field

from acier.js import N, mn
from acier.formats import F, pct

INF = 1e8


@dataclass
class Sub:
    """Une substitution : ``nom = template(symboles) = template(valeurs) =
    résultat unité``. ``vals`` : identifiant → (symbole, valeur)."""
    nom: str
    template: str
    vals: dict = field(default_factory=dict)
    res: float = 0.0
    unit: str = ""
    nd: int = None          # décimales du résultat (None : automatique ; -1 : pourcentage)


# --- formatage ----------------------------------------------------------
def decimales(v):
    """Décimales d'affichage : entier → 0 ; sinon quatre chiffres significatifs."""
    if v is None or not math.isfinite(v):
        return 0
    if abs(v - round(v)) < 1e-9:
        return 0
    if v == 0:
        return 0
    return max(0, min(4, 3 - int(math.floor(math.log10(abs(v))))))


def fmt(v, nd=None):
    """Un nombre tel qu'il est imprimé (virgule, sans zéros inutiles)."""
    if v is None or not math.isfinite(v):
        return "—"
    nd = decimales(v) if nd is None else nd
    s = F(v, nd)
    if "," in s:
        s = s.rstrip("0").rstrip(",")
    return s.replace("-", "−")


def nombre(s):
    """La valeur d'un nombre imprimé."""
    return float(s.replace("−", "-").replace(",", "."))


_IDENT = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_(])")
_FONCTIONS = {"sqrt", "min", "max", "abs", "pi"}


_KILO = "⁣k⁣"          # jeton d'un facteur 1000 (unités), posé avant la substitution


def _habiller(expr, numerique):
    """Opérateurs et constantes en typographie : ** → exposant, sqrt → √,
    abs → | |, * → · (littéral) ou × (numérique), point décimal → virgule,
    facteurs d'unités → 10³ / 10⁶."""
    expr = expr.replace("**2", "²").replace("**3", "³")
    expr = re.sub(r"sqrt\(3\)", "√3", expr)
    expr = expr.replace("sqrt(", "√(")
    expr = re.sub(r"abs\(([^()]*)\)", r"|\1|", expr)
    expr = expr.replace("pi", "π")
    expr = re.sub(r"(?<![A-Za-z])(\d+)\.(\d+)", r"\1,\2", expr)
    expr = expr.replace("1e6", "10⁶").replace(_KILO, "10³")
    expr = expr.replace("*", " × " if numerique else "·")
    expr = expr.replace(" - ", " − ")
    return expr


def _sans_unites(t):
    return re.sub(r"(\*|/)1(?:000|e6)(?![0-9.])", "", t)


def litteral(s):
    """La forme littérale (symboles), sans les facteurs d'unités."""
    def rep(m):
        ident = m.group(1)
        if ident in _FONCTIONS:
            return ident
        return s.vals[ident][0] if ident in s.vals else ident
    return _habiller(_IDENT.sub(rep, _sans_unites(s.template)), False)


def numerique(s):
    """La forme numérique (valeurs imprimées)."""
    t = re.sub(r"(?<=[*/])1000(?![0-9.])", _KILO, s.template)

    def rep(m):
        ident = m.group(1)
        if ident in _FONCTIONS:
            return ident
        return fmt(s.vals[ident][1]) if ident in s.vals else ident
    return _habiller(_IDENT.sub(rep, t), True)


def resultat(s):
    if s.nd == -1:
        return pct(s.res, 1)
    return fmt(s.res, s.nd) + (" " + s.unit if s.unit else "")


def rendre(s):
    """« Fv,Rd = αv·fub·A/γM2 = 0,6 × 800 × 245 / 1,25 = 94,08 kN »."""
    if not s.template:
        return s.nom + " = " + resultat(s)
    lit = litteral(s)
    num = numerique(s)
    parts = [s.nom]
    if lit != s.nom:
        parts.append(lit)
    if num != lit and num != fmt(s.res, s.nd):
        parts.append(num)
    parts.append(resultat(s))
    return " = ".join(parts)


def evaluer(s):
    """Évalue le gabarit avec les nombres tels qu'ils sont imprimés."""
    env = {k: nombre(fmt(v[1])) for k, v in s.vals.items()}
    env.update(sqrt=math.sqrt, min=min, max=max, abs=abs, pi=math.pi)
    return eval(s.template, {"__builtins__": {}}, env)  # noqa: S307 — gabarits internes


def verifier(s, tol=0.012):
    """``(ok, valeur évaluée)`` : la substitution imprimée redonne-t-elle le
    résultat du moteur ? Tolérance relative : l'arrondi d'affichage."""
    if not s.template or s.res is None or not math.isfinite(s.res) or abs(s.res) > INF:
        return True, s.res
    v = evaluer(s)
    ref = s.res
    if abs(ref) < 1e-9:
        return abs(v) < 1e-6, v
    return abs(v - ref) / abs(ref) <= tol, v


# --- les substitutions de chaque vérification -----------------------------
def _v(sym, val):
    return (sym, N(val) if not isinstance(val, (int, float)) else val)


def _eta(c, template="Ed/Rd", vals=None):
    if vals is None:
        vals = dict(Ed=("Ed", c.Ed), Rd=("Rd", c.Rd))
    return Sub("η", template, vals, c.eta, "", -1)


def _fv_rd(R):
    u = R.u
    return Sub("Fv,Rd", "av*fub*A/gM2/1000", dict(av=("αv", R.a_v), fub=("fub", R.f_ub), A=("A", R.A_cis),
                                                   gM2=("γM2", u.g_M2)), R.Fv_Rd, "kN")


def _bearing(R, K, t_sym, t, fu, div):
    """Les deux résistances de pression diamétrale et l'interaction."""
    u = R.u
    base = dict(fu=("fu", fu), d=("d", R.d_b), t=(t_sym, t), gM2=("γM2", u.g_M2))
    kt = "*kt" if R.k_trou != 1 else ""
    if kt:
        base["kt"] = ("0,8 (trou)", R.k_trou)
    fbv = Sub("Fb,ver,Rd", "k1v*abv*fu*d*t/gM2/1000" + kt,
              dict(k1v=("k1,ver", R["k1v_" + K]), abv=("αb,ver", R["abv_" + K]), **base), R["Fbv_" + K], "kN")
    fbh = Sub("Fb,hor,Rd", "k1h*abh*fu*d*t/gM2/1000" + kt,
              dict(k1h=("k1,hor", R["k1h_" + K]), abh=("αb,hor", R["abh_" + K]), **base), R["Fbh_" + K], "kN")
    return fbv, fbh


def _pd(R, c, K, t_sym, t, fu, X, div):
    fbv, fbh = _bearing(R, K, t_sym, t, fu, div)
    Fz, Fx = R["Fz_" + X], R["Fx_" + X]
    d = "/2" if div == 2 else ""
    eta = Sub("η", f"sqrt((Fz{d}/Fbv)**2 + (Fx{d}/Fbh)**2)",
              dict(Fz=("Fz", Fz), Fx=("Fx", Fx), Fbv=("Fb,ver,Rd", R["Fbv_" + K]), Fbh=("Fb,hor,Rd", R["Fbh_" + K])),
              R["ipd_" + K], "", -1)
    return [[fbv, fbh], [eta]]


def _efforts_S(R):
    u = R.u
    n = R.n_S
    if R.Ip_S > 0:
        fz = Sub("Fz", "V/n + MS*1000*xm/Ip", dict(V=("VEd", u.V_Ed), n=("n", n), MS=("MS", R.M_S), xm=("xmax", R.xm_S),
                                                    Ip=("Ip", R.Ip_S)), R.Fz_S, "kN")
        fx = Sub("Fx", "abs(NEd)/n + MS*1000*ym/Ip", dict(NEd=("NEd", u.N_Ed), n=("n", n), MS=("MS", R.M_S),
                                                          ym=("ymax", R.ym_S), Ip=("Ip", R.Ip_S)), R.Fx_S, "kN")
    else:
        fz = Sub("Fz", "V/n", dict(V=("VEd", u.V_Ed), n=("n", n)), R.Fz_S, "kN")
        fx = Sub("Fx", "abs(NEd)/n", dict(NEd=("NEd", u.N_Ed), n=("n", n)), R.Fx_S, "kN")
    f = Sub("FEd", "sqrt(Fz**2 + Fx**2)", dict(Fz=("Fz", R.Fz_S), Fx=("Fx", R.Fx_S)), R.F_S, "kN")
    return fz, fx, f


def _efforts_P(R):
    u = R.u
    n = R.n_P; H = abs(u.H_Ed)
    if R.Ip_P > 0 and R.M_P > 0:
        fz = Sub("Fz", "V/2/n + MP*1000*xm/Ip", dict(V=("VEd", u.V_Ed), n=("n", n), MP=("MP", R.M_P), xm=("xmax", R.xm_P),
                                                      Ip=("Ip", R.Ip_P)), R.Fz_P, "kN")
        fx = Sub("Fx", "H/2/n + MP*1000*ym/Ip", dict(H=("HEd", H), n=("n", n), MP=("MP", R.M_P), ym=("ymax", R.ym_P),
                                                      Ip=("Ip", R.Ip_P)), R.Fx_P, "kN")
    else:
        fz = Sub("Fz", "V/2/n", dict(V=("VEd", u.V_Ed), n=("n", n)), R.Fz_P, "kN")
        fx = Sub("Fx", "H/2/n", dict(H=("HEd", H), n=("n", n)), R.Fx_P, "kN")
    f = Sub("FEd", "sqrt(Fz**2 + Fx**2)", dict(Fz=("Fz", R.Fz_P), Fx=("Fx", R.Fx_P)), R.F_P, "kN")
    return fz, fx, f


def _bloc(R, nom, Ant, Anv, fu, fy, mult):
    u = R.u
    m = "2*" if mult == 2 else ""
    return Sub(nom, f"{m}(0.5*fu*Ant/gM2n + fy*Anv/(sqrt(3)*gM0))/1000",
               dict(fu=("fu", fu), Ant=("Ant", Ant), gM2n=("γM2", u.g_M2n), fy=("fy", fy), Anv=("Anv", Anv),
                    gM0=("γM0", u.g_M0)), None, "kN")


def lignes(R, c):
    """Les lignes de substitution d'une vérification active : liste de
    lignes, chaque ligne = liste de ``Sub``. Vide si la vérification n'a
    pas de gabarit (le texte ``vals`` du moteur sert alors)."""
    u = R.u; k = c.key; d0 = R.d_0
    V = u.V_Ed; NEd = u.N_Ed; H = abs(u.H_Ed); Mabs = abs(u.M_Ed)
    L = []
    if k == "bv_S":
        fz, fx, f = _efforts_S(R)
        frd = Sub("FRd", "2*FvRd*bLf", dict(FvRd=("Fv,Rd", R.Fv_Rd), bLf=("βLf", R.bLf_S)), c.Rd, "kN")
        L = [[_fv_rd(R), frd], [fz, fx, f], [_eta(c, "FEd/FRd", dict(FEd=("FEd", c.Ed), FRd=("FRd", c.Rd)))]]
    elif k == "bv_P":
        fz, fx, f = _efforts_P(R)
        frd = Sub("FRd", "kr*FvRd*bLf", dict(kr=("k", u.k_rot), FvRd=("Fv,Rd", R.Fv_Rd), bLf=("βLf", R.bLf_P)), c.Rd, "kN")
        L = [[_fv_rd(R), frd], [fz, fx, f], [_eta(c, "FEd/FRd", dict(FEd=("FEd", c.Ed), FRd=("FRd", c.Rd)))]]
    elif k == "bt_P":
        ft = Sub("Ft,Rd", "0.9*fub*As/gM2/1000", dict(fub=("fub", R.f_ub), As=("As", R.As_b), gM2=("γM2", u.g_M2)), R.Ft_Rd, "kN")
        bp = Sub("Bp,Rd", "0.6*pi*dm*min(tc*fuC, twP*fuP)/gM2/1000",
                 dict(dm=("dm", R.d_m), tc=("tc", R.t_C), fuC=("fu,c", R.fu_C), twP=("tw,P", R.tw_P), fuP=("fu,P", R.fu_P),
                      gM2=("γM2", u.g_M2)), R.Bp_Rd, "kN")
        vals = dict(NEd=("NEd", max(NEd, 0)), n=("n", R.n_P), H=("HEd", H), z=("z", R.zeff), p3=("p3", R.p_3))
        t = "NEd/(2*n) + H*z/(p3*n)"
        if R.n1_P > 1:
            t = "NEd/(2*n) + M*1000*ym/(2*n2*p1**2*n1*(n1**2 - 1)/12) + H*z/(p3*n)"
            vals.update(M=("MEd", Mabs), ym=("ymax", R.ym_P), n2=("n2", R.n2_P), p1=("p1", R.p1_P), n1=("n1", R.n1_P))
        fted = Sub("Ft,Ed", t, vals, R.Ft_P, "kN")
        L = [[ft, bp], [fted], [_eta(c, "FtEd/min(FtRd, BpRd)", dict(FtEd=("Ft,Ed", c.Ed), FtRd=("Ft,Rd", R.Ft_Rd), BpRd=("Bp,Rd", R.Bp_Rd)))]]
    elif k == "bi_P":
        L = [[Sub("η", "FvEd/(FvRd*bLf) + FtEd/(1.4*FtRd)",
                  dict(FvEd=("Fv,Ed", R.F_P), FvRd=("Fv,Rd", R.Fv_Rd), bLf=("βLf", R.bLf_P), FtEd=("Ft,Ed", R.Ft_P),
                       FtRd=("Ft,Rd", R.Ft_Rd)), c.eta, "", -1)]]
    elif k in ("gl_S", "gl_P"):
        catB = u.cat == "B"; ks = u.k_ser if catB else 1; g3 = u.g_M3s if catB else u.g_M3
        fp = Sub("Fp,C", "0.7*fub*As/1000", dict(fub=("fub", R.f_ub), As=("As", R.As_b)), R.Fp_C, "kN")
        if k == "gl_S":
            fs = Sub("Fs,Rd", "ks*2*mu*FpC/gM3", dict(ks=("ks", u.k_s), mu=("μ", u.mu_s), FpC=("Fp,C", R.Fp_C),
                                                      gM3=("γM3,ser" if catB else "γM3", g3)), c.Rd, "kN")
            ed = Sub("Fs,Ed", "FEd*kser", dict(FEd=("FEd", R.F_S), kser=("kser", ks)), c.Ed, "kN")
        else:
            fs = Sub("Fs,Rd", "ks*mu*(FpC - 0.8*FtEd*kser)/gM3",
                     dict(ks=("ks", u.k_s), mu=("μ", u.mu_s), FpC=("Fp,C", R.Fp_C), FtEd=("Ft,Ed", R.Ft_P), kser=("kser", ks),
                          gM3=("γM3,ser" if catB else "γM3", g3)), c.Rd, "kN")
            ed = Sub("Fs,Ed", "FEd*kser", dict(FEd=("FEd", R.F_P), kser=("kser", ks)), c.Ed, "kN")
        L = [[fp, fs], [ed, _eta(c, "FsEd/FsRd", dict(FsEd=("Fs,Ed", c.Ed), FsRd=("Fs,Rd", c.Rd)))]]
    elif k == "pdB":
        L = _pd(R, c, "B", "tc", R.t_C, R.fu_C, "S", 2)
    elif k == "pdA":
        L = _pd(R, c, "A", "tc", R.t_C, R.fu_C, "P", 1)
    elif k == "pdS":
        L = _pd(R, c, "S", "tw", R.tw_S, R.fu_S, "S", 1)
    elif k == "pdP":
        L = _pd(R, c, "P", "tw", R.tw_P, R.fu_P, "P", 1)
    elif k in ("cgB", "cgA"):
        L = [[Sub("VRd,g", "2*hc*tc*fy/(1.27*sqrt(3)*gM0)/1000",
                  dict(hc=("hc", R.L_C), tc=("tc", R.t_C), fy=("fy", R.fy_C), gM0=("γM0", u.g_M0)), c.Rd, "kN")],
             [_eta(c, "V/VRd", dict(V=("VEd", V), VRd=("VRd,g", c.Rd)))]]
    elif k in ("cnB", "cnA"):
        n1 = R.n1_S if k == "cnB" else R.n1_P
        av = Sub("Av,net", "tc*(hc - n1*d0)", dict(tc=("tc", R.t_C), hc=("hc", R.L_C), n1=("n1", n1), d0=("d0", d0)),
                 R.t_C * (R.L_C - n1 * d0), "mm²", 0)
        L = [[av, Sub("VRd,n", "2*Avn*fu/(sqrt(3)*gM2n)/1000", dict(Avn=("Av,net", av.res), fu=("fu", R.fu_C), gM2n=("γM2", u.g_M2n)), c.Rd, "kN")],
             [_eta(c, "V/VRd", dict(V=("VEd", V), VRd=("VRd,n", c.Rd)))]]
    elif k in ("cbB", "cbA"):
        X = "S" if k == "cbB" else "P"
        e2 = R.e2a_S if k == "cbB" else R.e2a_P; n2 = R["n2_" + X]; p2 = u.p2_S if X == "S" else u.p2_P
        e1 = R.e1_S if k == "cbB" else R.e1_P; n1 = R["n1_" + X]
        Ant = R.Ant_B if k == "cbB" else R.Ant_A; Anv = R.Anv_B if k == "cbB" else R.Anv_A
        ant = Sub("Ant", "tc*(e2 + (n2 - 1)*p2 - (n2 - 0.5)*d0)",
                  dict(tc=("tc", R.t_C), e2=("e2", e2), n2=("n2", n2), p2=("p2", p2), d0=("d0", d0)), Ant, "mm²", 0)
        anv = Sub("Anv", "tc*(hc - e1 - (n1 - 0.5)*d0)",
                  dict(tc=("tc", R.t_C), hc=("hc", R.L_C), e1=("e1", e1), n1=("n1", n1), d0=("d0", d0)), Anv, "mm²", 0)
        b = _bloc(R, "Veff,2,Rd", Ant, Anv, R.fu_C, R.fy_C, 2); b.res = c.Rd
        L = [[ant, anv], [b, _eta(c, "V/VRd", dict(V=("VEd", V), VRd=("Veff,2,Rd", c.Rd)))]]
    elif k == "flB":
        wel = Sub("Wel", "tc*hc**2/6", dict(tc=("tc", R.t_C), hc=("hc", R.L_C)), R.t_C * R.L_C * R.L_C / 6, "mm³", 0)
        mrd = Sub("MRd", "Wel*fy/gM0/1e6", dict(Wel=("Wel", wel.res), fy=("fy", R.fy_C), gM0=("γM0", u.g_M0)), c.Rd, "kNm")
        med = Sub("MEd", "V/2*z/1000 + M/2", dict(V=("VEd", V), z=("z", R.zeff), M=("|MEd,parasite|", Mabs)), c.Ed, "kNm")
        L = [[wel, mrd], [med, _eta(c, "MEd/MRd", dict(MEd=("MEd", c.Ed), MRd=("MRd", c.Rd)))]]
    elif k == "flA":
        bras = ("gA", R.g_A) if R.bolt_P else ("bA", R.b_A)
        med = Sub("MEd", "V/2*(g - tc/2)/1000", dict(V=("VEd", V), g=bras, tc=("tc", R.t_C)), c.Ed, "kNm")
        mrd = Sub("MRd", "tc*hc**2/6*fy/gM0/1e6", dict(tc=("tc", R.t_C), hc=("hc", R.L_C), fy=("fy", R.fy_C), gM0=("γM0", u.g_M0)), c.Rd, "kNm")
        L = [[med, mrd], [_eta(c, "MEd/MRd", dict(MEd=("MEd", c.Ed), MRd=("MRd", c.Rd)))]]
    elif k == "tsA":
        mpl = Sub("Mpl", "0.25*leff*tc**2*fy/gM0/1e6", dict(leff=("Σleff", R.leff_T), tc=("tc", R.t_C), fy=("fy", R.fy_C),
                                                          gM0=("γM0", u.g_M0)), R.Mpl_T, "kNm")
        f1 = Sub("FT,1", "(8*n - 2*ew)*Mpl*1000/(2*m*n - ew*(m + n))",
                 dict(n=("n", R.n_T), ew=("ew", R.ew_T), Mpl=("Mpl", R.Mpl_T), m=("m", R.m_T)), R.FT_1, "kN")
        f2 = Sub("FT,2", "(2*Mpl*1000 + n*SFt)/(m + n)", dict(Mpl=("Mpl", R.Mpl_T), n=("n", R.n_T), SFt=("ΣFt,Rd", R.SFt_T), m=("m", R.m_T)), R.FT_2, "kN")
        f3 = Sub("FT,3", "SFt", dict(SFt=("ΣFt,Rd", R.SFt_T)), R.FT_3, "kN")
        L = [[mpl, f1], [f2, f3], [_eta(c, "NEd/min(FT1, FT2, FT3)", dict(NEd=("NEd", NEd), FT1=("FT,1", R.FT_1), FT2=("FT,2", R.FT_2), FT3=("FT,3", R.FT_3)))]]
    elif k in ("tnB", "tnS"):
        if k == "tnB":
            nu = Sub("Nu,Rd", "2*0.9*tc*(hc - n1*d0)*fu/gM2n/1000", dict(tc=("tc", R.t_C), hc=("hc", R.L_C), n1=("n1", R.n1_S),
                                                                       d0=("d0", d0), fu=("fu", R.fu_C), gM2n=("γM2", u.g_M2n)), c.Rd, "kN")
        else:
            nu = Sub("Nu,Rd", "0.9*tw*(hc - n1*d0)*fu/gM2n/1000", dict(tw=("tw", R.tw_S), hc=("hc", R.L_C), n1=("n1", R.n1_S),
                                                                     d0=("d0", d0), fu=("fu", R.fu_S), gM2n=("γM2", u.g_M2n)), c.Rd, "kN")
        L = [[nu], [_eta(c, "NEd/NuRd", dict(NEd=("NEd", NEd), NuRd=("Nu,Rd", c.Rd)))]]
    elif k == "tbB":
        Ah = R.Ant_B / R.t_C
        eB = mn(R.e1_S, R.e1bot_S)
        v1 = Sub("Veff,1,Rd (cas 1)", "(fu*2*tc*(n1 - 1)*(p1 - d0)/gM2n + fy*4*tc*Ah/(sqrt(3)*gM0))/1000",
                 dict(fu=("fu", R.fu_C), tc=("tc", R.t_C), n1=("n1", R.n1_S), p1=("p1", R.p1_S), d0=("d0", d0), gM2n=("γM2", u.g_M2n),
                      fy=("fy", R.fy_C), Ah=("Ant/tc", Ah), gM0=("γM0", u.g_M0)), R.Vt1_B, "kN")
        v2 = Sub("Veff,1,Rd (cas 2)", "(fu*2*tc*(e1 + (n1 - 1)*p1 - (n1 - 0.5)*d0)/gM2n + fy*2*tc*Ah/(sqrt(3)*gM0))/1000",
                 dict(fu=("fu", R.fu_C), tc=("tc", R.t_C), e1=("e1,min", eB), n1=("n1", R.n1_S), p1=("p1", R.p1_S), d0=("d0", d0),
                      gM2n=("γM2", u.g_M2n), fy=("fy", R.fy_C), Ah=("Ant/tc", Ah), gM0=("γM0", u.g_M0)), R.Vt2_B, "kN")
        L = [[v1], [v2, _eta(c, "NEd/min(V1, V2)", dict(NEd=("NEd", NEd), V1=("cas 1", R.Vt1_B), V2=("cas 2", R.Vt2_B)))]]
    elif k == "tbS":
        Ah = R.Ant_S / R.tw_S
        v1 = Sub("Veff,1,Rd (cas 1)", "(fu*tw*(n1 - 1)*(p1 - d0)/gM2n + fy*2*tw*Ah/(sqrt(3)*gM0))/1000",
                 dict(fu=("fu", R.fu_S), tw=("tw", R.tw_S), n1=("n1", R.n1_S), p1=("p1", R.p1_S), d0=("d0", d0), gM2n=("γM2", u.g_M2n),
                      fy=("fy", R.fy_S), Ah=("Ant/tw", Ah), gM0=("γM0", u.g_M0)), R.Vt1_S, "kN")
        L = [[v1]]
        if R.cas_g > 0:
            v2 = Sub("Veff,1,Rd (cas 2)", "(fu*tw*(e1b + (n1 - 1)*p1 - (n1 - 0.5)*d0)/gM2n + fy*tw*Ah/(sqrt(3)*gM0))/1000",
                     dict(fu=("fu", R.fu_S), tw=("tw", R.tw_S), e1b=("e1,b", R.e1b_S), n1=("n1", R.n1_S), p1=("p1", R.p1_S), d0=("d0", d0),
                          gM2n=("γM2", u.g_M2n), fy=("fy", R.fy_S), Ah=("Ant/tw", Ah), gM0=("γM0", u.g_M0)), R.Vt2_S, "kN")
            L.append([v2, _eta(c, "NEd/min(V1, V2)", dict(NEd=("NEd", NEd), V1=("cas 1", R.Vt1_S), V2=("cas 2", R.Vt2_S)))])
        else:
            L.append([_eta(c, "NEd/V1", dict(NEd=("NEd", NEd), V1=("Veff,1,Rd", c.Rd)))])
    elif k == "vgS":
        if R.cas_g == 0:
            av = Sub("Av", "max(A - 2*b*tf + (tw + 2*r)*tf, (h - 2*tf)*tw)",
                     dict(A=("A", R.Asec_S), b=("b", R.b_S), tf=("tf", R.tf_S), tw=("tw", R.tw_S), r=("r", R.r_S), h=("h", R.h_S)), R.Av_S, "mm²", 0)
        elif R.cas_g == 1:
            av = Sub("Av", "AT - b*tf + (tw + 2*r)*tf/2",
                     dict(AT=("A,T", R.A_Tee), b=("b", R.b_S), tf=("tf", R.tf_S), tw=("tw", R.tw_S), r=("r", R.r_S)), R.Av_S, "mm²", 0)
        else:
            av = Sub("Av", "tw*hT", dict(tw=("tw", R.tw_S), hT=("h − dc,sup − dc,inf", R.h_T)), R.Av_S, "mm²", 0)
        L = [[av, Sub("VRd,g", "Av*fy/(sqrt(3)*gM0)/1000", dict(Av=("Av", R.Av_S), fy=("fy", R.fy_S), gM0=("γM0", u.g_M0)), c.Rd, "kN")],
             [_eta(c, "V/VRd", dict(V=("VEd", V), VRd=("VRd,g", c.Rd)))]]
    elif k == "vnS":
        avn = R.Av_S - R.n1_S * d0 * R.tw_S
        L = [[Sub("Av,net", "Av - n1*d0*tw", dict(Av=("Av", R.Av_S), n1=("n1", R.n1_S), d0=("d0", d0), tw=("tw", R.tw_S)), avn, "mm²", 0),
              Sub("VRd,n", "Avn*fu/(sqrt(3)*gM2n)/1000", dict(Avn=("Av,net", avn), fu=("fu", R.fu_S), gM2n=("γM2", u.g_M2n)), c.Rd, "kN")],
             [_eta(c, "V/VRd", dict(V=("VEd", V), VRd=("VRd,n", c.Rd)))]]
    elif k == "vbS":
        ant = Sub("Ant", "tw*(e2b + (n2 - 1)*p2 - (n2 - 0.5)*d0)",
                  dict(tw=("tw", R.tw_S), e2b=("e2,b", R.e2b_S), n2=("n2", R.n2_S), p2=("p2", u.p2_S), d0=("d0", d0)), R.Ant_S, "mm²", 0)
        anv = Sub("Anv", "tw*(e1b + (n1 - 1)*p1 - (n1 - 0.5)*d0)",
                  dict(tw=("tw", R.tw_S), e1b=("e1,b", R.e1b_S), n1=("n1", R.n1_S), p1=("p1", R.p1_S), d0=("d0", d0)), R.Anv_S, "mm²", 0)
        b = _bloc(R, "Veff,2,Rd", R.Ant_S, R.Anv_S, R.fu_S, R.fy_S, 1); b.res = c.Rd
        L = [[ant, anv], [b, _eta(c, "V/VRd", dict(V=("VEd", V), VRd=("Veff,2,Rd", c.Rd)))]]
    elif k == "mN":
        med = Sub("MEd", "V*(gh + cn)/1000 + M", dict(V=("VEd", V), gh=("gh", u.g_h), cn=("c", u.l_n), M=("|MEd,parasite|", Mabs)), c.Ed, "kNm")
        subs = [Sub("Wel,N", None, {}, R.W_N, "mm³", 0)]
        if V > 0.5 * R.VRd_gS:
            subs.append(Sub("ρ", "max(0, 1 - (2*V/Vpl - 1)**2)", dict(V=("VEd", V), Vpl=("Vpl,N,Rd", R.VRd_gS)), R.rho_N, "", 3))
            mrd = Sub("Mv,N,Rd", "Wel*fy/gM0/1e6*rho", dict(Wel=("Wel,N", R.W_N), fy=("fy", R.fy_S), gM0=("γM0", u.g_M0), rho=("ρ", R.rho_N)), c.Rd, "kNm")
        else:
            mrd = Sub("Mv,N,Rd", "Wel*fy/gM0/1e6", dict(Wel=("Wel,N", R.W_N), fy=("fy", R.fy_S), gM0=("γM0", u.g_M0)), c.Rd, "kNm")
        L = [[med] + subs, [mrd, _eta(c, "MEd/MRd", dict(MEd=("MEd", c.Ed), MRd=("Mv,N,Rd", c.Rd)))]]
    elif k == "m2":
        med = Sub("MEd", "V*(gh + e2b + p2)/1000", dict(V=("VEd", V), gh=("gh", u.g_h), e2b=("e2,b", R.e2b_S), p2=("p2", u.p2_S)), c.Ed, "kNm")
        L = [[med, Sub("Mc,2,Rd", None, {}, c.Rd, "kNm")], [_eta(c, "MEd/MRd", dict(MEd=("MEd", c.Ed), MRd=("Mc,2,Rd", c.Rd)))]]
    elif k == "stN":
        el = Sub("h/tw", "h/tw", dict(h=("h", R.h_S), tw=("tw", R.tw_S)), R.el_N, "", 1)
        if R.el_N <= R.lim_N:
            cmax = Sub("c,max", "h", dict(h=("h", R.h_S)), c.Rd, "mm", 0)
        else:
            K = 160000 if R.fy_S <= 275 else 110000
            cmax = Sub("c,max", "K*h/(h/tw)**3", dict(K=("K", K), h=("h", R.h_S), tw=("tw", R.tw_S)), c.Rd, "mm", 0)
        L = [[el, Sub("limite", None, {}, R.lim_N, "", 1), cmax], [_eta(c, "c/cmax", dict(c=("c", u.l_n), cmax=("c,max", c.Rd)))]]
    elif k == "stD":
        L = [[Sub("dc,max", "h/2" if R.cas_g == 1 else "h/5", dict(h=("h", R.h_S)), c.Rd, "mm", 1)],
             [_eta(c, "dc/dcmax", dict(dc=("dc", c.Ed), dcmax=("dc,max", c.Rd)))]]
    elif k == "vlP":
        if R.bolt_P:
            av = Sub("Av", "tw*(et + (n1 - 1)*p1 + eb)", dict(tw=("tw", R.tw_P), et=("et", R.et_P), n1=("n1", R.n1_P), p1=("p1", R.p1_P),
                                                            eb=("eb", R.eb_P)), R.Av_P, "mm²", 0)
        else:
            av = Sub("Av", "tw*hc", dict(tw=("tw", R.tw_P), hc=("hc", R.L_C)), R.Av_P, "mm²", 0)
        L = [[av, Sub("VRd", "Av*fy/(sqrt(3)*gM0)/1000", dict(Av=("Av", R.Av_P), fy=("fy", R.fy_P), gM0=("γM0", u.g_M0)), c.Rd, "kN")],
             [_eta(c, "V/2/VRd", dict(V=("VEd", V), VRd=("VRd", c.Rd)))]]
    elif k == "vnP":
        avn = R.Av_P - R.n1_P * d0 * R.tw_P
        L = [[Sub("Av,net", "Av - n1*d0*tw", dict(Av=("Av", R.Av_P), n1=("n1", R.n1_P), d0=("d0", d0), tw=("tw", R.tw_P)), avn, "mm²", 0),
              Sub("VRd,n", "Avn*fu/(sqrt(3)*gM2n)/1000", dict(Avn=("Av,net", avn), fu=("fu", R.fu_P), gM2n=("γM2", u.g_M2n)), c.Rd, "kN")],
             [_eta(c, "V/2/VRd", dict(V=("VEd", V), VRd=("VRd,n", c.Rd)))]]
    elif k in ("wS", "wP"):
        X = k[1]
        w1 = abs(NEd) / 2 if X == "S" else H / 2
        w2 = 0 if X == "S" else max(NEd, 0) / 2
        w3 = R.M_S / 2 if X == "S" else R.M_P
        lh = u["lh_" + X]
        fv = Sub("fvw,d", "fu/(sqrt(3)*bw*gM2)", dict(fu=("fu", R["fuw_" + X]), bw=("βw", R["bww_" + X]), gM2=("γM2", u.g_M2)), R["fvw_" + X], "MPa")
        fw = Sub("Fw,Rd", "fvwd*a", dict(fvwd=("fvw,d", R["fvw_" + X]), a=("a", u["a_" + X])), R["Fw_" + X], "N/mm")
        base = dict(Lw=("ΣL", R["Lw_" + X]), Iw=("Ip", R["Iw_" + X]))
        qz = Sub("qz", "V/2*1000/Lw + M3*1e6*(lh - xg)/Iw", dict(V=("VEd", V), M3=("M", w3), lh=("ℓh", lh), xg=("xg", R["xg_" + X]), **base), R["qz_" + X], "N/mm")
        qx = Sub("qx", "w1*1000/Lw + M3*1e6*(hc/2)/Iw", dict(w1=("|NEd|/2" if X == "S" else "HEd/2", w1), M3=("M", w3), hc=("hc", R.L_C), **base), R["qx_" + X], "N/mm")
        qn = Sub("qn", "w2*1000/Lw", dict(w2=("0" if X == "S" else "NEd/2", w2), **base), R["qn_" + X], "N/mm")
        qw = Sub("Fw,Ed", "sqrt(qz**2 + qx**2 + qn**2)", dict(qz=("qz", R["qz_" + X]), qx=("qx", R["qx_" + X]), qn=("qn", R["qn_" + X])), R["qw_" + X], "N/mm")
        L = [[fv, fw], [qz, qx, qn, qw], [_eta(c, "FwEd/FwRd", dict(FwEd=("Fw,Ed", c.Ed), FwRd=("Fw,Rd", c.Rd)))]]
    elif k in ("waS", "waP"):
        L = [[Sub("a", None, {}, c.Rd, "mm", 0), Sub("a,min", None, {}, 3, "mm", 0)], [_eta(c, "amin/a", dict(amin=("a,min", 3), a=("a", c.Rd)))]]
    elif k in ("wlS", "wlP"):
        L = [[Sub("L,min", "max(30, 6*a)", dict(a=("a", u["a_" + k[2]])), c.Ed, "mm", 0), Sub("hc", None, {}, c.Rd, "mm", 0)],
             [_eta(c, "Lmin/hc", dict(Lmin=("L,min", c.Ed), hc=("hc", c.Rd)))]]
    return L


def textes(R, c):
    """Les lignes de texte d'une vérification (deux ou trois), ou — sans
    gabarit — la formule et les valeurs du moteur."""
    L = lignes(R, c)
    if not L:
        from .notation import ec
        out = [ec(c.formula)]
        if c.vals:
            out.append(ec(c.vals))
        return out
    return [" ; ".join(rendre(s) for s in ligne) for ligne in L]


def toutes(R):
    """``[(vérification, [Sub, …]), …]`` pour toutes les vérifications actives
    (pour le test qui recalcule chaque substitution)."""
    out = []
    for c in R.checks:
        if c.active:
            out.append((c, [s for ligne in lignes(R, c) for s in ligne]))
    return out
