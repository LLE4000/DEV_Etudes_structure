# -*- coding: utf-8 -*-
"""Le JOINT des prédalles et ses BARRES DE COUTURE (beton/predalle/joint.py, lot 5 — décisions du propriétaire du
13/11/2026) : des cas calculés À LA MAIN, puis des mutations du fichier qui font rougir le contrôle qui les garde.

  1. la couture automatique : un treillis 10/10/100/100 donne Ø10/100 (ses porteurs), un treillis 5/5/150/150 donne Ø6
     (le premier Ø du moteur) / 150, « n barres » est repris tel quel ;
  2. le joint, prédalle par défaut (h = 22, h_p = 6, C30/37, B500, γs 1,5, jeu 1, arrondi 0,5), M_x,inf = 30 kN·m :
     couture posée à 6,5 cm (axe 7,5), d = 14,5 cm, A_s,req = 689,7 mm² ≤ 785,4 mm² ; la pleine prédalle garde la base
     dans la peau (axe 4,5, d = 17,5) ; la hauteur du joint gouverne (h_u,min + 7,5) ; l₀ = 536,1 mm → 55 cm × 2 ;
  3. une couture saisie Ø12/150, un renfort 3 Ø12 et un y_G imposé : le renfort est gardé, le y_G imposé ne vaut pas
     au joint ; l₀ Ø12 = 643,3 mm → 65 cm ;
  4. le joint qui ne tient pas (M = 38) : la section et la prédalle sont non conformes, la pleine prédalle reste
     conforme et le garde (``pleine_predalle``) ; le gouvernant est le joint ;
  5. la couture hors de l'épaisseur (h = 10, h_p = 9) : d ≤ 0, l'avertissement ; un gros diamètre (Ø40) : §8.8 ; un
     renfort principal posé sur la prédalle : la couture est posée SUR lui (6 + 1,2 + 0,5 = 7,7 cm, axe 9,0) ;
  6. la saisie : les refus (Ø hors liste, espacement hors bornes, clé inconnue, type), l'idempotence de normaliser ;
  7. mutations : chaque règle retirée du fichier fait rougir son contrôle.

    /usr/bin/python3.12 tests/test_predalle_joint.py
"""
import copy
import math
import os
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.predalle import joint as J  # noqa: E402
from beton.predalle import moteur as M  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def proche(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


# ---- les valeurs À LA MAIN (C30/37, B500, γs = 1,5 ; α_b et μ : la table du bureau pour C30/37)
FYD = 500 / 1.5
FCTM = 0.30 * 30 ** (2 / 3)
AS_MIN_EC = 0.26 * FCTM / 500 * 100 * 22 * 100           # mm² (b = 100 cm, h = 22 cm)
ALPHA_B, MU = 12.96, 0.1709


def hu(m):
    return math.sqrt(m * 1e6 / (ALPHA_B * 100 * 10 * MU)) / 10


def aire(d):
    return math.pi * d * d / 4


def l0_main(d):
    """EN 1992-1-1 §8.7.3 : f_ctk,0,05 = 0,7·f_ctm, f_ctd = /1,5, f_bd = 2,25·f_ctd, l_b,rqd = Ø/4·(500/1,15)/f_bd, α6 = 1,5."""
    fbd = 2.25 * (0.7 * FCTM / 1.5)
    return 1.5 * d / 4 * (500 / 1.15) / fbd


def prédalle(**sec):
    p = M.nouvelle_predalle()
    p["sections"][0].update(sec)
    return p


def cas(fonction):
    """Un cas qui lève (une mutation peut faire lever le moteur) rend None : ses contrôles rougissent, sans arrêter."""
    try:
        return fonction()
    except Exception as e:  # noqa: BLE001
        return {"_erreur": repr(e)}


# ---------------------------------------------------------------- les cas, en fonctions (rejoués par les mutations)
def cas_auto():
    s = M.nouvelle_predalle()["sections"][0]
    t10 = J.couche_couture(s, "x")
    s5 = copy.deepcopy(s)
    s5["faces"]["inf_x"]["couches"][0]["treillis"] = "5/5/150/150"
    t5 = J.couche_couture(s5, "x")
    sn = copy.deepcopy(s)
    sn["faces"]["inf_x"]["couches"][0].update(type="n_barres", n=5, d=10)
    tn = J.couche_couture(sn, "x")
    return {"t10": (t10["type"], t10["d"], t10["esp"]), "t5": (t5["type"], t5["d"], t5["esp"]), "tn": (tn["type"], tn["n"], tn["d"])}


def cas_defaut():
    r = M.verifier(prédalle(Mx_inf=30, My_inf=20))
    s = r["sections"][0]
    return {"s": s, "r": r, "v": {x["cle"]: x for x in s["verifications"]}}


def cas_saisie():
    p = prédalle(Mx_inf=30)
    s = p["sections"][0]
    s["faces"]["inf_x"]["couches"].append(M.nouvelle_couche(False))        # un renfort 3 Ø12, posé sur la prédalle
    s["faces"]["inf_x"]["yG"] = 4.0
    s["couture"] = {"d": 12, "esp": 150}
    r = M.verifier(p)
    return {"s": r["sections"][0], "r": r}


def cas_rouge():
    r = M.verifier(prédalle(Mx_inf=38))
    return {"s": r["sections"][0], "r": r}


def cas_pas():
    """(lot 6) le PAS d'arrondi de la longueur, une option de la note : 1 et 10 cm au lieu de 5."""
    un = M.verifier(dict(prédalle(Mx_inf=30), pas_couture=1))["sections"][0]["joint"]["longueur"]
    dix = M.verifier(dict(prédalle(Mx_inf=30), pas_couture=10.0))["sections"][0]["joint"]["longueur"]
    return {"un": un, "dix": dix}


def cas_bords():
    hors = M.verifier(dict(prédalle(Mx_inf=1), h=10, h_pre=9))
    gros = M.verifier(dict(prédalle(Mx_inf=1, couture={"d": 40, "esp": 200})))
    p = prédalle(Mx_inf=1, My_inf=5)
    p["sections"][0]["faces"]["inf_y"]["couches"].append(M.nouvelle_couche(False))  # un renfort principal sur la prédalle
    croise = M.verifier(p)
    return {"hors": hors, "gros": gros, "croise": croise}


# ---------------------------------------------------------------- les contrôles
def controles(prefixe=""):
    """Tous les contrôles des cas ; rend [(nom, vrai/faux)] — rejoués tels quels par chaque mutation."""
    out = []

    def c(nom, f):
        try:
            out.append((prefixe + nom, bool(f())))
        except Exception:  # noqa: BLE001
            out.append((prefixe + nom, False))
    a = cas(cas_auto)
    c("1. couture automatique : treillis 10/10/100/100 → barres Ø10/100 (ses porteurs : Øl tous les el)",
      lambda: a["t10"] == ("barres", 10, 100))
    c("1. un fil de 5 mm (treillis 5/5/150/150) → barres Ø6/150 (le premier Ø du moteur qui ne soit pas plus petit)",
      lambda: a["t5"] == ("barres", 6, 150))
    c("1. une base « n barres » (5 Ø10) est reprise telle quelle", lambda: a["tn"] == ("n_barres", 5, 10))

    d = cas(cas_defaut)
    s, v = d.get("s"), d.get("v")
    jt = (s or {}).get("joint") or {}
    cc = jt.get("couture") or {}
    As_c = aire(10) * 1000 / 100
    req = 30e6 / (FYD * 0.9 * 145)
    c("2. couture automatique Ø10/100, posée SUR la prédalle : h_p + Ø/2 = 6 + 0,5 = 6,5 cm ; axe ⌈6,5 + 1⌉ au 0,5 = 7,5 cm",
      lambda: jt["auto"] and cc["libelle"] == "Ø10/100" and proche(cc["e_reel"], 6.5) and proche(cc["e"], 7.5) and cc["pose"] == "sur_predalle")
    c("2. au joint : A_s = π·10²/4 × 1000/100 = 785,4 mm² ; y_G = 7,5 cm ; d = 22 − 7,5 = 14,5 cm",
      lambda: proche(jt["As"], As_c) and proche(jt["yG"], 7.5) and proche(jt["d"], 14.5))
    c("2. A_s,req = 30·10⁶ / (333,3 × 0,9 × 145) = 689,7 mm² ; A_s,min = 0,26·f_ctm/f_yk·b·h = 331,4 mm² ; 689,7 ≤ 785,4 : conforme (87,8 %)",
      lambda: proche(jt["As_req"], req) and proche(jt["As_min"], AS_MIN_EC) and proche(v["joint"]["valeur"], req)
      and proche(v["joint"]["limite"], As_c) and v["joint"]["etat"] == "ok" and proche(v["joint"]["taux"], req / As_c))
    c("2. la pleine prédalle garde la base DANS la peau : axe 4,5 cm, d = 17,5 cm, A_s,req = 571,4 mm²",
      lambda: proche(s["grandeurs"]["d_inf_x"], 17.5) and proche(s["grandeurs"]["As_req_inf_x"], 30e6 / (FYD * 0.9 * 175)))
    c("2. la vérification du joint suit la face inférieure secondaire (X), dans l'ordre de l'écran",
      lambda: [x["cle"] for x in s["verifications"]] == ["hauteur", "inf_y", "inf_x", "joint", "sup_y", "sup_x"])
    c("2. la hauteur : la famille du JOINT gouverne — h_min = h_u,min(30) + 7,5 = 11,64 + 7,5 = 19,14 cm ≤ 22 ; la pleine prédalle "
      "garde la sienne (11,64 + 4,5 = 16,14 cm, ses grandeurs inchangées)",
      lambda: proche(v["hauteur"]["valeur"], hu(30) + 7.5, 1e-6) and v["hauteur"]["face"] == "joint" and v["hauteur"]["etat"] == "ok"
      and proche(s["grandeurs"]["h_min"], hu(30) + 4.5, 1e-6) and proche(s["grandeurs"]["h_min_joint"], hu(30) + 7.5, 1e-6)
      and proche(s["pleine_predalle"]["hauteur"]["valeur"], hu(30) + 4.5, 1e-6))
    c("2. la longueur : l₀ = 1,5 × 10/4 × 434,8 / f_bd = 536,1 mm (C30/37, B500, α6 = 1,5) → 55 cm de chaque côté, 110 cm en tout",
      lambda: proche(jt["longueur"]["l0"], l0_main(10), 1e-6) and jt["longueur"]["cote"] == 55 and jt["longueur"]["total"] == 110)
    c("2. la fenêtre de calcul du joint commence par l'axe de la couture et finit par la longueur : l_b,rqd, l₀, puis 55 cm de chaque côté",
      lambda: v["joint"]["formules"][0]["cle"] == "e" and v["joint"]["formules"][0]["couture"]
      and [f["cle"] for f in v["joint"]["formules"][-3:]] == ["lb_rqd", "l0", "l_couture"]
      and v["joint"]["formules"][-1]["resultat"] == 55 and v["joint"]["formules"][-1]["total"] == 110)
    pz = cas(cas_pas)
    c("2. (lot 6) le pas de la note : l₀ = 536,1 mm arrondi au cm → 54 cm de chaque côté (108 cm) ; au 10 cm → 60 cm (120 cm) ; "
      "le défaut, 5 cm → 55 cm",
      lambda: pz["un"]["cote"] == 54 and pz["un"]["total"] == 108 and pz["un"]["pas"] == 1.0
      and pz["dix"]["cote"] == 60 and pz["dix"]["total"] == 120 and pz["dix"]["pas"] == 10.0
      and jt["longueur"]["pas"] == 5.0 and J.PAS_COUTURE == (1.0, 5.0, 10.0) and M.nouvelle_predalle()["pas_couture"] == 5.0)
    c("2. tout conforme : la section, la prédalle et leur calcul de pleine prédalle",
      lambda: s["statut"] == "ok" and d["r"]["statut"] == "ok" and s["pleine_predalle"]["statut"] == "ok")

    e = cas(cas_saisie)
    s2 = e.get("s") or {}
    j2 = s2.get("joint") or {}
    As2 = aire(12) * 1000 / 150 + 3 * aire(12)
    c("3. couture saisie Ø12/150 : posée à 6 + 0,6 = 6,6 cm, axe 8,0 cm ; le renfort 3 Ø12 gardé, sur la prédalle, axe 8,0 cm",
      lambda: not j2["auto"] and j2["couture"]["libelle"] == "Ø12/150" and proche(j2["couture"]["e_reel"], 6.6)
      and proche(j2["couture"]["e"], 8.0) and len(j2["geometrie"]["couches"]) == 2 and proche(j2["geometrie"]["couches"][1]["e"], 8.0))
    c("3. le y_G IMPOSÉ (4 cm) ne vaut pas au joint : y_G calculé 8,0 cm, d = 14 cm, A_s = 754,0 + 339,3 = 1 093,3 mm², "
      "A_s,req = 714,3 mm² ; la pleine prédalle garde son y_G imposé (d = 18 cm)",
      lambda: proche(j2["yG"], 8.0) and proche(j2["d"], 14.0) and proche(j2["As"], As2) and proche(j2["As_req"], 30e6 / (FYD * 0.9 * 140))
      and proche(s2["grandeurs"]["d_inf_x"], 18.0))
    c("3. l₀ Ø12 = 643,3 mm → 65 cm de chaque côté, 130 cm en tout",
      lambda: proche(j2["longueur"]["l0"], l0_main(12), 1e-6) and j2["longueur"]["cote"] == 65 and j2["longueur"]["total"] == 130)

    g = cas(cas_rouge)
    s3, r3 = g.get("s") or {}, g.get("r") or {}
    c("4. M = 38 kN·m : au joint, A_s,req = 873,6 > 785,4 mm² — non conforme ; en pleine prédalle 723,8 ≤ 785,4 — conforme",
      lambda: next(x for x in s3["verifications"] if x["cle"] == "joint")["etat"] == "nok" and s3["grandeurs"]["etat_inf_x"] == "ok"
      and proche(s3["joint"]["As_req"], 38e6 / (FYD * 0.9 * 145)))
    c("4. la section et la prédalle sont NON CONFORMES, le joint les gouverne ; le calcul de pleine prédalle reste conforme et le dit",
      lambda: s3["statut"] == "nok" and r3["statut"] == "nok" and r3["gouvernant"]["cle"] == "joint" and s3["gouvernant"] == "joint"
      and s3["pleine_predalle"]["statut"] == "ok" and r3["pleine_predalle"]["statut"] == "ok" and r3["pleine_predalle"]["gouvernant"]["cle"] != "joint")

    b = cas(cas_bords)
    av = lambda r, code: [a for a in r["avertissements"] if a["code"] == code]  # noqa: E731
    c("5. h = 10, h_p = 9 : la couture (Ø10 à 9,5 cm, axe 10,5) sort de l'épaisseur — d ≤ 0, non conforme, l'avertissement le dit",
      lambda: "d_utile_negatif" in next(x for x in b["hors"]["sections"][0]["verifications"] if x["cle"] == "joint")["motifs"]
      and len(av(b["hors"], "joint_d_utile_negatif")) == 1)
    c("5. une couture Ø40 : barre de gros diamètre (§8.8), l'avertissement des ancrages repris", lambda: len(av(b["gros"], "joint_gros_diametre")) == 1)
    jc = b["croise"]["sections"][0]["joint"]
    c("5. un renfort principal 3 Ø12 (Y) posé sur la prédalle : la couture Ø10 est posée SUR lui — h_p + Ø_Y + Ø/2 = 6 + 1,2 + 0,5 "
      "= 7,7 cm, axe ⌈8,7⌉ = 9,0 cm, d = 13 cm (le côté sûr) ; aucun croisement à dire",
      lambda: proche(jc["couture"]["e_reel"], 7.7) and proche(jc["couture"]["e"], 9.0) and proche(jc["d"], 13.0)
      and [t["symbole"] for t in jc["couture"]["detail"]["termes"]] == ["h_p", "Ø_Y", "Ø/2"] and not av(b["croise"], "joint_croisement"))

    def refus(x):
        try:
            J.normaliser_couture(x)
            return False
        except ValueError:
            return True
    c("6. la saisie refusée : Ø11, espacement 20 ou 600 ou 150,5, une clé inconnue, un Ø manquant, un texte, une liste",
      lambda: all(refus(x) for x in ({"d": 11, "esp": 150}, {"d": 10, "esp": 20}, {"d": 10, "esp": 600}, {"d": 10, "esp": 150.5},
                                     {"d": 10, "esp": 150, "x": 1}, {"esp": 150}, "Ø10/150", [10, 150], {"d": True, "esp": 150})))
    c("6. la saisie admise : null (automatique), {d, esp} (10.0 lu 10) ; normaliser(normaliser(p)) == normaliser(p)",
      lambda: J.normaliser_couture(None) is None and J.normaliser_couture({"d": 10.0, "esp": 150}) == {"d": 10, "esp": 150}
      and M.normaliser(M.normaliser(prédalle(couture={"d": 12, "esp": 100}))) == M.normaliser(prédalle(couture={"d": 12, "esp": 100})))
    c("6. (lot 6) le pas refusé : 2 cm, « 5 » en texte, vrai, 0 — le message nomme le champ et les pas permis ; absent : 5 cm",
      lambda: all("le pas d'arrondi" in _message(lambda x=x: J.lire_pas(x)) and "1, 5 ou 10 cm" in _message(lambda x=x: J.lire_pas(x))
                  for x in (2, "5", True, 0))
      and all("pas_couture : le pas d'arrondi" in _message(lambda x=x: M.normaliser(dict(prédalle(), pas_couture=x))) for x in (2, "5", True, 0))
      and M.normaliser(prédalle())["pas_couture"] == 5.0 and M.normaliser(dict(prédalle(), pas_couture=None))["pas_couture"] == 5.0)
    c("6. une couture refusée par la prédalle : le chemin du champ dans le message",
      lambda: (lambda m: "sections[0].couture.d" in m)(_message(lambda: M.normaliser(prédalle(couture={"d": 11, "esp": 150})))))
    return out


def _message(f):
    try:
        f()
        return ""
    except ValueError as e:
        return str(e)


for nom, ok in controles():
    chk(nom, ok)

# ---------------------------------------------------------------- 7. les mutations
SOURCE = open(os.path.join(RACINE, "beton", "predalle", "joint.py"), encoding="utf-8").read()
MUTATIONS = [
    ("la couture posée dans la peau (c + Ø/2) au lieu de SUR la prédalle", "2. couture automatique Ø10/100",
     '        return [("h_p", float(h_pre)), ("Ø/2", float(d) / 20.0)]', '        return [("c", 3.0), ("Ø/2", float(d) / 20.0)]'),
    ("le y_G imposé gardé au joint", "3. le y_G IMPOSÉ", 'face["yG"] = None', "pass"),
    ("la longueur arrondie au centimètre et non au pas", "2. la longueur",
     "cote = math.ceil(l0 / 10.0 / pas - 1e-9) * pas", "cote = math.ceil(l0 / 10.0 - 1e-9)"),
    ("(lot 6) le pas de la note ignoré : toujours 5 cm", "2. (lot 6) le pas de la note",
     'longueur(c0["d"], p["beton"], p["fyk"], p.get("pas_couture", PAS_LONGUEUR))', 'longueur(c0["d"], p["beton"], p["fyk"])'),
    ("(lot 6) un pas hors de la liste admis (2 cm)", "6. (lot 6) le pas refusé",
     "or float(x) not in PAS_COUTURE:", "or float(x) <= 0:"),
    ("le statut de la section sans le joint", "4. la section et la prédalle",
     's["statut"] = MD.fusion_statuts(*(x["etat"] for x in s["verifications"]))', 's["statut"] = s["pleine_predalle"]["statut"]'),
    ("la hauteur du joint ignorée", "2. la hauteur", "if M_j > 0 and h_j > ", "if False and h_j > "),
    ("le fil de 5 mm gardé tel quel (Ø5, hors des Ø du moteur)", "1. un fil de 5 mm", '"d": _diametre_au_moins(t[0])', '"d": t[0]'),
    ("ρ1 = 50 % (α6 = 1,41) au lieu de 100 %", "2. la longueur", "RHO1 = 100.0", "RHO1 = 50.0"),
    ("les renforts secondaires oubliés au joint", "3. couture saisie", 'face["couches"] = [c] + face["couches"][1:]', 'face["couches"] = [c]'),
    ("la couture posée sur la prédalle sous un renfort principal (le croisement ignoré)", "5. un renfort principal 3 Ø12",
     "    if not sur:\n", "    if True:\n"),
]
vrai = M.J
for nom, cible, ancien, nouveau in MUTATIONS:
    if SOURCE.count(ancien) != 1:
        chk(f"mutation · {nom} : la cible existe une fois dans joint.py", False, ancien)
        continue
    mutant = types.ModuleType("joint_mutant")
    mutant.__file__ = vrai.__file__
    exec(compile(SOURCE.replace(ancien, nouveau), "joint_mutant.py", "exec"), mutant.__dict__)  # noqa: S102
    M.J = mutant
    J_vrai, J = J, mutant                     # (les cas lisent aussi J : couche_couture, normaliser_couture)
    try:
        res = dict(controles("mutation : "))
    finally:
        M.J, J = vrai, J_vrai
    rouges = [n for n, ok in res.items() if not ok and n.startswith("mutation : " + cible)]
    chk(f"mutation · {nom} : le contrôle « {cible}… » rougit", bool(rouges), sorted(n for n, ok in res.items() if not ok)[:4])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
