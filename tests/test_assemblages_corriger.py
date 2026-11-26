# -*- coding: utf-8 -*-
"""« À corriger » : quand un assemblage ne passe pas, l'écran PROPOSE — et ce qu'il propose est juste (26/11/2026).

Le propriétaire, sur une capture de la platine d'about (HEA 320 sur HEA 400, V = 400 kN, M24 ; « Impossible — platine
trop étroite (pince e2) », un gros « η 131 % ») :
- « en haut à droite, il y a un gros 131 % et une autre ligne. Dans le module en béton, ça on avait supprimé, parce que
  ça ne sert à rien de l'afficher alors qu'en dessous on le voit » ;
- « ça me dit que c'est pas bon : que ça me propose une solution bonne. Les pinces, par exemple, je suis pas bon : que
  ça me propose un truc qui serait correct » ;
- « dans les options, ça propose un diamètre 16 à 22 […] que ça ne mette pas ça. Que ça mette les boulons qui
  conviennent ».

Ce que cette suite verrouille, sur les six attaches à prédimensionnement (et le pied de poteau pour le verdict) :

  1. CHAQUE retouche proposée (``acier/assemblages/corrections.corriger``) est recalculée ici, hors de la recherche : le
     défaut qu'elle vise n'est plus signalé, AUCUN nouveau défaut n'apparaît ; elle ne touche qu'UNE saisie, de la
     pièce d'attache ou des boulons — jamais un effort, un profilé, un acier, un niveau ni le grugeage ; sa valeur est
     sur la grille (5 mm, une rangée, une épaisseur de la série) ; c'est la PLUS PETITE qui lève le défaut ;
  2. le cas de la capture : les deux défauts, leurs retouches (b_p 150 → 155, p_3 90 → 85, p_1 65 → 60, h_p 190 → 200) ;
  3. l'écran : plus de gros taux en tête ; un assemblage non vérifié s'ouvre sur « À corriger » — ses défauts, leurs
     boutons (la valeur part par le chemin du champ : le bouton fait ce qu'il dit), la solution vérifiée et
     « Appliquer » ; une alerte dite là n'est pas redite dans les messages ; vérifié, l'écran est celui d'avant ;
  4. sans aucune solution : la vérification qui bloque, et « Profilés qui conviennent » — chaque profilé listé a une
     solution que le moteur vérifie, le retenir recopie le profilé ET sa solution ;
  5. plus de plage de diamètres : les Options ne la montrent plus, la recherche parcourt tout le catalogue ;
  6. des MUTATIONS : la recherche abîmée (elle accepte une retouche qui crée un autre défaut, ou qui ne lève rien ;
     elle ne resserre plus ; elle touche un niveau) et l'écran abîmé (un bouton qui n'envoie pas sa valeur, une alerte
     redite, le bloc montré sous un assemblage vérifié) font rougir les contrôles ci-dessus.

Lancement : python3 tests/test_assemblages_corriger.py
"""
import html
import importlib.util
import os
import re
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages import corrections as CO  # noqa: E402
from acier.bibliotheques import DB  # noqa: E402
from api_web import assemblages as A  # noqa: E402
from api_web import assemblages_corriger as AC  # noqa: E402
from api_web import commun as C  # noqa: E402
import api_web.assemblages_pied  # noqa: E402,F401  (enregistre le pied de poteau)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


# les DONNÉES DE BASE : ce qu'une retouche ou une solution ne doit jamais toucher (comme tests/test_assemblages_predim.py)
BASE = ("V_Ed", "N_Ed", "H_Ed", "M_Ed", "nat_N", "prof_P", "prof_S", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u", "hS_u", "bS_u",
        "twS_u", "tfS_u", "rS_u", "nu_P", "nu_S", "nu_pl", "nu_C", "fy_u", "fu_u", "bw_u", "d_top", "z_C", "d_nt", "d_nb", "l_n",
        "r_n", "lt_ok", "g_h", "appui", "rec_S", "j_lat", "classe", "cat", "trou", "filet", "fix_P", "fix_S", "eta_c", "pd_dmin",
        "pd_dmax", "k_e1", "k_p1", "k_e2", "mode_calc")
SIX = ("plat_ame", "doubles_cornieres", "platine_about", "plat_ame_poteau", "doubles_cornieres_poteau", "platine_about_poteau")
CAPTURE = ("platine_about", dict(prof_S="HEA 320", prof_P="HEA 400", V_Ed=400.0, boulon_u="M24", n1_u=3, p3_u=90.0, p1_u=65.0, e1_u=35.0))
# des géométries que le moteur REFUSE (une pince trop courte, un entraxe trop serré, une pièce trop étroite, trop de
# rangées), par attache : (modifications des saisies par défaut, la saisie que l'on s'attend à voir retouchée)
JEUX = {
    "plat_ame": [(dict(e1_u=20.0), "e1_u"), (dict(p1_u=40.0), "p1_u"), (dict(e2b_u=20.0), "e2b_u"), (dict(n1_u=6), "n1_u"),
                 (dict(boulon_u="M27"), None)],
    "doubles_cornieres": [(dict(p1S_u=70.0), "p1S_u"), (dict(e1P_u=15.0, gA_u=30.0), "gA_u"), (dict(e1S_u=15.0), "e1S_u"),
                          (dict(n1P_u=5), "n1P_u")],
    "platine_about": [(dict(boulon_u="M24"), "bp_u"), (dict(p1_u=40.0), "p1_u"), CAPTURE[1:] + ("bp_u",), (dict(e1_u=15.0), "e1_u")],
    "plat_ame_poteau": [(dict(e1_u=20.0), "e1_u"), (dict(p1_u=45.0), "p1_u"), (dict(e2b_u=15.0), "e2b_u")],
    "doubles_cornieres_poteau": [(dict(LC_u=200.0), "LC_u"), (dict(p1S_u=45.0), "p1S_u"), (dict(e1P_u=15.0), "e1P_u")],
    "platine_about_poteau": [(dict(p3_u=60.0), "p3_u"), (dict(p1_u=45.0), "p1_u"), (dict(e1_u=15.0), "e1_u")],
}


def etat_de(cle, **modifs):
    etat = A.nouvel_etat()
    etat["assemblages"] = [A.nouvel_assemblage(cle)]
    u = etat["assemblages"][0]["values"]
    inconnues = [k for k in modifs if k not in u]
    assert not inconnues, inconnues
    u.update(modifs)
    etat["assemblages"][0]["values"] = A.valeurs_normales(A.CONFIGS[cle], u)
    return etat


def chercher(cle, modifs, module=CO):
    """(cfg, saisies passées au moteur, R, les corrections de ``module``) du jeu ``modifs``."""
    etat = etat_de(cle, **modifs)
    el = A._actif(etat)
    cfg = A._cfg(el)
    u = el["values"]
    base = A.a_calculer(cfg, u)
    R = A.moteur(cfg, u)
    permises = {k for k in AC.RETOUCHABLES if k in cfg.E.CHAMPS and cfg.E.champ_visible(u, k)}
    X = module.corriger(lambda v: A.moteur(cfg, v), base, R, permises, A.COMPTES | set(cfg.entiers), AC.GRILLES)
    return cfg, base, R, X


def grille_ok(cfg, k, x):
    if k in A.COMPTES or k in cfg.entiers:
        return abs(x - round(x)) < 1e-9 and x >= 1
    g = AC.GRILLES.get(k, CO.PAS_MM)
    if isinstance(g, (list, tuple)):
        return any(abs(x - y) < 1e-9 for y in g)
    return abs(x / g - round(x / g)) < 1e-9 and x > 0


def fautes_retouches(cle, modifs, module=CO):
    """Les manquements des retouches du jeu ``modifs`` : ([texte], X, R)."""
    cfg, base, R, X = chercher(cle, modifs, module)
    f = []
    b0, k0 = CO.bloquantes(R), CO.pinces_ko(R)
    for x in X:
        d = x.defaut
        for r in x.retouches:
            nom = f"{d.id} · {r.cle} {r.de:g} → {r.vers:g}"
            if r.cle not in AC.RETOUCHABLES or r.cle in BASE or r.cle not in d.fields:
                f.append(f"{nom} : touche une saisie qui n'est pas de la pièce d'attache ni des boulons, ou que le défaut ne nomme pas")
            if abs(r.de - float(base[r.cle])) > 1e-9 or r.vers == r.de or not grille_ok(cfg, r.cle, r.vers):
                f.append(f"{nom} : valeur de départ ou grille")
            # le calcul refait ICI, hors de la recherche, avec cette seule saisie changée
            v = dict(base)
            v[r.cle] = int(r.vers) if (r.cle in A.COMPTES or r.cle in cfg.entiers) else r.vers
            R1 = A.moteur(cfg, v)
            b1, k1 = CO.bloquantes(R1), CO.pinces_ko(R1)
            if d.alerte is not None and d.alerte.id in b1:
                f.append(f"{nom} : l'alerte « {d.alerte.id} » est toujours là")
            if d.pince is not None and d.pince.id in k1:
                f.append(f"{nom} : la pince « {d.pince.id} » reste non conforme")
            if not (b1 <= b0 and k1 <= k0):
                f.append(f"{nom} : un NOUVEAU défaut apparaît ({sorted(b1 - b0)} {sorted(k1 - k0)})")
            if bool(R1.verified) != r.verifie:
                f.append(f"{nom} : « assemblage vérifié » mal dit ({r.verifie} / {R1.verified})")
            # la plus petite : un cran de moins ne lève pas le défaut
            if r.crans > 1:
                sens = 1 if r.vers > r.de else -1
                pas = AC.GRILLES.get(r.cle, 1.0 if (r.cle in A.COMPTES or r.cle in cfg.entiers) else CO.PAS_MM)
                avant = CO.cran(r.de, sens, r.crans - 1, pas)
                w = dict(base)
                w[r.cle] = int(avant) if (r.cle in A.COMPTES or r.cle in cfg.entiers) else avant
                try:
                    R2 = A.moteur(cfg, w)
                except Exception:  # noqa: BLE001
                    R2 = None
                if CO.leve(d, R, R2):
                    f.append(f"{nom} : {avant:g} (un cran de moins) levait déjà le défaut")
        vus = [r.cle for r in x.retouches]
        if len(vus) != len(set(vus)):
            f.append(f"{d.id} : deux retouches de la même saisie")
    if X and X[0].essais > CO.BUDGET:
        f.append(f"{X[0].essais} calculs (budget {CO.BUDGET})")
    return f, X, R


print("=== 1. Chaque retouche lève son défaut, sans en créer d'autre — les six attaches ===")
total, essais_maxi, duree_maxi = 0, 0, 0.0
for cle in SIX:
    for modifs, attendue in JEUX[cle]:
        t0 = time.time()
        f, X, R = fautes_retouches(cle, modifs)
        duree_maxi = max(duree_maxi, time.time() - t0)
        n = sum(len(x.retouches) for x in X)
        total += n
        essais_maxi = max(essais_maxi, X[0].essais if X else 0)
        vues = {r.cle for x in X for r in x.retouches}
        chk(f"{cle} {modifs} : non vérifié, {len(X)} défaut(s), {n} retouche(s) — chacune recalculée lève son défaut sans en "
            "créer d'autre, ne touche qu'une saisie de l'attache ou des boulons, sur la grille, la plus petite"
            + (f" ; {attendue} est retouchée" if attendue else ""),
            not f and not R.verified and len(X) >= 1 and (attendue is None or attendue in vues), f"{f[:3]} {sorted(vues)}")
chk(f"au moins 30 retouches vérifiées sur les six attaches ({total}) ; la recherche la plus longue : {essais_maxi} calculs "
    f"complets (budget {CO.BUDGET}), {duree_maxi * 1000:.0f} ms", total >= 30 and essais_maxi <= CO.BUDGET and duree_maxi < 2,
    f"{total} {essais_maxi} {duree_maxi:.2f}")
for cle in SIX:
    cfg, base, R, X = chercher(cle, {})
    chk(f"{cle}, les données par défaut : assemblage vérifié — aucun défaut, aucune retouche, aucun calcul de plus",
        R.verified and X == [] and CO.defauts(R) == [])

print("\n=== 2. Le cas de la capture du propriétaire : platine d'about, HEA 320 sur HEA 400, V = 400 kN, M24 ===")
f, X, R = fautes_retouches(*CAPTURE)
par = {x.defaut.id: {r.cle: (r.de, r.vers) for r in x.retouches} for x in X}
chk("deux défauts de géométrie — la pince basse du groupe (e1bot) et la platine trop étroite (e2p) ; leurs retouches : "
    "b_p 150 → 155 ou p_3 90 → 85 ; p_1 65 → 60, h_p 190 → 200 ou une rangée de moins",
    not f and list(par) == ["e1bot", "e2p"] and par["e2p"] == {"bp_u": (150.0, 155.0), "p3_u": (90.0, 85.0)}
    and par["e1bot"] == {"p1_u": (65.0, 60.0), "hp_u": (190.0, 200.0), "n1_u": (3.0, 2.0)}, str(par))
chk("une cote avant un nombre de rangées, le moins de crans d'abord (p_1, puis h_p, puis n_1)",
    [r.cle for r in X[0].retouches] == ["p1_u", "hp_u", "n1_u"], str([r.cle for r in X[0].retouches]))
d = CO.defauts(R)
chk("un défaut = une alerte bloquante ET la pince non conforme qu'elle signale (une fois, pas deux)",
    [x.id for x in d] == ["e1bot", "e2p"] and all(x.pince is not None and x.pince.alert == x.id and not x.pince.ok for x in d)
    and len(CO.pinces_ko(R)) == 2, str([(x.id, x.pince and x.pince.id) for x in d]))

print("\n=== 3. L'écran : plus de gros taux, « À corriger » en tête, les boutons font ce qu'ils disent ===")


def texte(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


RX_RET = re.compile(r'<button type="button" class="asm-ret[^"]*" data-agir data-chemin="(v/[^"]+)" data-valeur="([^"]*)"')
RX_SOL = re.compile(r'<button type="button" class="btn sm primary" data-agir data-chemin="solution" data-valeur="([^"]*)"')
for cle in list(SIX) + ["platine_beton"]:
    r = A.resultat(etat_de(cle))
    res = r["resultat"]
    chk(f"{cle}, vérifié : ni gros taux en tête (« chip xl », « verdict »), ni bloc « À corriger » ; le bloc « Vérifications » "
        "dit l'état et le taux" + (" ; le bloc « Prédimensionnement » suit" if cle in SIX else ""),
        "chip xl" not in res and 'class="verdict"' not in res and "asm-corr" not in res and AC.TITRE not in texte(res)
        and re.search(r'<span class="n">Vérifications</span><span class="e">[^<]*\d+\xa0%</span>', res) is not None
        and (('<span class="n">Prédimensionnement</span>' in res) == (cle in SIX)), texte(res)[:200])


def fautes_ecran(cle, modifs, module=AC):
    """Les manquements de l'écran d'un assemblage NON vérifié : ([texte], résultat rendu)."""
    etat = etat_de(cle, **modifs)
    el = A._actif(etat)
    cfg = A._cfg(el)
    u = el["values"]
    R = A.moteur(cfg, u)
    r = A.resultat(etat)
    res = r["resultat"]
    if module is not AC:                          # le bloc muté, à la place de celui de l'écran
        bloc, dites = module.bloc(A.pour(cfg, R), R, u)
        res = res.replace(AC.bloc(A.pour(cfg, R), R, u)[0], bloc) if AC.bloc(A.pour(cfg, R), R, u)[0] else res.replace('<div class="et-blocs">', '<div class="et-blocs">' + bloc)
    f = []
    if R.verified:
        if "asm-corr" in res:
            f.append("le bloc « À corriger » sous un assemblage vérifié")
        return f, res
    if "chip xl" in res or 'class="verdict"' in res:
        f.append("un gros taux en tête")
    if not ("asm-corr" in res and res.index("asm-corr") < res.index(">Vérifications<")):
        f.append("le bloc « À corriger » n'ouvre pas le résultat")
        return f, res
    if '<span class="n">Prédimensionnement</span>' in res:
        f.append("le bloc « Prédimensionnement » redit ce que « À corriger » propose déjà")
    defauts = CO.defauts(R)
    if res.count('class="asm-def"') != len(defauts):
        f.append(f"{res.count('class=\"asm-def\"')} défauts écrits, {len(defauts)} nommés par le moteur")
    c = A.pour(cfg, R)
    for d in defauts:
        titre = AC.dire_defaut(c, R, d)[0]
        if res.count(C.esc(titre)) != 1:
            f.append(f"« {titre} » est écrit {res.count(C.esc(titre))} fois (une alerte dite dans le bloc n'est pas redite)")
    # chaque bouton : la valeur part par le chemin du champ, et le défaut de SON paragraphe tombe
    blocs = re.findall(r'<div class="asm-def" data-defaut="([^"]+)">(.*?)</div></div>|<div class="asm-def" data-defaut="([^"]+)"><p>.*?</p></div>', res, re.S)
    n_boutons = 0
    for ident, corps, _seul in blocs:
        for chemin, valeur in RX_RET.findall(corps or ""):
            n_boutons += 1
            e2 = etat_de(cle, **modifs)
            msg = A.modifier(e2, chemin.split("/"), valeur)
            R2 = A.moteur(A._cfg(A._actif(e2)), A._actif(e2)["values"])
            apres = {x.id for x in CO.defauts(R2)}
            if msg is not None or ident in apres or not apres <= {x.id for x in defauts}:
                f.append(f"le bouton {chemin} = {valeur} du défaut « {ident} » : {msg or sorted(apres)}")
            autres = [k for k in A._actif(e2)["values"] if A._actif(e2)["values"][k] != u[k]]
            if autres != [chemin[2:]]:
                f.append(f"le bouton {chemin} change {autres}")
    # la solution : « Appliquer » recopie une proposition que le moteur vérifie, sans toucher aux données de base
    for nom in RX_SOL.findall(res):
        e2 = etat_de(cle, **modifs)
        msg = A.modifier(e2, ["solution"], html.unescape(nom))
        v = A._actif(e2)["values"]
        R2 = A.moteur(A._cfg(A._actif(e2)), v)
        if msg is not None or not R2.verified:
            f.append(f"« Appliquer » {nom} : {msg or 'assemblage non vérifié'}")
        touchees = [k for k in BASE if k in u and v.get(k) != u.get(k)]
        if touchees:
            f.append(f"« Appliquer » {nom} change {touchees}")
    return f, res


total_boutons = 0
for cle in SIX:
    for modifs, _attendue in JEUX[cle][:3]:
        f, res = fautes_ecran(cle, modifs)
        nb, ns = len(RX_RET.findall(res)), len(RX_SOL.findall(res))
        total_boutons += nb
        chk(f"{cle} {modifs} : le résultat s'ouvre sur « À corriger » — chaque défaut une fois, {nb} retouche(s) dont le bouton "
            f"lève SON défaut et ne change que sa saisie, {ns} solution(s) qu'« Appliquer » rend vérifiée(s) sans toucher aux "
            "données de base ; plus de gros taux", not f and nb + ns >= 1, str(f[:3]))
chk(f"au moins 25 boutons de retouche essayés ({total_boutons})", total_boutons >= 25, str(total_boutons))
f, res = fautes_ecran("doubles_cornieres", dict(prof_S="HEA 300", prof_P="HEA 400", V_Ed=200.0))
t = texte(res)
chk("doubles cornières, V = 200 kN (résistance insuffisante, aucune géométrie en cause) : pas de défaut, la solution "
    "CONSEILLÉE d'abord (elle tient le taux cible), puis celle « avec vos M20 » (au-dessus du taux cible), « Autres solutions »",
    not f and "résistance insuffisante" in t and res.count('class="asm-def"') == 0 and len(RX_SOL.findall(res)) == 2
    and t.index("conseillée") < t.index("avec vos M20") and 'data-pk="solution"' in res and AC.AUTRES in t, t[:400])
f, res = fautes_ecran("platine_about", dict(boulon_u="M24"))
t = texte(res)
chk("platine d'about, M24 (platine trop étroite) : la retouche d'abord, puis la solution « avec vos M24 » — elle tient le taux "
    "cible sans réserve : avant la conseillée", not f and t.index("avec vos M24") < t.index("conseillée")
    and t.index("150 → 155") < t.index("avec vos M24"), t[:500])
etat = etat_de("platine_about", boulon_u="M24")
msg = A.modifier(etat, ["solution"], "une solution qui n'existe pas")
chk("« Appliquer » d'un nom inconnu (une page restée ouverte) : refusé avec sa raison, l'assemblage intact",
    bool(msg) and "refusée" in msg and A._actif(etat)["values"] == A._actif(etat_de("platine_about", boulon_u="M24"))["values"], str(msg))
e_pied = etat_de("platine_beton")
chk("pied de poteau (pas de prédimensionnement) : « Appliquer » y est refusé ; pas de bloc « À corriger »",
    bool(A.modifier(e_pied, ["solution"], "x")) and AC.bloc(A.CONFIGS["platine_beton"], A.moteur(A.CONFIGS["platine_beton"], A._actif(e_pied)["values"]), {})[0] == "")

print("\n=== 4. Sans aucune solution : la vérification qui bloque, les profilés qui conviennent ===")
etat = etat_de(*CAPTURE[:1], **CAPTURE[1])
f, res = fautes_ecran(*CAPTURE)
t = texte(res)
chk("la capture : « Aucune solution avec ces profilés » EN TÊTE du bloc (avant les défauts : retoucher une pince n'y changera "
    "rien), la vérification qui bloque nommée avec son taux, « Profilés qui conviennent » ; aucun bouton « Appliquer »",
    not f and "Aucune solution avec ces profilés" in t and t.index("Aucune solution") < t.index("150 → 155")
    and "Poutre secondaire · grugeage – flexion + V reste à 130" in t.replace("\xa0", " ") and 'data-pk="profil_solution"' in res
    and not RX_SOL.findall(res) and "Toutes les dispositions ont été essayées" not in t, t[:500])
sel = A.selecteur(etat, "profil_solution")
lignes = re.findall(r'data-choix="([^"]+)"', sel)
ok_profils = []
for nom in lignes:
    e2 = etat_de(*CAPTURE[:1], **CAPTURE[1])
    u0 = dict(A._actif(e2)["values"])
    A.choisir_section(e2, "profil_solution", nom)
    v = A._actif(e2)["values"]
    R2 = A.moteur(A._cfg(A._actif(e2)), v)
    touchees = [k for k in BASE if k in u0 and v.get(k) != u0.get(k)]
    ok_profils.append((nom, bool(R2.verified), touchees))
chk(f"« Profilés qui conviennent » ({', '.join(lignes)}) : SEULS des profilés qui conviennent sont listés, le plus petit "
    "d'abord (★) ; en retenir un recopie le profilé ET sa solution — l'assemblage est vérifié, seule la poutre secondaire a "
    "changé parmi les données de base ; ceux essayés sans solution sont comptés, avec la raison du premier",
    lignes[:1] == ["HEA 340"] and all(ok and touchees == ["prof_S"] for _n, ok, touchees in ok_profils)
    and "999" not in sel and "sans solution" in texte(sel) and "Grugeage inférieur" in texte(sel), f"{ok_profils} {texte(sel)[:300]}")
for fam, mini in (("HEB", 2), ("HEM", 2)):
    sel_f = A.selecteur(etat, "profil_solution", [fam])
    noms = re.findall(r'data-choix="([^"]+)"', sel_f)
    chk(f"… une autre famille ({fam}) : {', '.join(noms)} — au moins {mini}, tous de la famille",
        len(noms) >= mini and all(n.startswith(fam + " ") for n in noms), str(noms))
try:
    A.choisir_section(etat_de(*CAPTURE[:1], **CAPTURE[1]), "profil_solution", "HEA 99999")
    refus = False
except ValueError:
    refus = True
chk("un profilé inconnu est refusé", refus)
e_haut = etat_de("plat_ame", V_Ed=5000.0)
t = texte(A.resultat(e_haut)["resultat"])
chk("plat d'âme, V = 5 000 kN : aucune solution — le bloc le dit, nomme ce qui bloque, propose de changer de profilé",
    "Aucune solution" in t and "Changer le profilé" in t, t[:300])

print("\n=== 5. Plus de plage de diamètres : tout le catalogue des boulons ===")
for cle in SIX:
    etat = etat_de(cle)
    don = A.resultat(etat)["donnees"]
    chk(f"{cle} : les Options ne portent plus « Ø min » ni « Ø max » ; le taux cible et les pinces k·d0 y restent",
        'data-chemin="v/pd_dmin"' not in don and 'data-chemin="v/pd_dmax"' not in don
        and all(f'data-chemin="v/{k}"' in don for k in ("eta_c", "k_e1", "k_p1", "k_e2")))
noms_boulons = [b["n"] for b in DB["boulons"]]
for cle in ("plat_ame", "platine_about_poteau"):
    etat = etat_de(cle, pd_dmin="M20", pd_dmax="M20")
    u = A._actif(etat)["values"]
    X = A.propositions(A._cfg(A._actif(etat)), u)
    diam = [s.boulon for s in X.solutions]
    avant = (u["pd_dmin"], u["pd_dmax"])
    A.choisir_section(etat, "solution", A._cfg(A._actif(etat)).predim["nom"](X.solutions[0].u))
    chk(f"{cle}, une note qui garde une plage « M20 – M20 » : la recherche l'ignore — {len(diam)} solutions, de "
        f"{diam and min(diam, key=noms_boulons.index)} à {diam and max(diam, key=noms_boulons.index)}, une par diamètre ; la "
        "recopier ne touche pas la plage gardée dans la note",
        len(diam) >= 5 and len(set(diam)) == len(diam) and set(diam) - {"M20"}
        and (u["pd_dmin"], u["pd_dmax"]) == avant, f"{diam} {avant} {(u['pd_dmin'], u['pd_dmax'])}")
chk("le sélecteur des solutions ne parle plus de « plage »", "plage" not in A.selecteur(etat_de("plat_ame", pd_dmin="M20", pd_dmax="M20"), "solution"))
# … mais la solution CONSEILLÉE reste un diamètre courant (le harnais navigateur a trouvé, 26/11/2026 : sans la plage, « le
# moins de boulons » conseillait « 2 M36 » pour 30 kN sur un poteau — deux rangées de M36 suffisent à la hauteur
# recommandée du plat, là où il en faut trois de M20)
from acier.assemblages import propositions as PR  # noqa: E402
COURANTS = noms_boulons[noms_boulons.index(A.CONFIGS["plat_ame"].E.defaults()["pd_dmin"]):noms_boulons.index(A.CONFIGS["plat_ame"].E.defaults()["pd_dmax"]) + 1]


def fautes_conseillee(module=PR):
    """La solution conseillée de quelques jeux, avec la recherche ``module`` : [texte] quand ce n'est pas un diamètre courant
    alors qu'un diamètre courant tient lui aussi le taux cible sans réserve."""
    f = []
    for cle, modifs in (("plat_ame_poteau", dict(V_Ed=30.0)), ("plat_ame", dict(V_Ed=30.0)), ("platine_about_poteau", {}),
                        ("doubles_cornieres", dict(V_Ed=200.0)), ("plat_ame", {})):
        etat = etat_de(cle, **modifs)
        cfg = A._cfg(A._actif(etat))
        u = A._actif(etat)["values"]
        defauts = cfg.E.defaults()
        base = dict(A.a_calculer(cfg, u), **{k: defauts[k] for k in A.PLAGE_DIAMETRES})
        X = module.chercher(cfg.predim["moteur"], cfg.S, base, cfg.predim["genre"], nb_max=len(noms_boulons), tout=True)
        s0 = X.solutions[0]
        mieux = [x for x in X.solutions if x.boulon in COURANTS and x.cible and not x.reserves]
        if s0.boulon not in COURANTS and mieux and s0.cible:
            f.append(f"{cle} {modifs} : « {cfg.predim['nom'](s0.u)} » conseillée, alors que « {cfg.predim['nom'](mieux[0].u)} » convient")
        if len({x.boulon for x in X.solutions}) != len(X.solutions):
            f.append(f"{cle} : deux solutions du même diamètre")
    return f


X30 = A.propositions(A.CONFIGS["plat_ame_poteau"], A._actif(etat_de("plat_ame_poteau", V_Ed=30.0))["values"])
noms30 = [A.CONFIGS["plat_ame_poteau"].predim["nom"](x.u) for x in X30.solutions]
chk(f"plat d'âme sur poteau, V = 30 kN : la conseillée est « 3 M20 · plat 240×90×8 » (un diamètre courant, de {COURANTS[0]} à "
    f"{COURANTS[-1]}), pas « 2 M36 » — qui reste proposée, plus bas ; {len(noms30)} diamètres conviennent",
    not fautes_conseillee() and noms30[0] == "3 M20 · plat 240×90×8" and any(n.startswith("2 M36") for n in noms30[1:])
    and len(noms30) >= 7, str(noms30))

print("\n=== 6. Mutations : la recherche et l'écran abîmés font rougir les contrôles ===")
SRC_CO = open(os.path.join(RACINE, "acier", "assemblages", "corrections.py"), encoding="utf-8").read()
SRC_AC = open(os.path.join(RACINE, "api_web", "assemblages_corriger.py"), encoding="utf-8").read()


def mutant(source, ancien, nouveau, nom, paquet=None):
    assert source.count(ancien) == 1, (ancien, source.count(ancien))
    spec = importlib.util.spec_from_loader(nom, loader=None)
    m = importlib.util.module_from_spec(spec)
    if paquet:
        m.__package__ = paquet
    exec(compile(source.replace(ancien, nouveau), nom + ".py", "exec"), m.__dict__)
    return m


def fautes_tous(m):
    return [x for cle in SIX for modifs, _a in JEUX[cle] for x in fautes_retouches(cle, modifs, m)[0]]


MUT_CO = [
    ("une retouche qui crée un AUTRE défaut est acceptée",
     "    return b1 <= b0 and k1 <= k0\n", "    return True\n", fautes_tous),
    ("une retouche qui ne lève pas l'alerte est acceptée",
     "    if d.alerte is not None and d.alerte.id in b1:\n        return False\n    if d.pince is not None and d.pince.id in k1:\n        return False\n",
     "", fautes_tous),
    ("la retouche n'est plus resserrée (le premier cran essayé qui passe, pas le plus petit)",
     "                for k in range(avant + 1, bon[0]):", "                for k in range(0):", fautes_tous),
    ("une retouche touche une saisie non permise (un niveau, le grugeage)",
     "            if cle not in permises or isinstance(u.get(cle), bool):", "            if isinstance(u.get(cle), bool):", fautes_tous),
    ("« assemblage vérifié » dit à tort",
     "verifie=bool(R1.verified)))", "verifie=True))", fautes_tous),
    ("la valeur proposée sort de la grille (la valeur brute, pas le cran)",
     "    v = base + sens * k * pas\n", "    v = v0 + sens * k * pas * 0.5\n", fautes_tous),
]
chk("témoin : la recherche telle quelle passe ces contrôles", not fautes_tous(CO))
survivants = [titre for titre, ancien, nouveau, controle in MUT_CO
              if not controle(mutant(SRC_CO, ancien, nouveau, "corrections_mutant"))]
chk(f"{len(MUT_CO)} mutations de la recherche des corrections : toutes vues rouges", not survivants, str(survivants))


def fautes_ecrans(m):
    out = []
    for cle, modifs in (("platine_about", dict(boulon_u="M24")), CAPTURE, ("plat_ame", dict(e1_u=20.0)), ("plat_ame", {})):
        out += fautes_ecran(cle, modifs, m)[0]
    return out


MUT_AC = [
    ("un bouton de retouche n'envoie pas SA valeur (la valeur de départ)",
     'data-valeur="{C.esc(vers)}" title=', 'data-valeur="{C.esc(de)}" title=', fautes_ecrans),
    ("un bouton de retouche vise une autre saisie",
     'data-agir data-chemin="v/{C.esc(r.cle)}" \'', 'data-agir data-chemin="v/z_C" \'', fautes_ecrans),
    ("le bloc « À corriger » est montré sous un assemblage vérifié",
     "    if R is None or R.verified or not cfg.predim:\n        return \"\", frozenset()", "    if R is None or not cfg.predim:\n        return \"\", frozenset()", fautes_ecrans),
    ("un défaut n'est plus écrit (les retouches sans leur raison)",
     '    defauts = "".join(_defaut(cfg, R, x) for x in cors)', '    defauts = "".join(_defaut(cfg, R, x) for x in cors[:1])', fautes_ecrans),
    ("« Appliquer » envoie un autre nom que celui de la solution",
     'data-agir data-chemin="solution" data-valeur="{C.esc(nom)}"', 'data-agir data-chemin="solution" data-valeur="{C.esc(nom)} bis"', fautes_ecrans),
]
SRC_PR = open(os.path.join(RACINE, "acier", "assemblages", "propositions.py"), encoding="utf-8").read()
chk("mutation « la conseillée est celle qui a le moins de boulons, quel que soit le diamètre » (2 M36 pour 30 kN) : vue rouge",
    bool(fautes_conseillee(mutant(SRC_PR, "key=lambda x: (not x.cible, not x.plage, len(x.reserves),", "key=lambda x: (not x.cible, len(x.reserves),",
                                  "propositions_mutant"))))
chk("témoin : l'écran tel quel passe ces contrôles", not fautes_ecrans(AC))
survivants = [titre for titre, ancien, nouveau, controle in MUT_AC
              if not controle(mutant(SRC_AC, ancien, nouveau, "api_web.assemblages_corriger_mutant", "api_web"))]
chk(f"{len(MUT_AC)} mutations du bloc « À corriger » : toutes vues rouges", not survivants, str(survivants))
# … et dans assemblages.py : une alerte dite dans le bloc est redite dans les messages
m0 = A._messages
A._messages = lambda cfg, R, dites=frozenset(): m0(cfg, R)
try:
    redite = bool(fautes_ecran("platine_about", dict(boulon_u="M24"))[0])
finally:
    A._messages = m0
chk("mutation « les messages redisent les alertes du bloc » : vue rouge", redite)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:500])
sys.exit(1 if KO else 0)
