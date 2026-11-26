# -*- coding: utf-8 -*-
"""Les PROPOSITIONS du prédimensionnement : vérifiées par le calcul complet, et elles gardent les données de base — les
six attaches (25/11/2026).

Le propriétaire, deux fois :
- « quand j'ai un prédimensionnement ou une optimisation, il faut que ça me garde mes données de base : les efforts, le
  profilé, le niveau auquel ils sont. Après, c'est la plaque, le nombre de boulons qui doit changer » ;
- « les propositions, ça doit nous proposer des propositions qui sont correctes. La seule raison pour laquelle ça
  n'irait pas, ce serait que les profilés ne seraient pas suffisants. Et qu'on devrait changer de profilé si on a 0
  solution. […] Ça pourrait épaissir ma platine, ça pourrait rajouter des boulons. »

Ce que cette suite verrouille, pour chaque attache et sur plusieurs jeux de données (défaut, autres profilés, autres
efforts, niveaux décalés, grugeage, appui sur l'âme, efforts forts) :

  1. CHAQUE proposition (``acier/assemblages/propositions.chercher``) est un assemblage que le MOTEUR déclare vérifié :
     ses saisies sont recalculées ici, hors de la recherche, et son taux est celui du calcul complet ;
  2. une proposition ne change ni un effort, ni un profilé, ni un acier, ni un niveau, ni le grugeage, ni l'appui, ni
     un choix de l'ingénieur (classe, catégorie, trou, fixation) — par la recherche comme par l'écran du site
     (``api_web.assemblages.choisir_solution``) ;
  3. une cornière proposée est une cornière du CATALOGUE (jamais une épaisseur qui n'existe pas) ;
  4. la plus simple d'abord : une file tant qu'une file suffit ; la deuxième file quand la hauteur manque ; des
     cornières dont une aile est soudée (leur nom ne parle de boulons que s'il y en a, leur gorge est dite) ; un plat
     d'âme plus court que la hauteur recommandée (l'avis « hp_rec » du moteur) est allongé juste ce qu'il faut — par
     l'entraxe, sous le maximum que le moteur nomme, sinon par une rangée de plus ;
  5. le cas de la capture du propriétaire (doubles cornières, HEA 300 sur HEA 400, V = 200 kN) : l'estimation du moteur
     ne trouvait RIEN ; la recherche propose des dispositions vérifiées ;
  6. sans solution, la recherche nomme la vérification qui bloque et l'écran dit de changer de profilé ;
  7. l'écran : le bloc « Prédimensionnement » du résultat ouvre le sélecteur ; plus de mode prédimensionnement (une note
     enregistrée dans ce mode est relue avec sa proposition recopiée ; la bascule est refusée) ;
  8. des MUTATIONS : la recherche abîmée (elle accepte un candidat non vérifié, touche un niveau, n'essaie plus la
     deuxième file, propose deux files d'abord, ne lève plus la réserve de hauteur ou allonge le plat plus que
     nécessaire, propose plusieurs fois les mêmes cornières soudées) fait rougir les contrôles ci-dessus.

Lancement : python3 tests/test_assemblages_predim.py
"""
import importlib.util
import os
import re
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages import propositions as PR  # noqa: E402
from acier.bibliotheques import DB, PERSO  # noqa: E402
from api_web import assemblages as A  # noqa: E402
import api_web.assemblages_poutre as PP  # noqa: E402,F401  (enregistre les configurations)
import api_web.assemblages_poteau as PT  # noqa: E402,F401

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


# les DONNÉES DE BASE : ce qu'une proposition ne doit jamais toucher (les clés absentes d'une attache sont ignorées)
BASE = ("V_Ed", "N_Ed", "H_Ed", "M_Ed", "nat_N",                                          # efforts
        "prof_P", "prof_S", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u",   # profilés
        "nu_P", "nu_S", "nu_pl", "nu_C", "fy_u", "fu_u", "bw_u",                          # aciers
        "d_top", "z_C",                                                                   # niveaux
        "d_nt", "d_nb", "l_n", "r_n", "lt_ok", "g_h",                                     # grugeage, jeu
        "appui", "rec_S", "j_lat",                                                        # appui sur le poteau
        "classe", "cat", "trou", "filet", "fix_P", "fix_S",                               # choix de l'ingénieur
        "eta_c", "pd_dmin", "pd_dmax", "k_e1", "k_p1", "k_e2",                            # règles du prédimensionnement
        "mode_calc")
CATALOGUE = {c["n"] for c in DB["cornieres"]}

# (les géométries sont VALIDES : une semelle qui heurte la poutre porteuse ne se règle pas par les boulons — ce cas-là
#  a son contrôle, plus bas)
JEUX = {
    "plat_ame": [dict(), dict(prof_P="IPE 500", prof_S="IPE 300", V_Ed=120, N_Ed=20, d_top=0, d_nt=40, l_n=110, z_C=50),
                 dict(prof_P="HEB 300", prof_S="HEA 240", nu_S="S235", V_Ed=70, M_Ed=2, d_nt=50, l_n=140, z_C=55),
                 dict(prof_P="IPE 500", prof_S="IPE 300", V_Ed=180, N_Ed=20, d_top=0, d_nt=40, l_n=110, z_C=50)],
    "doubles_cornieres": [dict(), dict(prof_P="IPE 500", prof_S="IPE 300", V_Ed=160, d_top=0, d_nt=40, l_n=110, z_C=50),
                          dict(prof_P="HEB 300", prof_S="HEA 240", nu_S="S235", V_Ed=80, N_Ed=30, H_Ed=5, d_nt=50, l_n=140, z_C=55),
                          dict(V_Ed=200)],
    "platine_about": [dict(), dict(prof_P="IPE 600", prof_S="IPE 400", V_Ed=260, d_top=0, d_nt=45, l_n=110, z_C=50),
                      dict(prof_P="HEA 800", prof_S="IPE 600", nu_pl="S235", V_Ed=350, N_Ed=40, d_nt=60, l_n=160, z_C=65)],
    "plat_ame_poteau": [dict(), dict(prof_P="HEB 240", prof_S="IPE 300", V_Ed=110, N_Ed=60, z_C=45),
                        dict(appui="Âme du poteau", prof_P="HEA 300", prof_S="IPE 330", nu_S="S235", V_Ed=95, rec_S=0),
                        dict(V_Ed=400)],
    "doubles_cornieres_poteau": [dict(), dict(prof_P="HEB 240", prof_S="IPE 300", V_Ed=110, N_Ed=40, z_C=45),
                                 dict(appui="Âme du poteau", prof_P="HEB 340", prof_S="IPE 330", V_Ed=95, M_Ed=2),
                                 dict(V_Ed=400)],
    "platine_about_poteau": [dict(), dict(prof_P="HEB 240", prof_S="IPE 300", V_Ed=110, N_Ed=50, z_C=45),
                             dict(appui="Âme du poteau", prof_P="HEB 340", prof_S="IPE 330", nu_pl="S235", V_Ed=95),
                             dict(V_Ed=500)],
}


def saisies(cle, modifs):
    cfg = A.CONFIGS[cle]
    u = A.nouvel_assemblage(cle)["values"]
    inconnues = [k for k in modifs if k not in u]
    assert not inconnues, inconnues
    u.update(modifs)
    return cfg, A.valeurs_normales(cfg, u)


def fautes_propositions(cle, modifs, module=PR, au_moins=1):
    """Les manquements des propositions du jeu ``modifs`` : [texte]. ``module`` : la recherche (ou sa version mutée)."""
    cfg, u = saisies(cle, modifs)
    X = module.chercher(cfg.predim["moteur"], cfg.S, A.a_calculer(cfg, u), cfg.predim["genre"])
    f = []
    if len(X.solutions) < au_moins:
        f.append(f"{len(X.solutions)} proposition(s) pour {au_moins} attendue(s)")
    for s in X.solutions:
        nom = cfg.predim["nom"](s.u)
        v = A.valeurs_normales(cfg, s.u)
        R = A.moteur(cfg, v)                                  # 1. recalculée HORS de la recherche
        if not R.verified or R.get("a_verifier"):
            f.append(f"{nom} : le moteur ne la vérifie pas ({R.statut}, η = {R.eta_max:.3f})")
        if abs(R.eta_max - s.eta) > 1e-9:
            f.append(f"{nom} : taux annoncé {s.eta:.4f} ≠ calcul complet {R.eta_max:.4f}")
        if s.cible != (R.eta_max <= X.cible + 1e-9):
            f.append(f"{nom} : taux cible mal dit")
        touchees = [k for k in BASE if k in u and v.get(k) != u.get(k)]
        if touchees:                                          # 2. les données de base
            f.append(f"{nom} : données de base changées {touchees}")
        if "corn_u" in v and (v["corn_u"] == PERSO or v["corn_u"] not in CATALOGUE):
            f.append(f"{nom} : cornière hors catalogue ({v['corn_u']})")       # 3.
    return f, X


total, essais_maxi, duree_maxi = 0, 0, 0.0
for cle, jeux in JEUX.items():
    cfg = A.CONFIGS[cle]
    print(f"\n=== {cle} ===")
    for i, modifs in enumerate(jeux):
        t0 = time.perf_counter()
        f, X = fautes_propositions(cle, modifs)
        dt = time.perf_counter() - t0
        total += len(X.solutions)
        essais_maxi, duree_maxi = max(essais_maxi, X.essais), max(duree_maxi, dt)
        chk(f"jeu {i + 1} {modifs or '(défaut)'} : {len(X.solutions)} proposition(s), chacune VÉRIFIÉE par le moteur (recalculée ici), "
            f"au taux du calcul complet, données de base gardées, pièce réelle — {X.essais} calculs, {dt * 1000:.0f} ms",
            not f and X.complet, f"{f[:3]} complet={X.complet}")
        # … et par l'écran du site : chaque ligne du sélecteur se recopie, l'assemblage est alors vérifié
        _c, u = saisies(cle, modifs)
        html = A.sel_solution(cfg, u)
        fs = []
        for s in X.solutions:
            nom = cfg.predim["nom"](s.u)
            if f'data-choix="{A.C.esc(nom)}"' not in html:
                fs.append(f"{nom} absente du sélecteur")
                continue
            v = dict(u)
            A.choisir_solution(cfg, v, nom)
            R = A.moteur(cfg, v)
            if not R.verified or v != A.valeurs_normales(cfg, v) or [k for k in BASE if k in u and v[k] != u[k]]:
                fs.append(f"{nom} : recopiée par l'écran, {R.statut} ; base changée {[k for k in BASE if k in u and v[k] != u[k]]}")
            if abs(R.eta_max - s.eta) > 1e-9:
                fs.append(f"{nom} : l'écran recopie un autre assemblage (η {R.eta_max:.4f} ≠ {s.eta:.4f})")
        chk(f"jeu {i + 1} : l'écran liste ces {len(X.solutions)} proposition(s) et les recopie telles quelles (assemblage vérifié, "
            "saisies normalisées, données de base gardées)", not fs and len(X.solutions) >= 1, str(fs[:3]))
    # 4. la plus simple d'abord
    _f, X0 = fautes_propositions(cle, {})
    chk("données par défaut : une file suffit — aucune proposition à deux files ; la première tient le taux cible et n'a "
        "aucune réserve", X0.solutions and not any(s.deux_files for s in X0.solutions) and X0.solutions[0].cible
        and not X0.solutions[0].reserves and all(s.plage for s in X0.solutions),
        str([(cfg.predim["nom"](s.u), s.deux_files, s.cible, len(s.reserves)) for s in X0.solutions]))

print("\n=== Une réserve que la disposition lève elle-même : le plat d'âme plus court que la hauteur recommandée ===")


def fautes_hauteur(cle, modifs, module=PR):
    """Les propositions de plat d'âme et l'avis « hp_rec » du moteur (hp < 0,6·h de la poutre portée, maintien en torsion) :
    celles qui le portent encore, et celles qu'on aurait allongées PLUS que nécessaire (un cran d'entraxe de moins ne
    porterait pas l'avis). ([fautes], X)"""
    cfg, u = saisies(cle, modifs)
    f, X = fautes_propositions(cle, modifs, module)
    f = list(f)
    for s in X.solutions:
        nom = cfg.predim["nom"](s.u)
        if any(a.id == PR.RESERVE_PLAT for a in s.R.alerts):
            f.append(f"{nom} : porte l'avis hp_rec")
        if "hauteur" in s.retouches:
            n, p1, hp = int(s.u["n1_u"]), float(s.u["p1_u"]), float(s.u["hp_u"])
            court = A.moteur(cfg, A.valeurs_normales(cfg, dict(s.u, p1_u=p1 - 5, hp_u=hp - 5 * (n - 1))))
            if not any(a.id == PR.RESERVE_PLAT for a in court.alerts):
                f.append(f"{nom} : allongé plus que nécessaire (p1 = {p1 - 5:.0f} mm suffisait)")
            if p1 > PR._plafond_p1(s.R) + 1e-9:
                f.append(f"{nom} : entraxe au-delà du maximum du moteur")
    return f, X


cfg, u = saisies("plat_ame", dict(V_Ed=30))
R0 = A.moteur(cfg, u)
f, X = fautes_hauteur("plat_ame", dict(V_Ed=30))
s0 = X.solutions[0] if X.solutions else None
chk("plat d'âme, V = 30 kN (HEA 300 : 0,6·h = 174 mm) : deux boulons suffisent, et le plat proposé fait 175 mm — son entraxe "
    "étiré de 60 à 95 mm, sous le maximum du Tableau 3.3 que le moteur nomme (112 mm = 14 × 8 mm) ; un cran de moins "
    "(170 mm) porterait l'avis ; aucune proposition ne le porte",
    not f and s0 is not None and cfg.predim["nom"](s0.u) == "2 M20 · plat 175×90×8" and s0.u["p1_u"] == 95 and not s0.reserves
    and "hauteur" in s0.retouches and PR._plafond_p1(s0.R) == 112.0 and 0.6 * R0.h_S == 174.0,
    f"{f} {[(cfg.predim['nom'](s.u), s.u['p1_u'], s.retouches) for s in X.solutions]}")
cfg, u = saisies("plat_ame_poteau", dict(V_Ed=30))
R0 = A.moteur(cfg, u)
f, X = fautes_hauteur("plat_ame_poteau", dict(V_Ed=30))
s0 = X.solutions[0] if X.solutions else None
chk("plat d'âme sur poteau, V = 30 kN (IPE 400 : 0,6·h = 240 mm) : deux rangées ne peuvent pas atteindre 240 mm sous "
    "l'entraxe maximal — la recherche passe à TROIS rangées (« 3 M20 · plat 240×90×8 »), sans l'avis",
    not f and s0 is not None and cfg.predim["nom"](s0.u) == "3 M20 · plat 240×90×8" and s0.n == 3 and not s0.reserves
    and 0.6 * R0.h_S == 240.0, f"{f} {[(cfg.predim['nom'](s.u), s.n, s.u['p1_u']) for s in X.solutions]}")
for cle in ("plat_ame", "plat_ame_poteau"):
    f, X = fautes_hauteur(cle, {})
    chk(f"{cle}, données par défaut : aucune proposition ne porte l'avis de hauteur, aucune n'est allongée plus que nécessaire",
        not f and len(X.solutions) >= 3, str(f))

print("\n=== Des cornières dont une aile, ou les deux, est SOUDÉE ===")
SOUDEE = "Soudée"


def fautes_soudees(cle, modifs, module=PR):
    """Les propositions de cornières à ailes soudées : celles que le moteur ne vérifie pas (``fautes_propositions``),
    deux propositions de même nom, un nom ou un détail qui parle de boulons qui n'existent pas, une gorge non dite.
    ([fautes], X)"""
    cfg, u = saisies(cle, modifs)
    f, X = fautes_propositions(cle, modifs, module)
    f = list(f)
    noms = [cfg.predim["nom"](s.u) for s in X.solutions]
    if len(set(noms)) != len(noms):
        f.append(f"deux propositions de même nom : {noms}")
    for s, nom in zip(X.solutions, noms):
        bS, bP = s.u["fix_S"] != SOUDEE, s.u["fix_P"] != SOUDEE
        det = cfg.predim["detail"](s.u)
        if not (bS or bP) and (re.search(r"\bM\d\d\b", nom + det) or "e1 " in det or s.boulons != 0 or "soudées" not in nom):
            f.append(f"{nom} : parle de boulons qui n'existent pas ({det})")
        if (bS or bP) and (s.boulons == 0 or not re.match(r"^\d+(×2)? M\d\d · L", nom)):
            f.append(f"{nom} : boulons mal dits ({s.boulons})")
        for X_, aile in (("S", "B"), ("P", "A")):
            if s.u["fix_" + X_] == SOUDEE and f"ailes {aile} soudées (a {A.C.nb(float(s.u['a_' + X_]))})" not in det:
                f.append(f"{nom} : la gorge des ailes {aile} n'est pas dite ({det})")
    return f, X


for cle in ("doubles_cornieres", "doubles_cornieres_poteau"):
    cfg = A.CONFIGS[cle]
    for titre, modifs, mini in (("les deux ailes soudées", dict(fix_P=SOUDEE, fix_S=SOUDEE), 1),
                                ("ailes A soudées, ailes B boulonnées", dict(fix_P=SOUDEE), 2),
                                ("ailes B soudées, ailes A boulonnées", dict(fix_S=SOUDEE), 2)):
        f, X = fautes_soudees(cle, modifs)
        chk(f"{cle}, {titre} : {len(X.solutions)} proposition(s) vérifiée(s), de noms distincts ; le nom ne parle de boulons que "
            "s'il y en a (« L80x80x8 · hc 195 · soudées »), la gorge d'une aile soudée est dite",
            not f and len(X.solutions) >= mini, f"{f} {[cfg.predim['nom'](s.u) for s in X.solutions]}")

print("\n=== Le cas de la capture du propriétaire : doubles cornières, HEA 300 sur HEA 400, V = 200 kN ===")
cfg, u = saisies("doubles_cornieres", dict(V_Ed=200))
R0 = A.moteur(cfg, u)
chk("l'assemblage saisi ne passe pas (128 %), et l'ESTIMATION du moteur ne trouvait aucune combinaison (« AUCUNE combinaison "
    "ne respecte le taux cible ») : c'est l'écran que le propriétaire a photographié",
    not R0.verified and A.C.pct(R0.eta_max) == 128 and not R0.pd.found and R0.pd.msg.startswith("AUCUNE combinaison"),
    f"{R0.eta_max:.3f} {R0.pd.found}")
f, X = fautes_propositions("doubles_cornieres", dict(V_Ed=200))
chk(f"la recherche propose {len(X.solutions)} disposition(s) VÉRIFIÉE(s) pour ce cas — sur deux files (la hauteur manque pour "
    "une rangée de plus), avec une cornière du catalogue",
    not f and len(X.solutions) >= 1 and all(s.deux_files for s in X.solutions) and X.solutions[0].cible,
    f"{f} {[(cfg.predim['nom'](s.u), round(s.eta, 3)) for s in X.solutions]}")

print("\n=== Sans solution : la vérification qui bloque, et « changer de profilé » ===")
cfg, u = saisies("plat_ame", dict(V_Ed=250))
X = PR.chercher(cfg.predim["moteur"], cfg.S, A.a_calculer(cfg, u), "plat")
chk("plat d'âme, HEA 300 sur HEA 400, V = 250 kN : aucune proposition ; le meilleur candidat reste bloqué par la pression "
    "diamétrale de l'âme de la poutre SECONDAIRE (133 %) — la géométrie de ce candidat est valide",
    not X.solutions and X.blocage is not None and X.blocage.gov.grp == "Poutre secondaire" and X.blocage.gov.key == "pdS"
    and X.blocage.R.geo_ok and X.blocage.R.dist_ok and 1.2 < X.blocage.R.eta_max < 1.5,
    str(X.blocage and (X.blocage.gov.grp, X.blocage.gov.key, X.blocage.R.eta_max)))
html = A.sel_solution(cfg, u)
chk("l'écran le dit : « Aucune solution avec ces profilés », la vérification nommée et son taux, « Changer le profilé de la "
    "poutre secondaire » ; aucune ligne à choisir, pas de « Aucune section dans cette sélection »",
    "<b>Aucune solution avec ces profilés.</b>" in html and "Poutre secondaire · pression diamétrale – âme" in html
    and "133" in html and "Changer le profilé de la poutre secondaire" in html and "data-choix=" not in html
    and "Aucune section" not in html, html[:600])
cfg, u = saisies("platine_about", dict(V_Ed=400))
html = A.sel_solution(cfg, u)
chk("platine d'about, V = 400 kN : la section grugée bloque — l'écran ajoute « ou réduire le grugeage »",
    "Aucune solution avec ces profilés" in html and "ou réduire le grugeage" in html, html[:500])
cfg, u = saisies("plat_ame", dict(prof_P="IPE 500", prof_S="IPE 300", V_Ed=180, d_top=30, d_nt=0, l_n=0, z_C=45))
X = PR.chercher(cfg.predim["moteur"], cfg.S, A.a_calculer(cfg, u), "plat")
html = A.sel_solution(cfg, u)
chk("une géométrie des POUTRES non valide (la semelle de la secondaire heurte la principale : grugeage à 0) : aucune "
    "disposition de boulons ne la lève — la recherche rend l'alerte bloquante, l'écran la dit et renvoie aux poutres",
    not X.solutions and X.blocage is None and X.geometrie is not None and any(a.id == "dnt_min" for a in X.geometrie.alerts)
    and "la géométrie n'est pas valide" in html and "Grugeage supérieur insuffisant" in html and "À corriger d'abord" in html
    and "data-choix=" not in html, html[:500])
cfg, u = saisies("plat_ame_poteau", dict(prof_S="IPE 120", V_Ed=60))
X = PR.chercher(cfg.predim["moteur"], cfg.S, A.a_calculer(cfg, u), "plat")
html = A.sel_solution(cfg, u)
chk("une poutre portée trop basse (IPE 120) : aucune disposition ne tient dans la hauteur ou ne passe — l'écran dit quoi "
    "changer, sans erreur", not X.solutions and "Aucune solution avec ces profilés" in html, html[:400])

print("\n=== L'écran : le bloc du résultat, le sélecteur, plus de mode prédimensionnement ===")
etat = A.nouvel_etat()
for cle in JEUX:
    A.modifier(etat, ["type"], cle)
    r = A.resultat(etat)
    res = r["resultat"]
    chk(f"{cle} : le résultat porte le bloc « Prédimensionnement » sous les vérifications — une ligne « Solutions » dont le "
        "bouton « Proposer » ouvre le sélecteur ; il dit ce qui est gardé ; pas de taux estimé, pas de « tout le calcul »",
        res.count('data-pk="solution"') == 1 and res.index('data-pk="solution"') > res.index("Vérifications")
        and '<span class="n">Prédimensionnement</span>' in res and "efforts, profilés et niveaux gardés" in res
        and "taux estimé" not in res and 'data-calcul="predim"' not in res and 'id="ck-predim"' not in r["volets"]
        and [v["cle"] for v in r["vues"]] == (["elevation", "plan", "droite"] if "poteau" not in cle else ["elevation", "plan", "coupe"]),
        str([v["cle"] for v in r["vues"]]))
    don = r["donnees"]
    chk(f"{cle} : plus de bascule « Mode » ni de champ verrouillé dans les Données ; les règles du prédimensionnement (taux "
        "cible, diamètres, pinces) sont dans « Options », la dernière rubrique",
        'data-chemin="v/mode_calc"' not in don and not re.search(r'data-chemin="v/[^"]*"[^>]* disabled', don)
        and all(f'data-chemin="v/{k}"' in don for k in ("eta_c", "pd_dmin", "pd_dmax", "k_e1", "k_p1", "k_e2"))
        and don.index('data-chemin="v/eta_c"') > don.index(">Options<"), don[:200])
chk("les mots communs : le sous-titre du sélecteur et la ligne du résultat disent « efforts, profilés et niveaux gardés »",
    "efforts, profilés et niveaux gardés" in A.SOUS_TITRE_SOLUTIONS and "efforts, profilés et niveaux gardés" in A.PROPOSITION_GARDE)
# une note enregistrée en « mode prédimensionnement » (avant le 25/11/2026)
for cle in JEUX:
    cfg = A.CONFIGS[cle]
    el = A.nouvel_assemblage(cle)
    el["values"]["mode_calc"] = cfg.E.MODE_PREDIM
    R_pred = cfg.calculer(A.a_calculer(cfg, el["values"]))
    relu = A.normaliser({"version": 1, "assemblages": [el], "actif": 0})
    v = relu["assemblages"][0]["values"]
    R = A.moteur(cfg, v)
    chk(f"{cle} : une note enregistrée en mode prédimensionnement est relue en vérification, sa proposition RECOPIÉE — le "
        f"même assemblage qu'avant (η {R_pred.eta_max:.4f}), relecture idempotente",
        R_pred.pred and v["mode_calc"] == cfg.E.MODE_VERIF and not R.pred and abs(R.eta_max - R_pred.eta_max) < 1e-9
        and R.verified == R_pred.verified and A.normaliser(relu) == relu, f"{R.eta_max} {R_pred.eta_max}")
etat = A.nouvel_etat()
msg = A.modifier(etat, ["v", "mode_calc"], A.CONFIGS[etat["assemblages"][0]["type"]].E.MODE_PREDIM)
chk("la bascule vers le mode prédimensionnement (une page restée ouverte) est refusée, avec sa raison ; l'assemblage reste "
    "en vérification", bool(msg) and "n'existe plus" in msg and etat["assemblages"][0]["values"]["mode_calc"] == A.MODE_VERIF, msg)
chk(f"au moins 60 propositions vérifiées sur les six attaches ({total}) ; la recherche la plus longue : {essais_maxi} calculs "
    f"complets (budget {PR.BUDGET}), {duree_maxi * 1000:.0f} ms", total >= 60 and essais_maxi < PR.BUDGET and duree_maxi < 5, f"{total} {essais_maxi}")

print("\n=== Mutations : la recherche abîmée fait rougir les contrôles ===")
SOURCE = open(os.path.join(RACINE, "acier", "assemblages", "propositions.py"), encoding="utf-8").read()


def mutant(ancien, nouveau):
    assert SOURCE.count(ancien) == 1, (ancien, SOURCE.count(ancien))
    spec = importlib.util.spec_from_loader("propositions_mutant", loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(SOURCE.replace(ancien, nouveau), "propositions_mutant.py", "exec"), m.__dict__)
    return m


MUTATIONS = [
    ("la recherche accepte un candidat que le moteur ne vérifie pas",
     "                    if not R.verified:\n", "                    if False:\n",
     lambda m: fautes_propositions("plat_ame", dict(V_Ed=250), m, au_moins=0)[0]),
    ("une proposition change un NIVEAU (z sous le dessus)",
     "    v = M.apply_solution(base, r)\n    p2 = _p2(base, r)\n    v.update(e1_u=s.e1,",
     "    v = M.apply_solution(base, r)\n    p2 = _p2(base, r)\n    v[\"z_C\"] = N(v.get(\"z_C\")) + 5\n    v.update(e1_u=s.e1,",
     lambda m: fautes_propositions("plat_ame", {}, m)[0]),
    ("une proposition change un EFFORT",
     "    base = dict(u, mode_calc=MODE_VERIF)\n", "    base = dict(u, mode_calc=MODE_VERIF, V_Ed=N(u.get(\"V_Ed\")) * 0.5)\n",
     lambda m: fautes_propositions("platine_about", {}, m)[0]),
    ("la deuxième file n'est plus essayée (cornières)",
     "    if deux and grp in (\"Boulons\", portee, porteuse, piece) and _boulonne(v, cote):", "    if False:",
     lambda m: fautes_propositions("doubles_cornieres", dict(V_Ed=200), m)[0]),
    ("deux files proposées d'abord, quand une file suffit",
     "            for deux in ((False, True) if G.files else (False,)):", "            for deux in ((True,) if G.files else (False,)):",
     lambda m: ["deux files"] if any(s.deux_files for s in fautes_propositions("plat_ame", {}, m)[1].solutions) else []),
    ("une cornière « personnalisée » proposée à la place d'une cornière du catalogue",
     "                mieux = (cle, dict(corn_u=c[\"n\"], orient=orient, gA_u=gA))",
     "                mieux = (cle, dict(corn_u=\"Personnalisé\", orient=orient, gA_u=gA, k1_u=c[\"a1\"], k2_u=c[\"a2\"], kt_u=c[\"t\"] - 1, kr_u=c[\"r\"]))",
     lambda m: fautes_propositions("doubles_cornieres", {}, m)[0]),
    ("le taux annoncé n'est plus celui du calcul complet",
     "boulons=G.boulons(v), eta=R.eta_max,", "boulons=G.boulons(v), eta=R.eta_max * 0.9,",
     lambda m: fautes_propositions("platine_about_poteau", {}, m)[0]),
    ("la réserve de hauteur n'est plus levée (un plat plus court que 0,6·h est proposé)",
     "files=True, reserve=RESERVE_PLAT, lever=_lever_plat, plafond=_plafond_p1, signature=None),",
     "files=True, reserve=None, lever=None, plafond=None, signature=None),",
     lambda m: fautes_hauteur("plat_ame", dict(V_Ed=30), m)[0]),
    ("le plat allongé plus que nécessaire (le plus grand entraxe possible, au lieu du plus petit qui lève l'avis)",
     "            bas, haut = 1, crans - 1\n", "            bas, haut = 1, 0\n",
     lambda m: fautes_hauteur("plat_ame", dict(V_Ed=30), m)[0]),
    ("des cornières soudées des deux côtés proposées plusieurs fois (une par diamètre de boulon qui n'existe pas)",
     "    if G.signature is not None:\n", "    if False:\n",
     lambda m: fautes_soudees("doubles_cornieres", dict(fix_P=SOUDEE, fix_S=SOUDEE), m)[0]),
    ("une rangée de plus n'est plus essayée quand l'entraxe ne suffit pas à lever la réserve",
     "                        if G.reserve is None or not _a_alerte(R, G.reserve) or de_plus >= RANGEES_DE_PLUS:",
     "                        if True:",
     lambda m: fautes_hauteur("plat_ame_poteau", dict(V_Ed=30), m)[0]),
]
chk("témoin : la recherche telle quelle passe ces contrôles", not any(c(PR) for _t, _a, _n, c in MUTATIONS))
survivants = [titre for titre, ancien, nouveau, controle in MUTATIONS if not controle(mutant(ancien, nouveau))]
chk(f"{len(MUTATIONS)} mutations de la recherche : toutes vues rouges", not survivants, str(survivants))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
