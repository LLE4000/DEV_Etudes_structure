# -*- coding: utf-8 -*-
"""Le TAUX D'ARMATURE SIMPLIFIÉ des poutres en béton armé (``beton/poutre/taux.py``, option de ``api_web/poutre.py``).

    /usr/bin/python3.12 tests/test_poutre_taux.py

  1. À LA MAIN : une poutre 30 × 60 à deux sections, chaque nombre recalculé ici avec ``math`` seul sur des valeurs
     écrites en clair (masses linéiques, longueur d'un cadre et d'une épingle, cours par mètre, latérales, majoration,
     volume, taux brut, arrondi, taux de la poutre pondéré par les longueurs).
  2. INVARIANTS : la majoration, le pas, le retour, l'arrondi supérieur (un multiple exact ne monte pas), le taux de la
     poutre entre ceux de ses sections, une section sans longueur laissée de côté, aucune longueur → pas de taux.
  3. L'OPTION : éteinte, rien ne change (ni l'entrée normalisée, ni la sortie du moteur) ; ses réglages et la longueur
     d'une section dans leurs bornes, refusés au-delà avec le nom du champ.
  4. L'ÉCRAN = LE MOTEUR : les champs (case, réglages, longueur) reliés à l'état ; chaque nombre montré (ligne de la
     section, tête, fenêtres de calcul) est celui du calcul à la main ; les refus laissent l'état intact ; un nom piégé
     est échappé ; la note PDF ne reçoit le taux de la poutre que si la case « dans la note de calcul » est cochée
     (éteinte par défaut : on ne donne pas toujours le taux à qui reçoit la note).
  5. MUTATIONS : le moteur rechargé depuis sa source, modifié d'UNE formule, est rejeté par les contrôles 1 et 2 ;
     l'écran modifié d'UN affichage est rejeté par les contrôles 4 ; la note qui donnerait le taux sans qu'on le demande
     est rejetée. Une mutation témoin (un commentaire) ne l'est pas.

La PARITÉ avec l'écran Streamlit (``modules/poutre.py`` rejoué) est dans ``tests/test_poutre_moteur.py`` (section 9).
"""
import copy
import json
import math
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import poutre as P  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.poutre import entree as E  # noqa: E402
from beton.poutre import moteur as M  # noqa: E402
from beton.poutre import taux as T  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK   " if cond else "  ÉCHEC ") + nom + ("" if cond else f"   → {str(info)[:400]}"))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as f:
        return f.read()


def proche(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ------------------------------------------------------------------ la poutre de l'exemple
# 30 × 60 cm, enrobage 3 cm, jeux 0 et 1 cm.
#   Section A (travée, 4,00 m) : 3Ø20 en bas, 2Ø12 en haut ; un cadre Ø8 + une épingle Ø8, pas de 20 cm.
#   Section B (appui, 2,00 m)  : 2Ø20 en bas, 4Ø16 en haut ; un cadre Ø8, pas de 10 cm.
#   Latérales automatiques (Ø10, s_max 30 cm).
POUTRE = {"nom": "V1", "b": 30.0, "h": 60.0, "enrobage": 3.0, "taux": {"actif": True},
          "sections": [{"nom": "A", "M_inf": 120.0, "V": 60.0, "inf": [{"n": 3, "d": 20}], "sup": [{"n": 2, "d": 12}],
                        "etriers": [{"type": "etrier", "d": 8}, {"type": "epingle", "d": 8}], "pas": 20.0, "longueur": 4.0},
                       {"nom": "B", "M_sup": 95.0, "V": 120.0, "inf": [{"n": 2, "d": 20}], "sup": [{"n": 4, "d": 16}],
                        "etriers": [{"type": "etrier", "d": 8}], "pas": 10.0, "longueur": 2.0}]}


def a_la_main():
    """L'exemple recalculé avec ``math`` seul. Rend {"A": {...}, "B": {...}, "poutre": {...}}."""
    rho = 7850.0                                             # kg/m³
    ml = {d: rho * math.pi * (d / 1000.0) ** 2 / 4.0 for d in (8, 10, 12, 16, 20)}   # kg/m
    b, h, c, retour, maj = 30.0, 60.0, 3.0, 10.0, 5.0
    cadre = 2.0 * ((b - 2 * c) + (h - 2 * c)) + 2.0 * retour  # 2·(24 + 54) + 20 = 176 cm
    epingle = (h - 2 * c) + 2.0 * retour                      # 54 + 20 = 74 cm
    volume = 0.30 * 0.60 * 1.0                                # m³ par mètre
    # les latérales : écart entre les lits 1 = h − dist₁,inf − dist₁,sup ; dist₁ = c + arr½(Ø_étrier/10 + Ø/20)
    arr = lambda x: math.ceil(x * 2.0) / 2.0                  # noqa: E731 — au 0,5 cm supérieur
    out = {}
    for nom, n_inf, d_inf, n_sup, d_sup, pas, epingles in (("A", 3, 20, 2, 12, 20.0, 1), ("B", 2, 20, 4, 16, 10.0, 0)):
        cours = 100.0 / pas
        ecart = h - (c + arr(0.8 + d_inf / 20.0)) - (c + arr(0.8 + d_sup / 20.0))
        n_lat = (math.ceil(ecart / 30.0) - 1) if ecart > 30.0 else 0
        lignes = [n_inf * ml[d_inf], n_sup * ml[d_sup], cours * (cadre / 100.0) * ml[8]]
        if epingles:
            lignes.append(cours * (epingle / 100.0) * ml[8])
        if n_lat:
            lignes.append(2 * n_lat * ml[10])
        masse = sum(lignes)
        maj_ = masse * (1.0 + maj / 100.0)
        brut = maj_ / volume
        out[nom] = {"lignes": lignes, "masse": masse, "masse_majoree": maj_, "volume": volume, "taux_brut": brut,
                    "taux": math.ceil(brut / 5.0) * 5.0, "cours": cours, "n_lat": n_lat, "ecart": ecart}
    lg = {"A": 4.0, "B": 2.0}
    masse = sum(out[k]["masse_majoree"] * lg[k] for k in lg)
    vol = sum(out[k]["volume"] * lg[k] for k in lg)
    out["poutre"] = {"longueur": 6.0, "masse": masse, "volume": vol, "taux_brut": masse / vol, "taux": math.ceil(masse / vol / 5.0) * 5.0}
    out["cadre"], out["epingle"], out["ml"] = cadre, epingle, ml
    return out


MAIN = a_la_main()


def rejets_moteur(mod):
    """Ce que les contrôles à la main et les invariants reprochent au moteur du taux ``mod`` (le vrai, ou un mutant) :
    la liste des fautes, vide quand tout tient. Le reste du moteur des poutres (géométrie) est le vrai."""
    out = []
    r = M.verifier(POUTRE)
    geos = [s["geometrie"] for s in r["sections"]]
    try:
        secs = [mod.section(g, {"actif": True}) for g in geos]
        for sec, nom in zip(secs, ("A", "B")):
            m = MAIN[nom]
            if len(sec["lignes"]) != len(m["lignes"]):
                out.append(f"{nom} : {len(sec['lignes'])} lignes pour {len(m['lignes'])}")
                continue
            for x, attendu in zip(sec["lignes"], m["lignes"]):
                if not proche(x["masse"], attendu):
                    out.append(f"{nom}/{x['cle']} : masse {x['masse']!r} ≠ {attendu!r}")
            for k in ("masse", "masse_majoree", "volume", "taux_brut", "taux", "cours"):
                if not proche(sec[k], m[k]):
                    out.append(f"{nom}.{k} : {sec[k]!r} ≠ {m[k]!r}")
        tp = mod.poutre(secs, [4.0, 2.0], {"actif": True})
        for k in ("longueur", "masse", "volume", "taux_brut", "taux"):
            if tp is None or not proche(tp[k], MAIN["poutre"][k]):
                out.append(f"poutre.{k} : {None if tp is None else tp[k]!r} ≠ {MAIN['poutre'][k]!r}")
        # les masses linéiques nominales (EN 10080), à 0,5 % près
        for d, nominal in ((8, 0.395), (10, 0.617), (12, 0.888), (16, 1.58), (20, 2.47), (25, 3.85), (32, 6.31)):
            if abs(mod.masse_lineique(d) / nominal - 1.0) > 0.005:
                out.append(f"masse linéique Ø{d} : {mod.masse_lineique(d):.4f} ≠ {nominal}")
        # l'arrondi : au multiple SUPÉRIEUR ; un multiple exact ne monte pas ; zéro reste zéro
        for x, pas, attendu in ((114.15, 5, 115.0), (115.0, 5, 115.0), (115.0000001, 5, 120.0), (89.99, 5, 90.0), (0.0, 5, 0.0),
                                (101.2, 1, 102.0), (3 * 0.1 * 50, 5, 15.0)):
            if not proche(mod.arrondi_superieur(x, pas), attendu):
                out.append(f"arrondi_superieur({x!r}, {pas}) = {mod.arrondi_superieur(x, pas)!r} ≠ {attendu!r}")
        # une section sans longueur est laissée de côté ; aucune longueur : pas de taux de poutre
        seule = mod.poutre(secs, [4.0, None], {"actif": True})
        if seule is None or not proche(seule["taux_brut"], MAIN["A"]["taux_brut"]) or seule["sans_longueur"] != [1] or seule["comptees"] != 1:
            out.append(f"section B sans longueur : {seule!r}")
        if mod.poutre(secs, [None, None], {"actif": True}) is not None or mod.poutre(secs, [0, 0.0], {"actif": True}) is not None:
            out.append("aucune longueur : un taux de poutre est rendu")
        # les réglages : majoration, retour, arrondi
        s0 = mod.section(geos[0], {"actif": True, "majoration": 0.0})
        s10 = mod.section(geos[0], {"actif": True, "majoration": 10.0})
        if not proche(s0["taux_brut"], MAIN["A"]["masse"] / 0.18) or not proche(s10["taux_brut"], 1.10 * MAIN["A"]["masse"] / 0.18):
            out.append("majoration : 0 % et 10 % ne donnent pas masse/V et 1,10·masse/V")
        sr = mod.section(geos[0], {"actif": True, "retour": 0.0})
        attendu = MAIN["A"]["masse"] - 5.0 * 0.20 * MAIN["ml"][8] * 2     # un cadre et une épingle : 2 × 2 retours de 10 cm, 5 cours
        if not proche(sr["masse"], attendu):
            out.append(f"retour nul : masse {sr['masse']!r} ≠ {attendu!r}")
        sa = mod.section(geos[0], {"actif": True, "arrondi": 1})
        if not proche(sa["taux"], math.ceil(MAIN["A"]["taux_brut"])):
            out.append(f"arrondi 1 kg/m³ : {sa['taux']!r}")
        # le pas : deux fois plus serré, deux fois plus d'étriers
        g2 = copy.deepcopy(geos[0])
        g2["etriers"]["pas"] = 10.0
        s2 = mod.section(g2, {"actif": True})
        et = lambda s: sum(x["masse"] for x in s["lignes"] if x["famille"] == "etriers")   # noqa: E731
        if not proche(et(s2), 2.0 * et(secs[0])):
            out.append("pas divisé par deux : la masse des étriers ne double pas")
        # un pas qui ne divise pas le mètre : 100/30 = 3,33 cours par mètre, pas 3 (le taux est un ratio au mètre courant)
        g3 = copy.deepcopy(geos[0])
        g3["etriers"]["pas"] = 30.0
        s3 = mod.section(g3, {"actif": True})
        if not proche(s3["cours"], 100.0 / 30.0) or not proche(et(s3), et(secs[0]) * 20.0 / 30.0):
            out.append(f"pas de 30 cm : {s3['cours']!r} cours par mètre, étriers {et(s3)!r} ≠ {et(secs[0]) * 20.0 / 30.0!r}")
        # les parts font 100 % ; le taux de la poutre est entre ceux de ses sections
        if not all(proche(sum(x["part"] for x in s["lignes"]), 1.0) for s in secs):
            out.append("les parts ne font pas 100 %")
        if tp is not None and not min(s["taux_brut"] for s in secs) - 1e-9 <= tp["taux_brut"] <= max(s["taux_brut"] for s in secs) + 1e-9:
            out.append("le taux de la poutre sort de l'intervalle de ses sections")
    except Exception as e:  # noqa: BLE001 — un mutant qui casse est un mutant rejeté
        out.append(f"{type(e).__name__} : {e}")
    return out


# ------------------------------------------------------------------ 1 et 2
print("=== 1. À la main : la poutre 30 × 60 (A : 3Ø20 / 2Ø12, cadre + épingle Ø8 à 20 cm ; B : 2Ø20 / 4Ø16, cadre Ø8 à 10 cm) ===")
chk("le cadre mesure 2·(24 + 54) + 2·10 = 176 cm, l'épingle 54 + 2·10 = 74 cm", MAIN["cadre"] == 176.0 and MAIN["epingle"] == 74.0)
chk("section A : écart entre les lits 1 = 60 − 5,0 − 4,5 = 50,5 cm > 30 → 1 latérale par face ; 5 cours par mètre",
    MAIN["A"]["ecart"] == 50.5 and MAIN["A"]["n_lat"] == 1 and MAIN["A"]["cours"] == 5.0)
chk("section A à la main : 15,34 kg/m → × 1,05 = 16,11 kg/m → / 0,18 m³ = 89,5 kg/m³ → 90 kg/m³",
    abs(MAIN["A"]["masse"] - 15.3395) < 5e-4 and abs(MAIN["A"]["taux_brut"] - 89.480) < 5e-3 and MAIN["A"]["taux"] == 90.0,
    (MAIN["A"]["masse"], MAIN["A"]["taux_brut"], MAIN["A"]["taux"]))
chk("section B à la main : 19,42 kg/m → × 1,05 = 20,39 kg/m → 113,3 kg/m³ → 115 kg/m³",
    abs(MAIN["B"]["masse"] - 19.4234) < 5e-4 and abs(MAIN["B"]["taux_brut"] - 113.303) < 5e-3 and MAIN["B"]["taux"] == 115.0,
    (MAIN["B"]["masse"], MAIN["B"]["taux_brut"], MAIN["B"]["taux"]))
chk("poutre à la main : (16,11·4 + 20,39·2) / (0,18·6) = 97,4 kg/m³ → 100 kg/m³",
    abs(MAIN["poutre"]["taux_brut"] - 97.425) < 5e-3 and MAIN["poutre"]["taux"] == 100.0, MAIN["poutre"])
fautes = rejets_moteur(T)
chk("le moteur donne ces nombres à 1e-9 près, ligne par ligne (lits, cadre, épingle, latérales), et tient ses invariants "
    "(majoration, retour, pas, arrondi supérieur, section sans longueur, parts)", not fautes, fautes)
RES = M.verifier(POUTRE)
TA = RES["taux_armature"]
chk("moteur.verifier porte « taux_armature » : les réglages, une entrée par section (nom, longueur), la poutre",
    TA["reglages"] == {"majoration": 5.0, "retour": 10.0, "arrondi": 5} and [s["nom"] for s in TA["sections"]] == ["A", "B"]
    and [s["longueur"] for s in TA["sections"]] == [4.0, 2.0] and proche(TA["poutre"]["taux"], 100.0)
    and proche(TA["sections"][0]["taux"], 90.0) and proche(TA["sections"][1]["taux"], 115.0))
f_cadre = TA["sections"][0]["lignes"][2]["formule"]
chk("la formule d'un cadre nomme ses termes (w = 24, h − 2c = 54, retour 10 → 176 cm), celle d'une épingle aussi (74 cm)",
    f_cadre == {"forme": "cadre", "largeur": 24.0, "hauteur": 54.0, "retour": 10.0, "l_cm": 176.0, "complet": True}
    and TA["sections"][0]["lignes"][3]["formule"] == {"forme": "epingle", "hauteur": 54.0, "retour": 10.0, "l_cm": 74.0}, f_cadre)
# un cadre intérieur : l'emprise des barres entourées, étrier compris
p_int = copy.deepcopy(POUTRE)
p_int["sections"] = [dict(p_int["sections"][0], inf=[{"n": 4, "d": 20}], etriers=[{"type": "etrier", "d": 8}, {"type": "etrier", "d": 8, "de": 2, "a": 3}])]
r_int = M.verifier(p_int)
xs = r_int["sections"][0]["geometrie"]["inf"]["lits"][0]["x"]          # 4 barres réparties entre 3 + 0,8 + 1,0 = 4,8 et 25,2 cm
w_int = (xs[2] - xs[1]) + 2.0 + 2 * 0.8                                 # entraxe + Ø barre + 2 Ø étrier
l_int = r_int["taux_armature"]["sections"][0]["lignes"][3]
chk("cadre intérieur (barres 2 à 3 de 4Ø20) : largeur = entraxe 6,8 + Ø 2,0 + 2 × 0,8 = 10,4 cm ; longueur 2·(10,4 + 54) + 20 = 148,8 cm",
    proche(xs[1], 4.8 + 20.4 / 3) and proche(w_int, 10.4) and proche(l_int["formule"]["largeur"], 10.4) and proche(l_int["l_barre"], 1.488)
    and l_int["formule"]["complet"] is False, (xs, w_int, l_int["formule"]))

# ------------------------------------------------------------------ 3. l'option
print("\n=== 3. L'option : éteinte, rien ne change ; ses réglages et la longueur dans leurs bornes ===")
sans = copy.deepcopy(POUTRE)
del sans["taux"]
r_sans = M.verifier(sans)
chk("option éteinte : la sortie du moteur n'a pas de clé « taux_armature », et le reste est identique à l'octet",
    "taux_armature" not in r_sans and json.dumps({k: v for k, v in RES.items() if k not in ("taux_armature", "poutre")}, sort_keys=True)
    == json.dumps({k: v for k, v in r_sans.items() if k != "poutre"}, sort_keys=True))
chk("une poutre neuve normalisée n'a ni « taux » ni « longueur » (une note d'avant reste la même)",
    "taux" not in E.normaliser(E.nouvelle_poutre()) and all("longueur" not in s for s in E.normaliser(E.nouvelle_poutre())["sections"]))
chk("« taux » éteint aux réglages par défaut disparaît ; éteint avec un réglage changé, il est gardé ; allumé, il reste",
    E.normaliser({"taux": {"actif": False}}).get("taux") is None
    and E.normaliser({"taux": {"actif": False, "majoration": 5.0, "retour": 10, "arrondi": 5}}).get("taux") is None
    and E.normaliser({"taux": {"actif": False, "majoration": 7}})["taux"] == {"actif": False, "majoration": 7.0}
    and E.normaliser({"taux": {"actif": True}})["taux"] == {"actif": True})
chk("« note » (le taux aussi dans la note de calcul) n'est écrite que vraie : fausse ou absente, la clé n'y est pas",
    E.normaliser({"taux": {"actif": True, "note": False}})["taux"] == {"actif": True}
    and E.normaliser({"taux": {"actif": True, "note": True}})["taux"] == {"actif": True, "note": True})
chk("normaliser est idempotent avec l'option et les longueurs", E.normaliser(E.normaliser(POUTRE)) == E.normaliser(POUTRE))
chk("la longueur d'une section éteinte reste dans la poutre (rallumer la retrouve)",
    E.normaliser(dict(sans))["sections"][0]["longueur"] == 4.0 and "taux_armature" not in M.verifier(sans))
REFUS = (("majoration négative", {"taux": {"actif": True, "majoration": -1}}, "majoration"),
         ("majoration de 80 %", {"taux": {"actif": True, "majoration": 80}}, "majoration"),
         ("retour négatif", {"taux": {"actif": True, "retour": -2}}, "retour"),
         ("retour de 60 cm", {"taux": {"actif": True, "retour": 60}}, "retour"),
         ("arrondi nul", {"taux": {"actif": True, "arrondi": 0}}, "arrondi"),
         ("arrondi de 2,5", {"taux": {"actif": True, "arrondi": 2.5}}, "arrondi"),
         ("arrondi de 60", {"taux": {"actif": True, "arrondi": 60}}, "arrondi"),
         ("actif en texte", {"taux": {"actif": "oui"}}, "actif"),
         ("note en texte", {"taux": {"actif": True, "note": "oui"}}, "note"),
         ("clé inconnue", {"taux": {"actif": True, "chutes": 3}}, "chutes"),
         ("taux qui n'est pas un dict", {"taux": True}, "taux"),
         ("longueur nulle", {"sections": [{"longueur": 0}]}, "longueur"),
         ("longueur négative", {"sections": [{"longueur": -3}]}, "longueur"),
         ("longueur de 450 (des centimètres ?)", {"sections": [{"longueur": 450}]}, "longueur"),
         ("longueur en texte", {"sections": [{"longueur": "4"}]}, "longueur"))
mal = []
for nom, champ, mot in REFUS:
    try:
        E.normaliser(champ)
        mal.append(f"{nom} : accepté")
    except ValueError as e:
        if mot not in str(e):
            mal.append(f"{nom} : le message ne nomme pas « {mot} » ({e})")
chk(f"{len(REFUS)} saisies hors bornes refusées par ValueError, le champ nommé", not mal, mal)

# ------------------------------------------------------------------ 4. l'écran = le moteur
print("\n=== 4. L'écran : les champs, les nombres montrés, les refus, la note ===")


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def saisir(etat, chemin, valeur):
    return req(module="poutre", action="modifier", etat=etat, chemin=chemin, valeur=valeur)


def texte(h):
    """Le texte visible d'un fragment : balises ôtées, entités rendues, blancs (insécables compris) réduits."""
    h = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", h, flags=re.S)
    h = re.sub(r"<[^>]+>", " ", h)
    for a, b in (("&#x27;", "'"), ("&quot;", '"'), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), (C.NBSP, " "), (C.FINE, " ")):
        h = h.replace(a, b)
    return re.sub(r"\s+", " ", h).strip()


def champs(h):
    """Chaque champ du fragment : chemin → (sorte, valeur affichée | coché)."""
    out = {}
    for m in re.finditer(r"<input[^>]*data-chemin=\"([^\"]+)\"[^>]*>", h):
        tag = m.group(0)
        sorte = re.search(r'data-sorte="([^"]+)"', tag).group(1)
        if 'type="checkbox"' in tag:
            out[m.group(1)] = (sorte, " checked" in tag)
        else:
            v = re.search(r' value="([^"]*)"', tag)
            out[m.group(1)] = (sorte, v.group(1) if v else None)
    return out


def volet(volets, ident):
    """Le HTML d'une fenêtre de calcul (de son ouverture à la suivante)."""
    i = volets.index(f'id="ck-{ident}"')
    j = volets.find('<div class="flyout', i + 1)
    return volets[i:j if j > 0 else len(volets)]


ETAT = {"version": 1, "poutres": [POUTRE], "actif": 0}
ECRAN = req(module="poutre", action="calculer", etat=ETAT)


def rejets_ecran(mod, ecran=None):
    """Ce que les contrôles reprochent à l'écran ``mod`` (api_web.poutre, ou un mutant) sur la poutre de l'exemple :
    chaque nombre montré est comparé au calcul À LA MAIN (pas au moteur). Liste vide quand tout tient."""
    out = []
    try:
        e = ecran or mod.resultat(mod.normaliser(copy.deepcopy(ETAT)))
        res, vol, don = texte(e["resultat"]), e["volets"], e["donnees"]
        if "Taux d'armature 100 kg/m³" not in res:
            out.append(f"tête : « Taux d'armature 100 kg/m³ » absent ({res[:160]})")
        masse, volume = MAIN["poutre"]["masse"], MAIN["poutre"]["volume"]
        if f"{C.fr(masse, 1)} kg · {C.fr(volume, 3)} m³ · 6,00 m".replace(C.FINE, " ") not in res:
            out.append("tête : masse, volume et longueur de la poutre absents")
        for nom, sec in (("A", MAIN["A"]), ("B", MAIN["B"])):
            attendu = f"Taux d'armature {C.fr(sec['taux'], 0)} kg/m³ {C.fr(sec['masse'], 2)} kg/m d'acier · + 5 % · sur {'4,00' if nom == 'A' else '2,00'} m"
            if attendu not in res:
                out.append(f"section {nom} : ligne « {attendu} » absente")
        va = texte(volet(vol, "s0-ta"))
        for morceau in ("Lit 1 inf. 3 Ø20 · 2,466 kg/m 1 m par barre 7,40 kg/m 3,00 m 48 %",
                        "Lit 1 sup. 2 Ø12 · 0,888 kg/m 1 m par barre 1,78 kg/m 2,00 m 12 %",
                        "Cadre C1 5,00 Ø8 par mètre · 0,395 kg/m 2·(w + h − 2c) + 2·retour = 2·(24 + 54) + 2·10 = 176 cm 3,47 kg/m 8,80 m 23 %",
                        "Épingle C2 5,00 Ø8 par mètre · 0,395 kg/m (h − 2c) + 2·retour = 54 + 2·10 = 74 cm 1,46 kg/m 3,70 m 10 %",
                        "Latérales 2 Ø10 · 0,617 kg/m 1 m par barre 1,23 kg/m 2,00 m 8 %",
                        "m = Σ masses = 7,40 + 1,78 + 3,47 + 1,46 + 1,23 = 15,34 kg/m",
                        "m maj = m · (1 + 5 %) = 15,34 · 1,05 = 16,11 kg/m", "V c = b · h · 1 m = 0,1800 m³/m",
                        "taux brut = m maj / V c = 16,11 / 0,1800 = 89,5 kg/m³", "taux = arrondi au 5 kg/m³ supérieur = 90 kg/m³",
                        "Majoration 5 %", "Retour d'étrier 10 cm , compté deux fois", "Acier 7 850 kg/m³"):
            if morceau not in va:
                out.append(f"fenêtre de la section A : « {morceau} » absent")
        vp = texte(volet(vol, "ta"))
        for morceau in ("Section A 16,11 kg/m · 0,1800 m³/m 64,4 kg 4,00 m 90 kg/m³", "Section B 20,39 kg/m · 0,1800 m³/m 40,8 kg 2,00 m 115 kg/m³",
                        "L = Σ l i = 6,00 m", f"M = Σ (m maj,i · l i ) = {C.fr(masse, 1)} kg", "V c = Σ (b · h · l i ) = 1,080 m³",
                        f"taux brut = M / V c = {C.fr(masse, 1)} / 1,080 = 97,4 kg/m³", "taux = arrondi au 5 kg/m³ supérieur = 100 kg/m³",
                        "Ce qui n'est pas compté"):
            if morceau not in vp:
                out.append(f"fenêtre de la poutre : « {morceau} » absent")
        ch = champs(don)
        attendus = {"taux/actif": ("case", True), "taux/note": ("case", False), "taux/majoration": ("nombre", "5"), "taux/retour": ("nombre", "10"),
                    "taux/arrondi": ("entier", "5"), "sections/0/longueur": ("texte", "4"), "sections/1/longueur": ("texte", "2")}
        for k, v in attendus.items():
            if ch.get(k) != v:
                out.append(f"champ {k} : {ch.get(k)!r} ≠ {v!r}")
        if "écrit dans la note" in res or "La note PDF ne donne pas le taux" not in texte(don):
            out.append("case « note » non cochée : l'écran ne dit pas que la note ne donne pas le taux")
        if 'data-calcul="ta"' not in e["resultat"] or e["resultat"].count('data-calcul="s0-ta"') != 1 or e["resultat"].count('data-calcul="s1-ta"') != 1:
            out.append("les fenêtres « ta », « s0-ta », « s1-ta » ne s'ouvrent pas depuis le résultat")
    except Exception as ex:  # noqa: BLE001 — un écran mutant qui casse est rejeté
        out.append(f"{type(ex).__name__} : {ex}")
    return out


fautes = rejets_ecran(P, ECRAN)
chk("option allumée : la tête (100 kg/m³, masse, volume, longueur), la ligne de chaque section, les deux fenêtres de calcul "
    "et les sept champs montrent les nombres du calcul À LA MAIN", not fautes, fautes)
chk("la masse de la poutre affichée est celle du calcul à la main (105,2 kg pour 1,080 m³)",
    C.fr(MAIN["poutre"]["masse"], 1) == "105,2" and proche(MAIN["poutre"]["volume"], 1.08), MAIN["poutre"])

# éteinte : aucun champ de réglage, aucune longueur, aucun résultat de taux — sauf la case
e0 = req(module="poutre", action="nouveau")
ch0 = champs(e0["donnees"])
chk("note neuve : la case « Taux d'armature » est là, éteinte ; ni réglage, ni longueur, ni résultat de taux ; l'état n'a pas de clé « taux »",
    ch0.get("taux/actif") == ("case", False) and not [k for k in ch0 if k.startswith("taux/") and k != "taux/actif" or k.endswith("/longueur")]
    and "Taux d'armature" not in texte(e0["resultat"]) and "ck-ta" not in e0["volets"] and "taux" not in e0["etat"]["poutres"][0])
e1 = saisir(e0["etat"], ["taux", "actif"], True)
ch1 = champs(e1["donnees"])
chk("la case cochée : les trois réglages et la longueur de la section apparaissent ; sans longueur, la tête dit quoi faire",
    not e1.get("message") and e1["etat"]["poutres"][0]["taux"] == {"actif": True} and ch1.get("taux/majoration") == ("nombre", "5")
    and ch1.get("sections/0/longueur") == ("texte", "") and "donner une longueur aux sections" in texte(e1["resultat"])
    and "Taux d'armature 115 kg/m³" in texte(e1["resultat"]), (e1.get("message"), ch1))
e2 = saisir(e1["etat"], ["sections", "0", "longueur"], "4,5")
chk("longueur « 4,5 » : écrite 4,5 m dans la section ; la tête donne le taux de la poutre (115 kg/m³, 41,1 kg, 0,360 m³, 4,50 m)",
    not e2.get("message") and e2["etat"]["poutres"][0]["sections"][0]["longueur"] == 4.5
    and "Taux d'armature 115 kg/m³ 41,1 kg · 0,360 m³ · 4,50 m" in texte(e2["resultat"]), texte(e2["resultat"])[:200])
e3 = saisir(e2["etat"], ["taux", "majoration"], "10")
e3 = saisir(e3["etat"], ["taux", "retour"], "12")
e3 = saisir(e3["etat"], ["taux", "arrondi"], "1")
# 20 × 40, 2Ø16 + 2Ø16, cadre Ø10 à 30 cm : 4·1,5783 + (100/30)·(2·(14 + 34) + 24)/100·0,6165 = 8,7795 kg/m ; × 1,10 / 0,08 = 120,7 → 121
m3 = 4 * 7850 * math.pi * 0.016 ** 2 / 4 + (100 / 30.0) * (2 * (14 + 34) + 2 * 12) / 100.0 * 7850 * math.pi * 0.010 ** 2 / 4
chk("réglages 10 %, 12 cm, 1 kg/m³ : l'état les porte et le taux devient ⌈8,78 · 1,10 / 0,08⌉ = 121 kg/m³",
    e3["etat"]["poutres"][0]["taux"] == {"actif": True, "majoration": 10.0, "retour": 12.0, "arrondi": 1}
    and math.ceil(m3 * 1.10 / 0.08) == 121 and "Taux d'armature 121 kg/m³" in texte(e3["resultat"]), (e3["etat"]["poutres"][0]["taux"], m3))
for nom, chemin, valeur, mot in (("majoration de 80 %", ["taux", "majoration"], "80", "majoration"),
                                 ("retour « dix »", ["taux", "retour"], "dix", "illisible"),
                                 ("arrondi de 2,5", ["taux", "arrondi"], "2,5", "arrondi"),
                                 ("case en texte", ["taux", "actif"], "false", "Case"),
                                 ("réglage inconnu", ["taux", "chutes"], "3", "inconnu"),
                                 ("longueur « abc »", ["sections", "0", "longueur"], "abc", "Longueur illisible"),
                                 ("longueur nulle", ["sections", "0", "longueur"], "0", "longueur"),
                                 ("longueur de 450", ["sections", "0", "longueur"], "450", "mètres")):
    r_ = saisir(e3["etat"], chemin, valeur)
    chk(f"refus : {nom} → un message qui le dit, l'état intact", bool(r_.get("message")) and mot in r_["message"] and r_["etat"] == e3["etat"],
        (r_.get("message"), r_["etat"] == e3["etat"]))
e4 = saisir(e3["etat"], ["sections", "0", "longueur"], "")
chk("longueur vidée : la clé disparaît de la section, la tête redemande une longueur",
    not e4.get("message") and "longueur" not in e4["etat"]["poutres"][0]["sections"][0] and "donner une longueur aux sections" in texte(e4["resultat"]))
e5 = saisir(e2["etat"], ["taux", "actif"], False)
chk("la case décochée (réglages par défaut) : la clé « taux » disparaît, la longueur reste dans la section, plus rien du taux à l'écran",
    "taux" not in e5["etat"]["poutres"][0] and e5["etat"]["poutres"][0]["sections"][0]["longueur"] == 4.5
    and "Taux d'armature" not in texte(e5["resultat"]) and "sections/0/longueur" not in champs(e5["donnees"]))
e6 = saisir(e3["etat"], ["taux", "actif"], False)
e7 = saisir(e6["etat"], ["taux", "actif"], True)
chk("décochée puis recochée avec des réglages changés : ils sont retrouvés (10 %, 12 cm, 1 kg/m³)",
    e6["etat"]["poutres"][0]["taux"] == {"actif": False, "majoration": 10.0, "retour": 12.0, "arrondi": 1}
    and e7["etat"]["poutres"][0]["taux"] == {"actif": True, "majoration": 10.0, "retour": 12.0, "arrondi": 1})
piege = copy.deepcopy(ETAT)
piege["poutres"][0]["nom"] = '<img src=x onerror=alert(1)>"'
piege["poutres"][0]["sections"][0]["nom"] = "<b>A</b>"
ep = req(module="poutre", action="calculer", etat=piege)
chk("un nom de poutre et de section piégés sont échappés dans la tête, les lignes et les fenêtres du taux",
    "<img" not in ep["resultat"] + ep["volets"] and "<b>A</b>" not in ep["volets"] and "&lt;b&gt;A&lt;/b&gt;" in volet(ep["volets"], "ta"))
deux = copy.deepcopy(ETAT)
deux["poutres"].append(dict(E.nouvelle_poutre(), nom="V2"))
ed = req(module="poutre", action="activer", etat=deux, index=1)
chk("l'option est celle de CHAQUE poutre : V2 (éteinte) ne montre aucun taux, V1 garde le sien",
    "Taux d'armature" not in texte(ed["resultat"]) and ed["etat"]["poutres"][0]["taux"] == {"actif": True} and "taux" not in ed["etat"]["poutres"][1])
# la case « Écrire le taux dans la note de calcul » : éteinte par défaut (demande de l'ingénieur, 10/11/2026 : « des fois,
# on n'a pas envie de donner le taux d'armature aux gens »)
en = saisir(e2["etat"], ["taux", "note"], True)
chk("la case « dans la note de calcul » cochée : l'état porte « note », la tête dit « écrit dans la note », l'aide le confirme",
    not en.get("message") and en["etat"]["poutres"][0]["taux"] == {"actif": True, "note": True}
    and "écrit dans la note" in texte(en["resultat"]) and "La note PDF donne le taux de la poutre." in texte(en["donnees"])
    and champs(en["donnees"]).get("taux/note") == ("case", True), (en.get("message"), en["etat"]["poutres"][0].get("taux")))
en2 = saisir(en["etat"], ["taux", "note"], False)
chk("décochée : la clé « note » disparaît, la tête ne le dit plus", en2["etat"]["poutres"][0]["taux"] == {"actif": True}
    and "écrit dans la note" not in texte(en2["resultat"]))
en3 = saisir(saisir(en["etat"], ["taux", "actif"], False)["etat"], ["taux", "actif"], True)
chk("l'option éteinte puis rallumée repart SANS la note (le taux n'y retourne que sur demande)",
    en3["etat"]["poutres"][0]["taux"] == {"actif": True} and champs(en3["donnees"]).get("taux/note") == ("case", False))
r_ = saisir(e2["etat"], ["taux", "note"], "true")
chk("refus : case « note » en texte → un message, l'état intact", bool(r_.get("message")) and r_["etat"] == e2["etat"], r_.get("message"))


def rejets_note(NPm):
    """Ce que les contrôles reprochent à la note PDF ``NPm`` (api_web.note_poutre, ou un mutant) : le taux de la poutre
    n'y est QUE si l'option est allumée, des longueurs données ET la case « dans la note » cochée."""
    out = []
    try:
        def tg(taux):
            p = copy.deepcopy(POUTRE)
            if taux is None:
                p.pop("taux", None)
            else:
                p["taux"] = taux
            return [pl.get("ta_global") for pl in NPm.resultats(P.normaliser({"version": 1, "poutres": [p], "actif": 0}))]
        for nom, taux, attendu in (("option allumée, case « note » non cochée", {"actif": True}, [None, None]),
                                   ("option allumée, case « note » cochée", {"actif": True, "note": True}, [100.0, 100.0]),
                                   ("option éteinte, « note » restée cochée dans l'état", {"actif": False, "note": True}, [None, None]),
                                   ("option jamais allumée", None, [None, None])):
            if tg(taux) != attendu:
                out.append(f"{nom} : la note porte {tg(taux)!r}, attendu {attendu!r}")
        p = copy.deepcopy(POUTRE)
        p["taux"] = {"actif": True, "note": True}
        for s in p["sections"]:
            s.pop("longueur", None)
        if [pl.get("ta_global") for pl in NPm.resultats(P.normaliser({"version": 1, "poutres": [p], "actif": 0}))] != [None, None]:
            out.append("sans longueur de section : la note porte un taux de poutre")
    except Exception as ex:  # noqa: BLE001
        out.append(f"{type(ex).__name__} : {ex}")
    return out


try:
    from api_web import note_poutre as NP
    fautes = rejets_note(NP)
    chk("note PDF : le taux de la poutre (100 kg/m³) n'y est QUE si la case « dans la note de calcul » est cochée — pas par "
        "défaut, pas l'option éteinte, pas sans longueur", not fautes, fautes)
    from ndc_pdf import data as ND
    avec_note = copy.deepcopy(ETAT)
    avec_note["poutres"][0]["taux"] = {"actif": True, "note": True}
    blocs = ND.construire_sections(NP.resultats(P.normaliser(avec_note)))[0]["blocs"]
    blocs0 = ND.construire_sections(NP.resultats(P.normaliser(copy.deepcopy(ETAT))))[0]["blocs"]
    chk("note PDF : case cochée, le bloc « TAUX D'ARMATURE » dit « T.A. de la poutre 100 kg/m³ » ; non cochée, aucun bloc de ce nom",
        blocs[-1] == ("TAUX D'ARMATURE", [("T.A. de la poutre", None, "100", "kg/m³")]) and all(b[0] != "TAUX D'ARMATURE" for b in blocs0),
        (blocs[-1], [b[0] for b in blocs0]))
except ImportError as ex:  # pragma: no cover — le générateur PDF demande reportlab
    NP = None
    chk("le générateur de la note se charge", False, ex)

# ------------------------------------------------------------------ 5. mutations
print("\n=== 5. Mutations : UNE formule du moteur, UN affichage de l'écran, modifiés, sont rejetés ===")
SRC_T = lire("beton", "poutre", "taux.py")
SRC_P = lire("api_web", "poutre.py")


def mutant(src, nom, paquet, avant, apres, fichier=None):
    if src.count(avant) != 1:
        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {nom})")
    m = types.ModuleType(nom)
    m.__package__, m.__file__ = paquet, fichier or nom
    exec(compile(src.replace(avant, apres), nom, "exec"), m.__dict__)
    return m


MUT_MOTEUR = (
    ("masse volumique 7 800 au lieu de 7 850", "RHO_ACIER = 7850.0", "RHO_ACIER = 7800.0"),
    ("masse linéique : π·Ø²/2", "(d_mm / 1000.0) ** 2 / 4.0", "(d_mm / 1000.0) ** 2 / 2.0"),
    ("cadre : un seul retour", "l_cm = 2.0 * (w + haut) + 2.0 * retour", "l_cm = 2.0 * (w + haut) + retour"),
    ("cadre : le contour compté une fois", "l_cm = 2.0 * (w + haut) + 2.0 * retour", "l_cm = (w + haut) + 2.0 * retour"),
    ("épingle : un seul retour", "l_cm = haut + 2.0 * retour", "l_cm = haut + retour"),
    ("hauteur du cadre : h − c", "haut = h - 2.0 * c", "haut = h - c"),
    ("cours par mètre tronqués à l'entier", "cours = 100.0 / pas if pas > 0 else 0.0", "cours = float(int(100.0 / pas)) if pas > 0 else 0.0"),
    ("latérales : une seule face", '2 * int(lat["n"]), float(lat["d"]), 1.0', 'int(lat["n"]), float(lat["d"]), 1.0'),
    ("lits : une barre par lit", 'face, int(lit["n"]), float(lit["d"]), 1.0', 'face, 1, float(lit["d"]), 1.0'),
    ("majoration oubliée", 'masse_maj = masse * (1.0 + float(r["majoration"]) / 100.0)', "masse_maj = masse"),
    ("volume sans l'enrobage (noyau)", "volume = (b / 100.0) * (h / 100.0) * 1.0", "volume = ((b - 2.0 * c) / 100.0) * ((h - 2.0 * c) / 100.0) * 1.0"),
    ("arrondi au plus proche, pas au supérieur", "return math.ceil(x / pas - EPS) * pas", "return round(x / pas) * pas"),
    ("arrondi sans tolérance (un multiple exact monte)", "return math.ceil(x / pas - EPS) * pas", "return math.ceil(x / pas + 1e-12) * pas if x / pas != int(x / pas) else (int(x / pas) + 1) * pas"),
    ("poutre : la masse non pondérée par la longueur", 'masse += s["masse_majoree"] * lg', 'masse += s["masse_majoree"]'),
    ("poutre : le volume non pondéré par la longueur", 'volume += s["volume"] * lg', 'volume += s["volume"]'),
    ("poutre : une section sans longueur comptée pour 1 m", "            sans.append(i)\n            continue", "            sans.append(i)\n            lg = 1.0"),
    ("poutre : le taux d'une seule section (la première)", "    brut = masse / volume\n    return {\"longueur\"", "    brut = sections[0][\"taux_brut\"]\n    return {\"longueur\""),
)
survivants = []
for nom, avant, apres in MUT_MOTEUR:
    if not rejets_moteur(mutant(SRC_T, "beton.poutre.taux_mutant", "beton.poutre", avant, apres)):
        survivants.append(nom)
chk(f"{len(MUT_MOTEUR)} mutations du moteur (masse linéique, cadre, épingle, cours, latérales, majoration, volume, arrondi, "
    "pondération par les longueurs) : toutes rejetées", not survivants, survivants)
chk("la mutation témoin (un commentaire) n'est pas rejetée : les rejets viennent bien des mutations",
    not rejets_moteur(mutant(SRC_T, "beton.poutre.taux_mutant", "beton.poutre", "# kg/m³ — masse volumique de l'acier", "# kg/m³ — masse volumique")))

MUT_ECRAN = (
    ("la tête montre le taux brut (97 au lieu de 100)", '<b>{C.fr(tp["taux"], 0)}{C.NBSP}kg/m³</b>', '<b>{C.fr(tp["taux_brut"], 0)}{C.NBSP}kg/m³</b>'),
    ("la ligne de la section montre la masse majorée", 'cmp_ = (f"{C.fr(sec[\'masse\'], 2)}{C.NBSP}kg/m d\'acier', 'cmp_ = (f"{C.fr(sec[\'masse_majoree\'], 2)}{C.NBSP}kg/m d\'acier'),
    ("la ligne de la section montre le taux brut", 'etat_txt=f"{C.fr(sec[\'taux\'], 0)}{C.NBSP}kg/m³")\n    return ligne, ""', 'etat_txt=f"{C.fr(sec[\'taux_brut\'], 0)}{C.NBSP}kg/m³")\n    return ligne, ""'),
    ("la fenêtre écrit la largeur du cadre à la place de sa hauteur", "2·({C.nb(f['largeur'], 2)} + {C.nb(f['hauteur'], 2)})", "2·({C.nb(f['largeur'], 2)} + {C.nb(f['largeur'], 2)})"),
    ("la fenêtre de la poutre montre la masse au mètre à la place de la masse de la section",
     '''"valeur": f"{C.fr(sec['masse_majoree'] * sec['longueur'], 1)}{C.NBSP}kg" if a_lg else "—",''',
     '''"valeur": f"{C.fr(sec['masse_majoree'], 1)}{C.NBSP}kg" if a_lg else "—",'''),
    ("la longueur d'une section n'est plus un champ", '+ (_longueur(i, s) if ta_on else ""))', '+ "")'),
    ("les réglages ne sont plus montrés", "    if not _ta_actif(p):\n        return f'<div class=\"row full pt-ta\">{case}</div>'", "    if True:\n        return f'<div class=\"row full pt-ta\">{case}</div>'"),
)
survivants = []
for nom, avant, apres in MUT_ECRAN:
    if not rejets_ecran(mutant(SRC_P, "api_web.poutre_mutant", "api_web", avant, apres)):
        survivants.append(nom)
chk(f"{len(MUT_ECRAN)} mutations de l'écran (tête, ligne, fenêtres, champs) : toutes rejetées", not survivants, survivants)
chk("l'écran témoin (un commentaire changé) n'est pas rejeté",
    not rejets_ecran(mutant(SRC_P, "api_web.poutre_mutant", "api_web", "# ------------------------------------------------------------------ le taux d'armature simplifié (option)",
                            "# ------------------------------------------------------------------ le taux d'armature (option)")))
if NP is not None:
    SRC_N = lire("api_web", "note_poutre.py")
    F_N = os.path.join(RACINE, "api_web", "note_poutre.py")
    chk("mutation de la note : « le taux part dans la note sans qu'on le demande » est rejetée",
        bool(rejets_note(mutant(SRC_N, "api_web.note_poutre_mutant", "api_web",
                                'if (p.get("taux") or {}).get("note") else None', "if True else None", F_N))))
    chk("la note témoin (un commentaire changé) n'est pas rejetée",
        not rejets_note(mutant(SRC_N, "api_web.note_poutre_mutant", "api_web", "# C'est le nombre de l'écran, calculé par le moteur du site.",
                               "# Le nombre de l'écran.", F_N)))
chk("après les mutations, le moteur et l'écran chargés sont intacts", not rejets_moteur(T) and not rejets_ecran(P))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
