# -*- coding: utf-8 -*-
"""Le module « Enrobage » du site (api_web/enrobage.py) : ce que l'écran affiche est ce que le moteur calcule.

  1. Affichage = moteur, sur plus de 300 éléments (types, classes, Ø pairs et impairs, feu, variantes) : la valeur
     retenue du verdict, la décomposition, les lignes durabilité et feu, la phrase, les fenêtres de calcul (chaque
     substitution « valeurs = résultat » relue et recalculée à partir des seuls nombres affichés), les avertissements.
  2. Champs : chaque champ de l'écran Streamlit est là, avec les MÊMES options (types, positions, classes, Δc_dev,
     durées, variantes de largeur) et apparaît quand l'écran le montre (durée et largeur : feu coché ; largeur :
     poutre et poteau).
  3. Saisie : les effets de bord de l'écran (classe structurale, largeur au feu), l'écart voulu (le feu décoché garde
     la durée), les refus avec message (Ø non entier, sous le minimum, illisible ; valeurs hors liste) — l'état reste
     intact.
  4. Éléments de la note, résumé de la note (l'élément au plus grand enrobage), liste.
  5. Sélecteur de la largeur : trié, retenue ✓, optimale ★, choix, rôles inconnus.
  6. Dessins : à l'échelle (l'étrier, la barre, c_nom, c_min relus dans le SVG en mm), tableaux (la case retenue).
  7. Sécurité et robustesse : échappement d'un nom piégé partout, notes abîmées refusées clairement, saisies absurdes
     (Ø énorme, 10⁴⁰⁰, « false » dans la case du feu, nom fleuve, rang illisible) refusées avec un message.
  8. BOUT À BOUT : l'écran Streamlit d'origine (modules/enrobage.py rejoué avec un faux st) et le site (routeur.traiter)
     affichent les mêmes valeurs, sur 20 cas, à l'arrondi de l'écran près ; le contrôle sait rougir.
  9. Ce que la relecture du 07/11/2026 a ajouté : l'hypothèse du tableau feu dans les Données, μ_fi seulement pour les
     poteaux et les voiles, des règles sans doublon, Δc_dev avec les armatures, les bornes des champs Ø.
 10. Depuis le 10/11/2026 (lot Q) : le module est masqué et ne montre plus une valeur fausse — le BANDEAU qui renvoie à la
     Bibliothèque › Enrobages › Calcul, et les TROIS ÉCARTS VOULUS avec l'écran Streamlit, tenus à l'écran contre la
     norme : D1 (poteaux : tableau 5.2a, μ_fi = 0,7 ; R240 « non couvert par le tableau »), D2 (voiles R180 et R240 :
     50 et 60 mm), D6 (aucun étrier retiré de a pour une dalle ou un voile). Le bout à bout (§ 8) garde la parité sur
     tout le reste et NOMME ces trois écarts.

    python3.12 tests/test_api_web_enrobage.py
"""
import copy
import html
import json
import os
import random
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import enrobage as W  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.enrobage import donnees as D  # noqa: E402
from beton.enrobage import entree as EN  # noqa: E402
from beton.enrobage import moteur as M  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def texte(h):
    """Le texte lu à l'écran : balises retirées, entités rendues, espaces normalisés."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", h))).replace(" ", " ").replace("\xa0", " ").strip()


def nombre(s):
    return float(s.replace(" ", "").replace(" ", "").replace(",", "."))


def etat_de(**k):
    el = EN.normaliser({**EN.nouvel_element("E1"), "b_min": None, **k})
    return {"version": 1, "elements": [el], "actif": 0}


def mm(x):
    return C.nb(x, 1)


# =================================================================== 1. affichage = moteur
print("\n=== 1. Ce que l'écran affiche = ce que le moteur calcule ===")
rng = random.Random(2611)
CAS = [etat_de()]
for t in D.TYPES:
    for r in D.R_LEVELS:
        for v in (EN.variantes(t, r) or [(None, None)]):
            CAS.append(etat_de(type=t, r=r, b_min=v[0], feu=True, diam_barre=rng.choice([6, 7, 12, 16, 20, 25, 32, 40]),
                               diam_etrier=rng.choice([0, 6, 8, 9, 10, 12]), delta_dev=rng.choice(D.DELTAS_DEV),
                               classe_expo=rng.choice(D.EXPO_CLASSES), classe_struct=rng.choice(D.CLASSES_STRUCTURALES),
                               position=rng.choice(D.POSITIONS)))
for _ in range(200):
    t, r = rng.choice(D.TYPES), rng.choice(D.R_LEVELS)
    vs = EN.variantes(t, r)
    CAS.append(etat_de(type=t, r=r, b_min=rng.choice(vs)[0] if vs else None, feu=rng.random() < 0.6, diam_barre=rng.randint(6, 45),
                       diam_etrier=rng.randint(0, 14), delta_dev=rng.choice(D.DELTAS_DEV), classe_expo=rng.choice(D.EXPO_CLASSES),
                       classe_struct=rng.choice(D.CLASSES_STRUCTURALES), position=rng.choice(D.POSITIONS)))
FX = re.compile(r'<div class="fx">(.*?)</div>', re.S)
ecarts = []
for k, etat in enumerate(CAS):
    out = W.resultat(etat)
    res = M.calculer(etat["elements"][0])
    d, f = res["durabilite"], res["feu"]
    e = []
    v = re.search(r'class="chip xl en-c">([^<]+)<small>', out["resultat"])
    if not v or v.group(1) != mm(res["c_nom"]):
        e.append(f"verdict {v and v.group(1)} ≠ {mm(res['c_nom'])}")
    gouv = texte(re.search(r'<div class="gov">(.*?)</div></div>', out["resultat"], re.S).group(1))
    if f"gouverné par {W.GOUV[res['gouvernant']]}" not in gouv:
        e.append(f"gouvernant : {gouv}")
    dec = re.findall(r'<div class="(g?)"><span>(.*?)</span><b>([^<]+) mm</b></div>', out["resultat"])
    att = [d["c_min_b"], d["c_min_dur"], 0, d["plancher"], d["c_min"], d["delta_dev"]]
    if [nombre(x[2]) for x in dec] != [float(x) for x in att]:
        e.append(f"décomposition {[x[2] for x in dec]} ≠ {att}")
    marque = [i for i, x in enumerate(dec) if x[0] == "g"]
    if marque != [["c_min_b", "c_min_dur", "delta_dur", "plancher"].index(d["terme_c_min"])]:
        e.append(f"terme gouvernant de c_min marqué {marque} ≠ {d['terme_c_min']}")
    lignes = re.findall(r'<div class="el( gov)?" data-calcul="(\w+)".*?<b>(.*?)</b>.*?<span>(.*?)</span>.*?<span class="pct c">([^<]+)</span>',
                        out["resultat"], re.S)
    # (un poteau R240 — écart voulu D1 — n'a PAS de valeur au feu : sa ligne dit « non couvert par le tableau », sans
    #  nombre ni part ; elle est contrôlée à part)
    chiffre = f is not None and not f["sans_valeur"]
    att_l = [("dur", f"{mm(d['c_nom'])} mm")] + ([("feu", f"{mm(f['c_nom'])} mm")] if chiffre else [])
    if [(x[1], texte(x[3]).rsplit(" · ", 1)[-1]) for x in lignes] != att_l:
        e.append(f"lignes {[(x[1], texte(x[3])) for x in lignes]} ≠ {att_l}")
    if [bool(x[0]) for x in lignes] != [res["gouvernant"] == "durabilite"] + ([res["gouvernant"] == "feu"] if chiffre else []):
        e.append("ligne gouvernante mal marquée")
    if f is not None and f["sans_valeur"] and not re.search(
            r'<div class="el" data-calcul="feu"[^>]*><div class="l1"><b>Feu · R240</b><span class="prof"><span>non couvert par le tableau</span></span></div>'
            r'.*?<span class="pct r">non couvert</span>', out["resultat"], re.S):
        e.append("poteau R240 : la ligne « non couvert par le tableau » manque")
    for x, cn in zip(lignes, [d["c_nom"]] + ([f["c_nom"]] if chiffre else [])):
        if texte(x[4]) != f"{int(cn / res['c_nom'] * 100 + 0.5)} %":
            e.append(f"part {texte(x[4])} ≠ {cn}/{res['c_nom']}")
    # les fenêtres : chaque ligne « … = valeurs = résultat » recalculée à partir de ses seuls nombres
    for fx in FX.findall(out["volets"]):
        morceaux = [texte(m) for m in re.findall(r"<span[^>]*>(.*?)</span>", fx, re.S)]
        morceaux = [m for m in morceaux if m != "="]
        expr, resu = morceaux[-2], morceaux[-1]
        if not re.fullmatch(r"[\d ,+−;()max]+", expr.replace(" ", "")) or "mm" not in resu:
            continue
        py = re.sub(r"(\d),(\d)", r"\1.\2", expr).replace("−", "-").replace(";", ",")
        try:
            valeur = eval(py, {"max": max})  # noqa: S307 — une expression faite de nombres, de + − et de max
        except Exception as ex:  # noqa: BLE001
            e.append(f"substitution illisible « {expr} » ({ex})")
            continue
        if abs(valeur - nombre(resu.replace("mm", ""))) > 1e-9:
            e.append(f"« {expr} = {resu} » : {valeur}")
    if f"{mm(res['c_nom'])} mm" not in texte(out["volets"]):
        e.append("la fenêtre « retenu » ne porte pas c_nom")
    # les avertissements du moteur, tous affichés, les graves en rouge
    msgs = re.findall(r'<div class="msg( grave)?">(.*?)</div>', out["resultat"], re.S)
    for a in res["avertissements"]:
        if not any(texte(C.symb_math(a["message"])) == texte(m[1]) and bool(m[0]) == a["grave"] for m in msgs):
            e.append(f"avertissement {a['code']} absent ou mal coloré")
    if (f is None) != any("Feu non vérifié" in texte(m[1]) for m in msgs):
        e.append("invite « feu non vérifié » mal posée")
    # la phrase et le résumé
    if f"Ø{etat['elements'][0]['diam_barre']}" not in out["phrase"] or (f and f["r"] not in out["phrase"]):
        e.append(f"phrase {out['phrase']}")
    if out["resume"]["niveau"] != "i" or out["resume"]["taux"] is not None or out["statut_html"] != C.statut("i"):
        e.append("ce n'est pas une consultation")
    if e:
        ecarts.append((k, etat["elements"][0], e))
chk(f"{len(CAS)} éléments : verdict, décomposition, lignes, parts, fenêtres (substitutions recalculées), avertissements, phrase",
    not ecarts, ecarts[:2])

# =================================================================== 2. les champs de l'écran
print("\n=== 2. Les champs de l'écran Streamlit, avec les mêmes options ===")
MENU = re.compile(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>(.*?)</select>', re.S)
SEG = re.compile(r'<button type="button" class="(?:on)?" data-chemin="([^"]+)" data-sorte="choix" data-valeur="([^"]*)"')


def champs(etat):
    h = W.resultat(etat)["donnees"]
    menus = {c: re.findall(r'<option value="([^"]*)"', o) for c, o in MENU.findall(h)}
    segs = {}
    for c, v in SEG.findall(h):
        segs.setdefault(c, []).append(v)
    nums = re.findall(r'<input data-chemin="([^"]+)" data-sorte="(?:nombre|entier|texte)"', h)
    cases = re.findall(r'<input type="checkbox" data-chemin="([^"]+)"', h)
    return menus, segs, nums, cases, h


menus, segs, nums, cases, h = champs(etat_de())
chk("type (4) et position (2) : les listes de l'écran", segs.get("type") == D.TYPES and segs.get("position") == D.POSITIONS, segs)
chk("Δc_dev : 10, 5, 0 mm, dans l'ordre de l'écran", segs.get("delta_dev") == [str(x) for x in D.DELTAS_DEV], segs.get("delta_dev"))
chk("classe d'exposition : les sept classes, libellées « X — titre » comme l'écran",
    menus.get("classe_expo") == D.EXPO_CLASSES and all(f"{k} — {html.escape(D.EXPOSITION_INFO[k][0], quote=True)}" in h for k in D.EXPO_CLASSES))
chk("classe structurale : S1 à S6, chacune avec son c_min,dur pour l'exposition, la proposée marquée",
    menus.get("classe_struct") == D.CLASSES_STRUCTURALES and "S1 — 10 mm (proposée)" in h and "S6 — 25 mm<" in h)
chk("Ø barre et Ø étrier, le nom ; la case du feu", nums == ["nom", "diam_barre", "diam_etrier"] and cases == ["feu"], (nums, cases))
chk("feu non coché : ni durée ni largeur (comme l'écran)", "r" not in menus and "b_min" not in menus)
chk("les textes de l'exposition (mécanisme, quand la choisir, exemples) et le guide de l'écran",
    all(html.escape(x, quote=True) in h for x in D.EXPOSITION_INFO["X0"][1:]) and "Règle pratique" in h
    and "<b>plusieurs classes</b>" in h and "<li>XC1 : intérieur sec, ou béton en permanence immergé</li>" in h)
chk("les aides de l'écran (étrier, Δc_dev, classe structurale) sont reprises",
    all(html.escape(a, quote=True)[:40] in h for a in (D.AIDE_ETRIER, D.AIDE_DELTA_DEV, D.AIDE_CLASSE_STRUCT)))
for t in D.TYPES:
    menus, segs, nums, cases, h = champs(etat_de(type=t, feu=True, r="R90"))
    vs = EN.variantes(t, "R90")
    chk(f"feu coché, {t.lower()} : la durée (R30 à R240) " + ("et les largeurs du tableau, libellées « b ≥ … → a = … »" if vs else "sans largeur"),
        menus.get("r") == D.R_LEVELS and (menus.get("b_min") == [str(b) for b, _ in vs] if vs else "b_min" not in menus)
        and all(f"b ≥ {b} mm → a = {a} mm" in h for b, a in vs) and "μ<sub>fi</sub> ≈ 0,7" in h, menus)

# =================================================================== 3. la saisie
print("\n=== 3. Saisie : effets de bord de l'écran, refus avec message ===")


def saisir(etat, chemin, valeur):
    e = copy.deepcopy(etat)
    return W.modifier(e, chemin, valeur), e


e0 = W.nouvel_etat()
m, e = saisir(e0, ["classe_expo"], "XC1")
chk("X0 → XC1 : la classe structurale passe à S2 (la proposée change, l'écran la reprend)", m is None and e["elements"][0]["classe_struct"] == "S2")
_, e = saisir(etat_de(classe_expo="XC2/XC3", classe_struct="S6"), ["classe_expo"], "XC4")
chk("XC2/XC3 (S6 choisie) → XC4 : S6 est gardée (même classe proposée S4)", e["elements"][0]["classe_struct"] == "S6")
_, e = saisir(etat_de(feu=True, r="R60", b_min=200), ["r"], "R90")
chk("poutre R60 b ≥ 200 → R90 : la largeur revient à la première variante (150)", e["elements"][0]["b_min"] == 150)
_, e = saisir(etat_de(type="Poteau", feu=True, r="R30", b_min=300), ["r"], "R60")
chk("poteau R30 b ≥ 300 → R60 : la liste change (tableau 5.2a, μ_fi = 0,7 : 200 · 300 puis 250 · 350), la largeur revient à la première (250)",
    e["elements"][0]["b_min"] == 250)
_, e = saisir(etat_de(type="Poteau", feu=True, r="R120", b_min=450), ["r"], "R240")
chk("poteau R120 → R240 : le tableau n'a pas de valeur, plus de largeur (null) ; R240 → R180 : la largeur du tableau revient (450)",
    e["elements"][0]["b_min"] is None and saisir(e, ["r"], "R180")[1]["elements"][0]["b_min"] == 450)
_, e = saisir(etat_de(feu=True, r="R60", b_min=200), ["r"], "R60")
chk("une durée re-saisie à l'identique garde la largeur choisie (même liste)", e["elements"][0]["b_min"] == 200)
_, e = saisir(etat_de(feu=True, r="R60", b_min=200), ["type"], "Dalle")
chk("poutre → dalle : plus de largeur (null)", e["elements"][0]["b_min"] is None)
_, e = saisir(etat_de(feu=True, r="R120"), ["feu"], False)
_, e = saisir(e, ["feu"], True)
chk("écart voulu : décocher puis recocher le feu garde la durée (R120 ; l'écran Streamlit revient à R60)", e["elements"][0]["r"] == "R120")
_, e = saisir(e0, ["diam_barre"], "25")
chk("« 25 » dans Ø barre : 25 mm, entier", e["elements"][0]["diam_barre"] == 25 and type(e["elements"][0]["diam_barre"]) is int)
_, e = saisir(e0, ["diam_barre"], "25,0")
chk("« 25,0 » est un entier : accepté", e["elements"][0]["diam_barre"] == 25)
REFUS = [(["diam_barre"], "4", "au moins 6 mm"), (["diam_barre"], "20,5", "entier"), (["diam_barre"], "abc", "illisible"),
         (["diam_etrier"], "-2", "au moins 0 mm"), (["delta_dev"], "7", "n'est pas proposé"), (["type"], "Mur", "n'est pas proposé"),
         (["classe_expo"], "XF1", "n'est pas proposé"), (["r"], "R45", "n'est pas proposé"), (["b_min"], "130", "n'est pas proposé"),
         (["zz"], "1", "Champ inconnu"), (["diam_barre", "x"], "1", "Champ inconnu"), (["position"], "Dehors", "n'est pas proposé"),
         # les saisies absurdes (relecture du 07/11/2026) : énormes, nulles, un texte pour la case du feu, un nom fleuve
         (["diam_barre"], "0", "au moins 6 mm"), (["diam_barre"], "61", "au plus 60 mm"), (["diam_barre"], "1e9", "au plus 60 mm"),
         (["diam_barre"], "1e400", "illisible"), (["diam_barre"], "nan", "illisible"), (["diam_etrier"], "41", "au plus 40 mm"),
         (["diam_etrier"], "1e6", "au plus 40 mm"), (["feu"], "false", "vrai ou faux"), (["feu"], "0", "vrai ou faux"),
         (["feu"], None, "vrai ou faux"), (["nom"], "x" * 81, "80 caractères au plus"), (["delta_dev"], True, "illisible")]
for chemin, v, motif in REFUS:
    m, e = saisir(e0, chemin, v)
    chk(f"refusé avec message : {'/'.join(chemin)} = « {v} » ({motif})", m is not None and motif in m and e == e0, m)
_, e = saisir(e0, ["diam_barre"], "60")
_, e = saisir(e, ["diam_etrier"], "40")
chk("les bornes elles-mêmes sont admises (Ø60, étrier Ø40) et se calculent", e["elements"][0]["diam_barre"] == 60
    and e["elements"][0]["diam_etrier"] == 40 and W.resultat(e)["resume"]["niveau"] == "i")
_, e = saisir(e0, ["feu"], True)
chk("la case du feu reçoit vrai (un booléen) : le feu est vérifié", e["elements"][0]["feu"] is True)
r = json.loads(R.traiter(json.dumps({"module": "enrobage", "action": "modifier", "etat": e0, "chemin": ["diam_barre"], "valeur": "5"})))
chk("par le routeur : la réponse est rendue, le message est porté, l'état reste celui d'avant",
    r["ok"] and r["message"] and r["etat"] == e0, r.get("message"))

# =================================================================== 4. éléments de la note
print("\n=== 4. Plusieurs éléments par note ===")
e = W.nouvel_etat()
W.gerer_elements(e, "ajouter")
W.modifier(e, ["type"], "Dalle")
W.modifier(e, ["classe_expo"], "XD3/XS3")
W.gerer_elements(e, "dupliquer")
chk("ajouter (E2), copier (« E2 (copie) »), l'ouvert est le dernier", [x["nom"] for x in e["elements"]] == ["E1", "E2", "E2 (copie)"] and e["actif"] == 2)
out = W.resultat(e)
chk("la liste montre chaque élément avec son enrobage retenu",
    all(f"{mm(M.calculer(x)['c_nom'])}{C.NBSP}mm" in out["elements"] for x in e["elements"]) and out["elements"].count('data-activer="') == 3)
chk("le résumé de la note nomme l'élément au plus grand enrobage (E2, XD3/S6 : 65 mm)",
    out["resume"]["gouvernant"].startswith("E2 · c_nom = 65 mm") and out["resume"]["niveau"] == "i", out["resume"])
W.gerer_elements(e, "supprimer", 0)
chk("supprimer le premier : deux restent, l'ouvert suit", len(e["elements"]) == 2 and e["actif"] == 1)
W.gerer_elements(e, "activer", 0)
chk("ouvrir le premier", e["actif"] == 0 and W.resultat(e)["donnees"].count('value="E2"') == 1)

# =================================================================== 5. le sélecteur de la largeur
print("\n=== 5. Sélecteur de la largeur au feu ===")
e = etat_de(feu=True, r="R90", b_min=150, classe_expo="XC4", classe_struct="S4")
out = W.resultat(e)
chk("la ligne feu d'une poutre ouvre le sélecteur (pk « feu ») ; l'étoile dit qu'une largeur plus grande abaisse l'enrobage",
    'data-pk="feu"' in out["resultat"] and "À partir de b ≥ 200 mm, l&#x27;enrobage retenu descend à 40 mm" in out["resultat"])
s = W.selecteur(e, "feu")
lignes = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)".*?<span class="kg">([^<]+)</span><span class="chip c">([^<]+)</span>', s)
cands = M.candidats_feu(e["elements"][0])
chk("quatre largeurs triées, a de chacune, ✓ la retenue (150), ★ l'optimale (200 : c_nom 40 = durabilité)",
    [x[2] for x in lignes] == [f"b ≥ {c['b_min']} mm" for c in cands] == ["b ≥ 150 mm", "b ≥ 200 mm", "b ≥ 300 mm", "b ≥ 400 mm"]
    and [bool(x[0]) for x in lignes] == [True, False, False, False] and [bool(x[1]) for x in lignes] == [False, True, False, False]
    and [x[3] for x in lignes] == ["a = 55", "a = 45", "a = 40", "a = 35"], lignes)
chk("le pourcentage est c_nom,feu / c_nom,dur (47/40, 37/40, 32/40, 27/40)",
    [texte(x[4]) for x in lignes] == ["118 %", "93 %", "80 %", "68 %"], [x[4] for x in lignes])
W.choisir_section(e, "feu", "b ≥ 300 mm")
chk("choisir « b ≥ 300 mm » : la largeur retenue est 300, le feu donne 32 mm, la durabilité gouverne (40)",
    e["elements"][0]["b_min"] == 300 and M.calculer(e["elements"][0])["c_nom"] == 40)
for role in ("x", "inf-0"):
    try:
        W.selecteur(e, role)
        refuse = False
    except ValueError:
        refuse = True
    chk(f"un sélecteur inconnu ({role}) est refusé", refuse)
chk("pas de sélecteur pour une dalle", 'data-pk=' not in W.resultat(etat_de(type="Dalle", feu=True))["resultat"])

# =================================================================== 6. dessins et tableaux
print("\n=== 6. Le dessin à l'échelle, les tableaux ===")
for k in ({}, {"feu": True, "r": "R120", "b_min": 200, "diam_barre": 25, "diam_etrier": 10},
          {"type": "Poteau", "diam_barre": 32, "diam_etrier": 12, "classe_expo": "XD3/XS3", "classe_struct": "S6", "delta_dev": 0}):
    e = etat_de(**k)
    res = M.calculer(e["elements"][0])
    el = res["element"]
    svg = W.resultat(e)["vues"][0]["svg"]
    c, de, d = res["c_nom"], el["diam_etrier"], el["diam_barre"]
    ch = re.search(r'<path d="M([\d.]+) 0 V[\d.]+ A[\d. ]+ ([\d.]+) ([\d.]+) H[\d.]+" fill="none" stroke="#33415C" stroke-width="([\d.]+)"', svg)
    cercles = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)" fill="#15181F"/>', svg)
    H = float(re.search(r'<rect x="0" y="0" width="[\d.]+" height="([\d.]+)"', svg).group(1))
    ok = (ch and abs(float(ch.group(1)) - (c + de / 2)) < 0.01 and abs(float(ch.group(4)) - de) < 0.01
          and abs(float(cercles[0][0]) - (c + de + d / 2)) < 0.01 and abs(H - float(cercles[0][1]) - (c + de + d / 2)) < 0.01
          and abs(float(cercles[0][2]) - d / 2) < 0.01)
    chk(f"détail d'angle {el['type'].lower()} à l'échelle : étrier Ø{de} à c_nom = {mm(c)} du parement, barre Ø{d} logée, a = {mm(c + de + d / 2)}",
        ok and f"= {mm(c)}</tspan>" in svg and f"= {mm(c + de + d / 2)}<" in svg, (ch and ch.groups(), cercles[:1], H))
    if res["feu"]:
        chk("la distance demandée au feu est repérée en rouge sur la cote a", f"≥ {mm(res['feu']['a'])} · {res['feu']['r']}" in svg)
for t in ("Dalle", "Voile"):
    e = etat_de(type=t, diam_barre=12, diam_etrier=8, feu=True, r="R60")
    res = M.calculer(e["elements"][0])
    svg = W.resultat(e)["vues"][0]["svg"]
    H = float(re.search(r'<rect x="0" y="0" width="[\d.]+" height="([\d.]+)"', svg).group(1))
    bande = re.search(r'<rect x="0" y="([\d.]+)" width="[\d.]+" height="([\d.]+)" fill="#33415C"/>', svg)
    chk(f"coupe {t.lower()} à l'échelle : barre extérieure Ø8 à c_nom = {mm(res['c_nom'])} du parement exposé",
        bande and abs(H - float(bande.group(1)) - res["c_nom"] - 8) < 0.01 and abs(float(bande.group(2)) - 8) < 0.01
        and "parement exposé" in svg and f" = {mm(res['c_nom'])} + 8 + 6 = {mm(res['c_nom'] + 14)}<" in svg
        and f">a = {mm(res['c_nom'] + 14)}<" in svg)
barres = []
for k in ({}, {"diam_barre": 6, "diam_etrier": 0, "delta_dev": 0}, {"diam_barre": 60, "diam_etrier": 40},
          {"classe_expo": "XC4", "classe_struct": "S4", "feu": True, "r": "R90", "b_min": 150}):
    e = etat_de(**k)
    res = M.calculer(e["elements"][0])
    svg = W.resultat(e)["vues"][0]["svg"]
    c, xc0 = res["c_nom"], res["c_nom"] + res["element"]["diam_etrier"] + res["element"]["diam_barre"] / 2
    u = float(re.search(r'<text x="[-\d.]+" y="[-\d.]+" text-anchor="\w+" font-size="([\d.]+)"', svg).group(1)) / 0.95
    # le texte horizontal « c_nom = … » sous le parement : son x et son ancrage
    t = re.search(r'<text x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)" font-size="[\d.]+" fill="#15181F" font-family="[^"]+">c<tspan[^>]*>nom</tspan>', svg)
    x, ancre = float(t.group(1)), t.group(3)
    larg = len(f"cnom = {mm(c)}") * 0.56 * u * 0.92
    gauche, droite = {"middle": (x - larg / 2, x + larg / 2), "end": (x - larg, x), "start": (x, x + larg)}[ancre]
    if gauche < xc0 < droite:
        barres.append((k, x, ancre, xc0))
chk("détail d'angle : le texte de c_nom n'est jamais barré par la ligne d'attache de l'axe (placé à gauche s'il ne tient pas)",
    not barres, barres)
e = etat_de(classe_expo="XC4", classe_struct="S3", feu=True, r="R120", b_min=240)
tab = W.resultat(e)["vues"][1]["svg"]
on = re.findall(r'<td class="(?:[^"]* )?on(?: [^"]*)?">(?:<small>[^<]*</small>)?([^<]+)</td>', tab)
chk("vue Tableaux : la case retenue de c_min,dur (S3 / XC4 = 25) et celle du feu (R120, b ≥ 240 : 60)", on == ["25", "60"], on)
chk("vue Tableaux : le tableau 4.4N complet (42 cases) et le tableau 5.5 (6 durées)",
    tab.count("<td") >= 42 + 24 and all(f"<th>{r}</th>" in tab for r in D.R_LEVELS))

# =================================================================== 7. sécurité et robustesse
print("\n=== 7. Échappement, notes abîmées ===")
PIEGE = '<img src=x onerror=alert(1)>'
e = W.nouvel_etat()
W.modifier(e, ["nom"], PIEGE)
W.modifier(e, ["feu"], True)
out = W.resultat(e)
fuites = [k for k in ("resultat", "volets", "donnees", "elements", "phrase") if "<img" in out[k]] + \
         [v["cle"] for v in out["vues"] if "<img" in v["svg"]] + (["sélecteur"] if "<img" in W.selecteur(e, "feu") else [])
chk("un nom piégé est échappé partout (fenêtres, liste, données, dessins, sélecteur)", not fuites and "&lt;img" in out["volets"], fuites)
for abime, motif in (([1, 2], "illisible"), ({"elements": "abîmé"}, "liste"), ({"elements": [{"type": "Mur"}]}, "type"),
                     ({"elements": [{"diam_barre": 2}]}, "diam_barre"), ({"elements": [{"zz": 1}]}, "inconnue"),
                     ({"elements": [{"diam_barre": 10 ** 400}]}, "démesuré"), ({"elements": [{"diam_barre": 1e300}]}, "au plus 60 mm"),
                     ({"elements": [{"diam_etrier": 500}]}, "au plus 40 mm"), ({"elements": [{"feu": "oui"}]}, "vrai ou faux"),
                     ({"elements": [{"nom": "n" * 200}]}, "80 caractères"), ({"elements": [None]}, "dict attendu")):
    r = json.loads(R.traiter(json.dumps({"module": "enrobage", "action": "calculer", "etat": abime})))
    chk(f"note abîmée {json.dumps(abime, ensure_ascii=False)[:40]} : refusée avec un message lisible ({motif})",
        not r["ok"] and motif in r["erreur"], r.get("erreur"))
for action, index in (("activer", "abc"), ("supprimer", [1]), ("activer", True)):
    e = W.normaliser({"elements": [EN.nouvel_element(), EN.nouvel_element()], "actif": 0})
    r = json.loads(R.traiter(json.dumps({"module": "enrobage", "action": action, "etat": e, "index": index})))
    chk(f"« {action} » avec un rang illisible ({index!r}) : refusé avec un message, sans exception", not r["ok"] and "rang" in r["erreur"], r)
e = W.normaliser({"elements": [EN.nouvel_element(), EN.nouvel_element()], "actif": 1e300})
chk("un « actif » démesuré est ramené au dernier élément", e["actif"] == 1)
e = W.normaliser({"elements": [EN.nouvel_element(), EN.nouvel_element()], "actif": 9})
chk("un « actif » hors de la liste est ramené au dernier élément", e["actif"] == 1)
chk("une note sans élément est une note neuve", W.normaliser({"version": 1}) == W.nouvel_etat())

# =================================================================== 8. bout à bout : l'écran Streamlit = le site
# Le site passe par le moteur extrait ; ici, on lit ce que l'écran Streamlit D'ORIGINE affiche (modules/enrobage.py rejoué
# avec un faux st) et ce que le SITE affiche (routeur.traiter → fragments HTML), et on les compare valeur par valeur, à
# l'arrondi d'affichage de l'écran près (:.0f, au pair — défaut D7 : le site affiche la valeur exacte).
print("\n=== 8. Bout à bout : l'écran Streamlit d'origine rejoué = ce que le site affiche ===")
try:
    import contextlib
    import logging
    logging.disable(logging.WARNING)
    from modules import enrobage as E_ST
    from modules import navigation as NAV
except Exception as ex:  # noqa: BLE001
    E_ST = None
    print(f"     (streamlit absent : comparaison bout à bout non jouée — {ex})")


class FauxSt:
    """Un « st » qui répond aux widgets avec les valeurs de l'élément et ENREGISTRE ce que l'écran affiche."""

    def __init__(self, rep):
        self.rep, self.ev = rep, []

    def markdown(self, body="", **_):
        self.ev.append(("md", str(body)))

    def caption(self, body="", **_):
        self.ev.append(("cap", str(body)))

    def columns(self, spec, **_):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def expander(self, *a, **k):
        return contextlib.nullcontext()

    def divider(self):
        pass

    def selectbox(self, label, options, index=0, format_func=str, **_):
        options = list(options)
        self.ev.append(("sel", label, [format_func(o) for o in options], index))
        return next((o for o in options if o == self.rep.get(label)), options[index])

    def radio(self, label, options, **_):
        return self.rep[label]

    def number_input(self, label, **_):
        return self.rep[label]

    def checkbox(self, label, **_):
        return self.rep[label]

    def select_slider(self, label, **_):
        return self.rep[label]


def ecran_streamlit(el):
    v = next((x for x in EN.variantes(el["type"], el["r"]) if x[0] == el["b_min"]), None)
    rep = {"Type d'élément": el["type"], "Position dans l'ouvrage": el["position"], "Ø max armatures principales [mm]": el["diam_barre"],
           "Ø étrier / cadre [mm]": el["diam_etrier"], "Tolérance d'exécution Δc_dev [mm]": el["delta_dev"],
           "Classe d'exposition": el["classe_expo"], "Classe structurale": el["classe_struct"], D.LIBELLE_FEU: el["feu"],
           "Durée de résistance requise": el["r"], "Largeur minimale de l'élément b [mm]": v}
    faux, blocs = FauxSt(rep), []
    vrais = (E_ST.st, E_ST._bloc, NAV.bouton_accueil)
    E_ST.st, E_ST._bloc, NAV.bouton_accueil = faux, (lambda l, r="", e="ok": blocs.append((l, r, e))), (lambda *a, **k: None)
    try:
        E_ST.show()
    finally:
        E_ST.st, E_ST._bloc, NAV.bouton_accueil = vrais
    return faux.ev, blocs


def comme_ecran(x, texte_ecran):
    """La valeur exacte du site, arrondie comme l'écran (:.0f), redonne le texte de l'écran."""
    return f"{x:.0f}" == texte_ecran


def bout_a_bout(el):
    ev, blocs = ecran_streamlit(el)
    out = json.loads(R.traiter(json.dumps({"module": "enrobage", "action": "calculer", "etat": {"version": 1, "actif": 0, "elements": [el]}})))
    if not out["ok"]:
        return [out["erreur"]]
    e, Rh, V = [], out["resultat"], out["volets"]
    b = {l: r for l, r, _ in blocs}
    md = [x[1] for x in ev if x[0] == "md"]
    st_dur = re.search(r"c_nom = (\d+) mm", b["A — Durabilité"]).group(1)
    st_ret = re.search(r"c_nom = (\d+) mm", b["C — Enrobage nominal retenu"]).group(1)
    st_cmd, st_cmb, st_cmin, st_dev, st_cnd = re.findall(r"\*\*(?:c_nom,durabilité = )?(\d+) mm\*\*", next(m for m in md if m.startswith("- c_min,dur")))
    gouv_st = re.search(r"gouvernée par la \*\*(\w+)\*\*", " ".join(md)).group(1)
    if not comme_ecran(nombre(re.search(r'class="chip xl en-c">([^<]+)<small>', Rh).group(1)), st_ret):
        e.append(f"c_nom retenu ≠ écran {st_ret}")
    if gouv_st not in texte(re.search(r'<div class="gov">(.*?)</div></div>', Rh, re.S).group(1)):
        e.append(f"gouvernant ≠ écran {gouv_st}")
    dec = [nombre(x) for x in re.findall(r"<b>([\d  ,]+) mm</b></div>", Rh)]
    for nom, x, st_v in (("c_min,b", dec[0], st_cmb), ("c_min,dur", dec[1], st_cmd), ("c_min", dec[4], st_cmin), ("Δc_dev", dec[5], st_dev)):
        if not comme_ecran(x, st_v):
            e.append(f"{nom} {x} ≠ écran {st_v}")
    l_dur = texte(re.search(r'<b>Durabilité</b>(.*?)</div>', Rh, re.S).group(1))
    if not (comme_ecran(nombre(l_dur.rsplit("·", 1)[1].replace("mm", "")), st_dur) and st_dur == st_cnd):
        e.append(f"ligne durabilité « {l_dur} » ≠ écran {st_dur}")
    fx = {texte(m.group(1)): texte(m.group(2)) for m in re.finditer(r'<div class="fx"><span class="s">(.*?)</span>.*?<span class="r">(.*?)</span></div>', V, re.S)}
    for sym, st_v in (("cmin,dur", st_cmd), ("cmin,b", st_cmb), ("cmin", st_cmin), ("cnom,dur", st_cnd), ("cnom", st_ret)):
        if not comme_ecran(nombre(fx[sym].replace("mm", "")), st_v):
            e.append(f"fenêtre {sym} {fx[sym]} ≠ écran {st_v}")
    if el["feu"]:
        st_cnf = re.search(r"c_nom = (\d+) mm", b[f"B — Feu ({el['r']})"]).group(1)
        md_f = next(m for m in md if m.startswith("- Distance à l'axe"))
        st_a = re.search(r"a = \*\*(\d+) mm\*\*", md_f).group(1)
        st_sub = re.search(r"= (\d+) − (\d+) − (\d+) = \*\*(\d+) mm\*\*", md_f).groups()
        for sym, st_v in (("a", st_a), ("cfeu,min", st_sub[3]), ("cnom,feu", st_cnf)):
            if not comme_ecran(nombre(fx[sym].replace("mm", "")), st_v):
                e.append(f"fenêtre feu {sym} {fx[sym]} ≠ écran {st_v}")
        if el["type"] in D.TYPES_A_ETRIER:
            sub = re.search(r"max\(([\d,]+) − ([\d,]+) − ([\d,]+) ; 0\)", texte(V)).groups()
            if not all(comme_ecran(nombre(x), y) for x, y in zip(sub, st_sub[:3])):
                e.append(f"substitution feu {sub} ≠ écran {st_sub[:3]}")
        else:
            # une dalle, un voile : le site n'écrit plus le terme de l'étrier (écart voulu D6) ; en parité l'étrier vaut 0
            sub = re.search(r"max\(([\d,]+) − ([\d,]+) ; 0\)", texte(V)).groups()
            if not all(comme_ecran(nombre(x), y) for x, y in zip(sub, st_sub[:2])) or st_sub[2] != "0":
                e.append(f"substitution feu {sub} ≠ écran {st_sub[:3]}")
        note = [x[1] for x in ev if x[0] == "cap"][-1]          # la caption de l'écran sous « B — Feu »
        if texte(C.symb_math(W._note({"note": note}))) not in texte(out["donnees"]) or texte(C.symb_math(W._note({"note": note}))) not in texte(V):
            e.append(f"caption feu absente : {note}")
        l_feu = texte(re.search(r'<b>Feu · R\d+</b>(.*?)</div>', Rh, re.S).group(1))
        if not comme_ecran(nombre(l_feu.rsplit("·", 1)[1].replace("mm", "")), st_cnf):
            e.append(f"ligne feu « {l_feu} » ≠ écran {st_cnf}")
        if any(l.startswith("Distance à l'axe couverte") for l, _, _ in blocs) != ("Distance à l'axe couverte par la durabilité" in texte(Rh)):
            e.append("encadré « couverte par la durabilité » ≠ écran")
    elif b.get("B — Feu") != "non vérifié" or "Feu non vérifié" not in texte(Rh):
        e.append("« feu non vérifié » ≠ écran")
    if any(l.startswith("Durabilité et feu proches") for l, _, _ in blocs) != ("Durabilité et feu proches" in texte(Rh)):
        e.append("encadré « proches » ≠ écran")
    sel = {x[1]: x[2] for x in ev if x[0] == "sel"}
    if not all(html.escape(o, quote=True) in out["donnees"] for o in sel["Classe d'exposition"]):
        e.append("options d'exposition ≠ écran")
    return e


if E_ST is not None:
    def el_(**k):
        return EN.normaliser({**EN.nouvel_element("E1"), "b_min": None, **k})
    BOUT = [("défaut (X0/S1, Ø20, feu non vérifié)", el_()),
            ("XC1 / S2, Ø12", el_(classe_expo="XC1", classe_struct="S2", diam_barre=12)),
            ("poutre R90 b ≥ 150, XC4 : le feu gouverne", el_(classe_expo="XC4", classe_struct="S4", feu=True, r="R90", b_min=150)),
            ("poutre R90 b ≥ 400 : la durabilité gouverne", el_(classe_expo="XC4", classe_struct="S4", feu=True, r="R90", b_min=400)),
            ("égalité 35 / 35", el_(classe_expo="XC4", classe_struct="S4", diam_etrier=0, feu=True, r="R60", b_min=160)),
            ("écart 2 mm pile (encadré « proches »)", el_(classe_expo="XC4", classe_struct="S4", diam_etrier=7, feu=True, r="R90", b_min=200)),
            ("écart 2,5 mm (pas d'encadré)", el_(classe_expo="XC4", classe_struct="S4", diam_barre=21, diam_etrier=7, feu=True, r="R90", b_min=200)),
            ("dalle R30 : c_feu,min nul pile", el_(type="Dalle", diam_etrier=0, feu=True, r="R30")),
            ("dalle R30 Ø25 : c_feu,min négatif borné", el_(type="Dalle", diam_barre=25, diam_etrier=0, feu=True, r="R30")),
            ("voile R120, XD3/XS3 S6, sans étrier", el_(type="Voile", classe_expo="XD3/XS3", classe_struct="S6", feu=True, r="R120", diam_barre=12, diam_etrier=0)),
            ("poutre R60 Ø25 étrier 8 : 19,5 affiché « 20 » par l'écran", el_(diam_barre=25, diam_etrier=8, feu=True, r="R60", b_min=120)),
            ("poutre R60 Ø25 étrier 9 : 18,5 affiché « 18 » par l'écran", el_(diam_barre=25, diam_etrier=9, feu=True, r="R60", b_min=120)),
            ("poutre R240 b ≥ 280, Δc_dev 0, Ø32 étrier 12, XD2/XS2", el_(feu=True, r="R240", b_min=280, delta_dev=0, diam_barre=32, diam_etrier=12,
                                                                        classe_expo="XD2/XS2", classe_struct="S5")),
            ("Δc_dev 5, Ø40 : c_min,b gouverne", el_(delta_dev=5, diam_barre=40, classe_expo="XC2/XC3", classe_struct="S4")),
            ("Ø6, étrier 0 : le plancher à égalité", el_(diam_barre=6, diam_etrier=0)),
            ("poutre R240 b ≥ 700, extérieur, XC4/S3", el_(feu=True, r="R240", b_min=700, position="Extérieur", classe_expo="XC4", classe_struct="S3")),
            ("dalle R180 Ø11 sans étrier : 59,5 (feu)", el_(type="Dalle", feu=True, r="R180", diam_barre=11, diam_etrier=0, classe_expo="XD1/XS1", classe_struct="S5")),
            ("poutre R30 b ≥ 80, Ø7 : 25,5 (feu)", el_(feu=True, r="R30", b_min=80, diam_barre=7, diam_etrier=6)),
            ("poutre R30 b ≥ 200, Ø60 étrier 40 (les bornes)", el_(feu=True, r="R30", b_min=200, diam_barre=60, diam_etrier=40,
                                                                 classe_expo="XC1", classe_struct="S6")),
            ("voile R120 Ø13 Δc_dev 5, sans étrier : 33,5 (feu)", el_(type="Voile", feu=True, r="R120", diam_barre=13, diam_etrier=0, delta_dev=5))]
    for nom, el in BOUT:
        ec = bout_a_bout(el)
        chk(f"bout à bout · {nom} : verdict, décomposition, lignes, fenêtres, encadrés = écran Streamlit", not ec, ec)
    # LES TROIS ÉCARTS VOULUS avec l'écran Streamlit depuis le 10/11/2026 (docs/site/LOT_Q_QUATRE_MODULES.md § 4) : sur ces
    # cas — et ceux-là seulement — le site n'affiche plus ce que l'écran affiche ; le bout à bout le CONSTATE (il rougit
    # sur eux), et la section 10 les tient contre la norme
    VOULUS = [("D1 · poteau R120 b ≥ 350, Ø25", el_(type="Poteau", diam_barre=25, feu=True, r="R120", b_min=350)),
              ("D1 · poteau R30 b ≥ 300, Ø60 étrier 40", el_(type="Poteau", feu=True, r="R30", b_min=300, diam_barre=60, diam_etrier=40, classe_expo="XC1", classe_struct="S6")),
              ("D2 · voile R240, XD3/XS3 S6, sans étrier", el_(type="Voile", classe_expo="XD3/XS3", classe_struct="S6", feu=True, r="R240", diam_barre=12, diam_etrier=0)),
              ("D6 · voile R60, Ø16, étrier 8", el_(type="Voile", diam_barre=16, feu=True, r="R60")),
              ("D6 · dalle R180 Ø11 étrier 5", el_(type="Dalle", feu=True, r="R180", diam_barre=11, diam_etrier=5, classe_expo="XD1/XS1", classe_struct="S5"))]
    for nom, el in VOULUS:
        try:
            ec = bout_a_bout(el)
        except (KeyError, StopIteration, AttributeError) as ex:     # l'écran n'a même plus la largeur ou le libellé du site
            ec = [f"{type(ex).__name__}"]
        chk(f"écart voulu avec l'écran Streamlit depuis le 10/11/2026 · {nom} : le site n'affiche plus ce que l'écran affiche (le bout à bout le voit)", bool(ec), ec)
    # le contrôle sait rougir : un site qui afficherait 1 mm de plus sur c_min,dur serait vu
    vrai = W._synthese
    W._synthese = lambda res: vrai({**res, "durabilite": {**res["durabilite"], "c_min_dur": res["durabilite"]["c_min_dur"] + 1}})
    try:
        ec = bout_a_bout(BOUT[2][1])
    finally:
        W._synthese = vrai
    chk("le contrôle bout à bout rougit si le site affiche c_min,dur faux d'1 mm", any("c_min,dur" in x for x in ec), ec)

# =================================================================== 9. ce que la relecture a ajouté
print("\n=== 9. Relecture : hypothèses du feu, règles, tolérance ===")
out = W.resultat(etat_de(feu=True, r="R90", b_min=150))
chk("poutre au feu : l'hypothèse du tableau (appuis simples, tab. 5.5) est dite dans les Données, μ_fi n'est pas dans les hypothèses",
    "Poutre sur appuis simples, b<sub>min</sub> = 150 mm (tab. 5.5)." in out["donnees"]
    and "μ<sub>fi</sub> ≈ 0,7</span>" not in out["volets"])
out = W.resultat(etat_de(type="Voile", feu=True, r="R90"))
chk("voile au feu : « exposé sur 1 face, μ_fi ≈ 0,7 » dans les Données et μ_fi dans les hypothèses",
    "Voile porteur exposé sur 1 face, μ<sub>fi</sub> ≈ 0,7 (tab. 5.4)." in out["donnees"] and "μ<sub>fi</sub> ≈ 0,7</span>" in out["volets"])
regles = re.findall(r"<li><b>(.*?)</b>", out["volets"].split('id="ck-feu"')[0])
chk("fenêtre durabilité : chaque règle a son symbole, aucun en double (le plancher n'est plus un second « c_min »)",
    len(regles) == len(set(regles)) and "c<sub>min</sub> ≥ 10 mm" in regles, regles)
h = W.resultat(etat_de())["donnees"]
chk("Δc_dev est dans la section « Armatures et tolérance » (il s'ajoute aux deux enrobages), plus dans « Durabilité »",
    h.index('data-chemin="delta_dev"') < h.index('data-sec="du"') and "Armatures et tolérance" in h)
chk("les champs Ø portent leurs bornes (data-min, data-max) : le navigateur refuse avant d'envoyer",
    'data-chemin="diam_barre" data-sorte="nombre" inputmode="decimal" value="20" data-min="6" data-max="60"' in h
    and 'data-min="0" data-max="40"' in h)

# =================================================================== 10. depuis le 10/11/2026 : le bandeau, les trois écarts voulus
print("\n=== 10. Lot Q (10/11/2026) : le bandeau vers la bibliothèque, les trois écarts voulus tenus contre la norme ===")
from beton.bibliotheque import feu as FEU  # noqa: E402
out = W.resultat(etat_de())
bandeau = re.search(r'<div class="pbody"><div class="msg">(.*?)</div>', out["resultat"], re.S)
chk("en tête du résultat, un bandeau non grave : « Le calcul d'enrobage est maintenant dans la Bibliothèque › Enrobages › Calcul », avec un lien "
    "ordinaire et RELATIF vers bibliotheque-beton?rubrique=enrobages&onglet=calcul",
    bandeau is not None and "Le calcul d'enrobage est maintenant dans la Bibliothèque › Enrobages › Calcul" in texte(bandeau.group(1))
    and '<a href="bibliotheque-beton?rubrique=enrobages&amp;onglet=calcul">' in bandeau.group(1)
    and not re.search(r'href="(?:https?:|/|[^"]*\.html)', bandeau.group(1)), bandeau and bandeau.group(1))
chk("le bandeau est sur chaque élément, feu vérifié ou non, et n'a ni script ni gestionnaire",
    all(W.BANDEAU in W.resultat(etat_de(**k))["resultat"] for k in ({}, {"type": "Poteau", "feu": True, "r": "R240"}, {"type": "Dalle", "feu": True}))
    and not re.search(r"<script|\son[a-z]+\s*=", W.BANDEAU, re.I))
# D1 — les poteaux : le tableau 5.2a de l'EN 1992-1-2:2004, colonne μ_fi = 0,7 (transcrite ici du § 4.5 de la conception)
POTEAU_NORME = {"R30": [(200, 32), (300, 27)], "R60": [(250, 46), (350, 40)], "R90": [(350, 53), (450, 40)],
                "R120": [(350, 57), (450, 51)], "R180": [(450, 70)], "R240": []}
faux = []
for r_, vs in POTEAU_NORME.items():
    for b_, a_ in vs:
        o = W.resultat(etat_de(type="Poteau", feu=True, r=r_, b_min=b_, classe_expo="XC1", classe_struct="S2"))
        cfm = max(a_ - 10 - 8, 0)                       # Ø20, étrier Ø8 : c_feu,min = a − Ø/2 − Ø_étrier
        cn = max(30, cfm + 10)                          # durabilité XC1/S2, Ø20 : 20 + 10 = 30 ; feu : c_feu,min + Δc_dev
        if (f"b ≥ {b_} · a {a_} · {mm(cfm + 10)} mm" not in texte(o["resultat"]) or f'class="chip xl en-c">{mm(cn)}<small>' not in o["resultat"]
                or f"b ≥ {b_} mm → a = {a_} mm" not in o["donnees"] or f"max({a_} − 10 − 8 ; 0)" not in texte(o["volets"])):
            faux.append((r_, b_, a_))
chk("D1 · poteaux : chaque couple b_min / a du tableau 5.2a à μ_fi = 0,7 (R90 : 350/53 et 450/40, là où l'écran donnait 200/31) est dans le menu, "
    "la ligne du feu, la fenêtre de calcul, et le verdict en découle (9 couples)", not faux and sum(len(v) for v in POTEAU_NORME.values()) == 9, faux)
o = W.resultat(etat_de(type="Poteau", feu=True, r="R240", classe_expo="XC4", classe_struct="S4"))
chk("D1 · poteau R240 : « non couvert par le tableau » — plus de valeur inventée : pas de menu de largeur, la ligne du feu sans nombre, le verdict = "
    "durabilité (40 mm), l'avertissement GRAVE, la fenêtre « aucune valeur », aucun « feu non dimensionnant »",
    'data-chemin="b_min"' not in o["donnees"] and "non couvert par le tableau" in texte(o["resultat"]) and 'class="chip xl en-c">40<small>' in o["resultat"]
    and re.search(r'<div class="msg grave">Poteau R240 : non couvert par le tableau 5\.2a', o["resultat"]) is not None
    and "aucune valeur" in texte(o["volets"]) and "feu non dimensionnant" not in texte(o["resultat"]) and 'data-pk="feu"' not in o["resultat"]
    and "c<sub>nom,feu</sub>" not in o["volets"].split('id="ck-retenu"')[1].split("ck-sub")[0])
tab = W.resultat(etat_de(type="Poteau", feu=True, r="R90", b_min=350))["vues"][1]["svg"]
chk("D1 · vue Tableaux d'un poteau : le tableau 5.2a de l'EN (μ_fi = 0,7), la case 350/53 retenue, la ligne R240 « non couvert par le tableau »",
    re.findall(r'<td class="on"><small>b ≥ (\d+)</small>(\d+)</td>', tab) == [("350", "53")] and "tableau 5.2a de l'EN 1992-1-2:2004" in tab
    and re.search(r'<tr class=""><th>R240</th><td colspan="2" class="vide">non couvert par le tableau</td></tr>', tab) is not None)
o = W.resultat(etat_de(type="Poteau", feu=True, r="R90", b_min=450))
chk("D1 · la source est dite à l'écran (tableau 5.2a, colonne μ_fi = 0,7) et « au moins 8 barres » pour 450/40 ; plus aucun message rouge « à vérifier dans la norme »",
    "tableau 5.2a" in texte(o["resultat"]) and "au moins 8 barres" in texte(o["resultat"]) and 'class="msg grave"' not in o["resultat"]
    and "à vérifier dans la norme" not in o["resultat"] and "EN 1992-1-2:2004, tableau 5.2a" in texte(o["volets"]))
# D2 — les voiles : tableau 5.4, μ_fi = 0,7, une face
VOILE_NORME = {"R30": 10, "R60": 10, "R90": 25, "R120": 35, "R180": 50, "R240": 60}
faux = [(r_, a_) for r_, a_ in VOILE_NORME.items()
        for o in (W.resultat(etat_de(type="Voile", feu=True, r=r_, diam_barre=12, diam_etrier=0, classe_expo="XC1", classe_struct="S2")),)
        if f"a {a_} · {mm(max(a_ - 6, 0) + 10)} mm" not in texte(o["resultat"]) or f"max({a_} − 6 ; 0)" not in texte(o["volets"])]
chk("D2 · voiles : a = 10, 10, 25, 35, 50, 60 mm de R30 à R240 (l'écran : 45 et 55 pour R180 et R240) — la ligne du feu et la fenêtre de calcul",
    not faux and FEU.VOILE_5_4["R180"]["mu07_1"] == (210, 50, ""), faux)
# D6 — une dalle, un voile : aucun étrier retiré de la distance à l'axe
o8 = W.resultat(etat_de(type="Dalle", feu=True, r="R90", diam_barre=12, diam_etrier=8))
o0 = W.resultat(etat_de(type="Dalle", feu=True, r="R90", diam_barre=12, diam_etrier=0))
chk("D6 · dalle R90, Ø12 : c_feu,min = 30 − 6 = 24 mm que l'étrier saisi vaille 8 ou 0 (l'écran retirait les 8 mm : 16) ; la fenêtre écrit « max(a − Ø/2 ; 0) », "
    "et un message (non grave) dit que l'étrier n'est pas retiré",
    all("a 30 · 34 mm" in texte(o["resultat"]) and "max(30 − 6 ; 0)" in texte(o["volets"]) and "max(a − Ø/2 ; 0)" in texte(o["volets"]) for o in (o8, o0))
    and "n'est pas retiré de la distance à l'axe" in texte(o8["resultat"]) and "n'est pas retiré" not in texte(o0["resultat"])
    and 'class="msg grave"' not in o8["resultat"])
op = W.resultat(etat_de(feu=True, r="R90", b_min=200, diam_barre=12, diam_etrier=8))
chk("… une poutre garde son étrier : c_feu,min = 45 − 6 − 8 = 31 mm, « max(a − Ø/2 − Ø étrier ; 0) »",
    "max(45 − 6 − 8 ; 0)" in texte(op["volets"]) and "max(a − Ø/2 − Ø étrier ; 0)" in texte(op["volets"]))
# mutations : chaque écart voulu REVENU fait rougir son contrôle
vrai = D.FEU_POTEAU
D.FEU_POTEAU = {**vrai, "R90": [(200, 31), (300, 25), (400, 25)]}
try:
    o = W.resultat(etat_de(type="Poteau", feu=True, r="R90", b_min=200))
    chk("mutation « D1 revenu : la table de l'écran pour les poteaux » : 350/53 n'est plus dans le menu (le contrôle D1 rougit)",
        "b ≥ 350 mm → a = 53 mm" not in o["donnees"] and "b ≥ 200 mm → a = 31 mm" in o["donnees"])
finally:
    D.FEU_POTEAU = vrai
vrai = D.TYPES_A_ETRIER
D.TYPES_A_ETRIER = ("Poutre", "Poteau", "Dalle", "Voile")
try:
    chk("mutation « D6 revenu : l'étrier retiré d'une dalle » : c_feu,min repasse à 16 mm (le contrôle D6 rougit)",
        "a 30 · 26 mm" in texte(W.resultat(etat_de(type="Dalle", feu=True, r="R90", diam_barre=12, diam_etrier=8))["resultat"]))
finally:
    D.TYPES_A_ETRIER = vrai
vrai = W.BANDEAU
W.BANDEAU = ""
try:
    chk("mutation « bandeau retiré » : le contrôle du bandeau rougit", 'href="bibliotheque-beton' not in W.resultat(etat_de())["resultat"])
finally:
    W.BANDEAU = vrai

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
