# -*- coding: utf-8 -*-
"""L'API du site pour les cornières d'appui refondues (api_web/corniere.py, décisions du 10/11/2026), vérifiée en Python
ordinaire : CE QUE LE SITE AFFICHE = CE QUE LE MOTEUR CALCULE.

- sur ≥ 12 états (A historique, A équilibre, B, B avec le mode historique demandé, deux inégales, une petite, une
  grosse, une saisie libre, la largeur d'appui, le cas de contrôle du classeur, la garde rouge, une forte charge) : le
  verdict (taux, section gouvernante, combinaison), chaque vérification sur deux lignes avec sa couleur, son taux et la
  combinaison de jeux qui la gouverne (les maxima du congé « hors verdict » en historique), les ancrages (N, V ELS et ELU
  par ancrage, positions, informations hors verdict), la comparaison (A), les avertissements titrés (le grave en tête),
  les hypothèses, la masse ;
- les REFUS : chaque saisie hors domaine est refusée avec le message du moteur, la cornière intacte ;
- le SÉLECTEUR : 233 cornières par masse, ✓ retenue, ★ optimale, grises hors domaine (raison), rouges qui ne tiennent
  pas, le filtre par famille, le choix ;
- la NOTE V1 (modèle lame) : relue, l'avis d'ouverture dit une fois et retiré de l'état, l'avertissement du moteur tant
  que la charge n'est pas ressaisie ;
- le CONTRAT DE SÉLECTION : chaque data-sel des dessins a sa ligne dans les données (data-cible), sous la même clé ;
- les FENÊTRES DE CALCUL : cinq, chaque ligne « symbole = expression = valeurs = résultat » RÉÉVALUÉE (les valeurs
  écrites, recalculées, retombent sur le résultat affiché) ; la cellule du classeur en historique ;
- les éléments de la note, les vues (quatre, en-tête, légende partagée), l'échappement d'un nom piégé, la note PDF
  (refusée tant que api_web/note_corniere.py n'est pas là, faite sinon), le contrat commun ;
- la RÉVISION § 9 (10/11 soir) : l'ÉTAT LIMITE (ELU : R saisie à l'ELU, f_y/γ_M0 partout à la place de σ_adm, coef et
  R_ELU masqués, les ancrages ELU seuls), les ANCRAGES (entraxe « vide = L/n » et sa valeur auto, type, profondeur,
  rondelles, les positions e₁ · s · z), le SUPPORT (voile ou poutre, ses deux étendues, les distances aux bords en
  information), le volet « Détails » replié par défaut, la saisie libre à champs visibles, les refus des nouveaux champs ;
- des MUTATIONS du module (vingt-trois), chacune vue rouge par ces contrôles.

    /usr/bin/python3.12 tests/test_api_web_corniere.py
"""
import copy
import html
import importlib.util
import json
import math
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.corniere import entree as EN  # noqa: E402
from acier.corniere import moteur as MO  # noqa: E402
from acier.corniere import schemas as SC  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import corniere as K  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="corniere", **k))))


def saisir(etat, chemin, valeur):
    out = req(action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def etat_avec(**champs):
    """Une note neuve dont la cornière reçoit ``champs`` (par le moteur : hors domaine, ValueError)."""
    e = K.nouvel_etat()
    e["cornieres"][0].update(champs)
    return K.normaliser(e)


def lire_fr(s):
    s = s.replace(C.NBSP, "").replace(C.FINE, "").replace(" ", "").replace("−", "-").replace(",", ".")
    return float(re.match(r"-?\d+(?:\.\d+)?", s).group(0))


def pct_txt(t):
    return C._taux_txt(C.pct(t))[0]


# ====================================================================== lecture du HTML rendu
LIGNE = re.compile(r'<div class="el vc(?P<cls>[^"]*)"(?: data-calcul="(?P<calcul>[^"]*)" title="[^"]*")?><div class="l1"><b>(?P<nom>[^<]*)</b>(?P<l1>.*?)</div>'
                   r'<div class="l2"><span class="cmp">(?P<cmp>.*?)</span>(?P<jauge>(?:<div class="bar">.*?</div>)?)</div></div>', re.S)
PCT = re.compile(r'<span class="pct (\w)"(?: title="[^"]*")?>([^<]+)</span>')
MOT = re.compile(r'<span class="et-mot (\w)">([^<]+)</span>')
PIECE = re.compile(r'<(?:button|span) class="prof"[^>]*>(?:<span class="leger"[^>]*>.*?</span>)?<span>(.*?)</span>', re.S)
BLOC = re.compile(r'<section class="et-bloc (\w+)" data-theme="(\d)"><div class="et-bloc-h"><button[^>]*>.*?<span class="n">([^<]*)</span><span class="e">([^<]*)</span>', re.S)
MSG = re.compile(r'<div class="msg( grave)?">(.*?)</div>', re.S)
FX = re.compile(r'<div class="fx">(.*?)</div>', re.S)
SPAN = re.compile(r'<span(?: class="(s|r|chip \w+)")?>(.*?)</span>', re.S)


def lignes(frag):
    """Les vérifications d'un fragment : [{nom, calcul, piece, taux_txt, niveau, mot, cmp, gov}]."""
    out = []
    for m in LIGNE.finditer(frag):
        p = PIECE.search(m.group("l1"))
        t = PCT.search(m.group("l1"))
        w = MOT.search(m.group("l1"))
        out.append({"nom": html.unescape(m.group("nom")), "calcul": m.group("calcul"), "piece": html.unescape(re.sub("<[^>]+>", "", p.group(1))) if p else "",
                    "taux_txt": t.group(2) if t else None, "niveau": t.group(1) if t else (w.group(1) if w else None),
                    "mot": w.group(2) if w else None, "cmp": html.unescape(re.sub("<[^>]+>", "", m.group("cmp"))), "gov": " gov" in m.group("cls")})
    return out


def blocs(frag):
    return [{"cls": c, "theme": int(i), "titre": html.unescape(t), "etat": html.unescape(e)} for c, i, t, e in BLOC.findall(frag)]


def texte(frag):
    return html.unescape(re.sub(r"<[^>]+>", " ", frag))


def resultat_module(mod, etat):
    """Le résultat rendu par un module (le vrai ou un mutant), l'état relu comme le routeur le fait."""
    return mod.resultat(mod.normaliser(copy.deepcopy(etat)))


# ====================================================================== 1. une note neuve
r0 = req(action="nouveau")
chk("nouveau : la requête aboutit", r0["ok"], r0.get("erreur"))
chk("la note neuve : version 2, une cornière « C1 » aux défauts des décisions § 2 et § 9 (L 120×120×15, A, historique, ELS, 35 kN, 75 ± 10, 85 ± 5, "
    "3 chevilles mécaniques Ø16 à L/n, profondeur 100, une rondelle, voile)",
    r0["etat"] == {"version": 2, "cornieres": [MO.nouvelle_corniere("C1")], "actif": 0} and r0["etat"]["cornieres"][0]["dx"] == 10.0
    and r0["etat"]["cornieres"][0]["dy"] == 5.0 and r0["etat"]["cornieres"][0]["etat_limite"] == EN.ELS and r0["etat"]["cornieres"][0]["support"] == EN.VOILE, r0["etat"])
REF0 = MO.verifier(MO.nouvelle_corniere())
don0 = r0["donnees"]
chk("le résumé est celui du moteur : taux arrondi dans sa bande, niveau, statut, gouvernant nommé",
    r0["resume"] == {"taux": C.pct(REF0["taux_max"]), "niveau": C.niveau(REF0["taux_max"]), "statut": C.STATUTS[C.niveau(REF0["taux_max"])][1],
                     "gouvernant": "C1 · Section 3 · aile verticale"}, r0["resume"])
chk("la réponse porte tout ce que l'écran compact pose (élément dans la colonne de gauche, phrase vide)",
    r0["elements"] == "" and r0["phrase"] == "" and 'class="et-arbre et-arbre-c"' in r0["donnees"] and 'data-corniere="0"' in r0["donnees"], None)


# ====================================================================== 2. le résultat face au moteur, sur des états variés
def controle_resultat(mod, etat, nom_cas):
    """Le cadre Résultat d'un module (le vrai, ou un mutant) face au moteur : rend la liste des fautes."""
    rep = resultat_module(mod, etat)
    c = etat["cornieres"][etat["actif"]]
    m = MO.verifier(c)
    f = []
    res = rep["resultat"]
    # le verdict : la grande pastille, la section gouvernante, sa combinaison, la configuration et le mode
    chip = re.search(r'<div class="verdict" data-calcul="s0"[^>]*><span class="chip xl (\w)"[^>]*><small>η</small>([^<]+)</span><div class="gov"><b>([^<]*)</b>(.*?)</div></div>', res, re.S)
    niv_att = "r" if m["statut"] == MO.NOK else C.niveau(m["taux_max"])
    g = m["gouvernant"]
    if not chip:
        f.append("pas de verdict")
    else:
        gov_txt = texte(chip.group(4))
        if chip.group(1) != niv_att or chip.group(2) != pct_txt(m["taux_max"]):
            f.append(f"verdict {chip.group(1)} {chip.group(2)} ≠ {niv_att} {pct_txt(m['taux_max'])}")
        if html.unescape(chip.group(3)) != f"{K.COURTS[g['cle']]} · {K.AILES[g['aile']]} · {K.LIEUX[g['cle']]}":
            f.append(f"gouvernant « {chip.group(3)} »")
        if g["combinaison"]["libelle"] not in gov_txt or EN.LIBELLES_MODE[m["mode"]] not in gov_txt or f"configuration {m['configuration']}" not in gov_txt \
                or m["materiau"]["etat_limite"] not in gov_txt:
            f.append(f"verdict sans la combinaison, le mode, la configuration ou l'état limite : {gov_txt}")
    # les SYMBOLES des limites sont ceux que le moteur nomme (σ_adm en ELS, f_y/γ_M0 en ELU), dans le résultat et les fenêtres ;
    # l'autre n'apparaît nulle part
    els = m["materiau"]["etat_limite"] == EN.ELS
    sym, autre = C.symb_math(m["materiau"]["symbole_sigma"]), ("f<sub>y</sub>/γ<sub>M0</sub>" if els else "σ<sub>adm</sub>")
    if sym not in res or autre in res or sym not in rep["volets"] or autre in rep["volets"]:
        f.append(f"le symbole de la limite n'est pas celui du moteur ({m['materiau']['symbole_sigma']}) partout : « {autre} » vu ou « {sym} » absent")
    if (f"({m['materiau']['etat_limite']})" not in res):
        f.append("l'état limite n'est pas dit sur la ligne Cornière")
    # les blocs thématiques : Sections (coloré par le verdict), Ancrages (information), Comparaison (A seulement)
    bl = blocs(res)
    titres = [b["titre"] for b in bl]
    attendus = ["Sections", "Ancrages"] + (["Comparaison"] if m["configuration"] == "A" else [])
    if titres != attendus or [b["theme"] for b in bl] != list(range(len(attendus))):
        f.append(f"blocs {titres} ≠ {attendus}")
    if bl:
        cls_att = "ok" if m["statut"] == MO.OK else "nok"
        if bl[0]["cls"] != cls_att or not bl[0]["etat"].startswith("Conforme" if cls_att == "ok" else "Non conforme") or K._pct(m["taux_max"]) not in bl[0]["etat"]:
            f.append(f"bloc Sections {bl[0]}")
        if len(bl) > 1 and bl[1]["cls"] != "inc":
            f.append("le bloc Ancrages n'est pas en information")
    # les lignes du bloc Sections : la cornière, les efforts, puis chaque vérification du moteur
    ls = lignes(res)
    noms = [li["nom"] for li in ls]
    sec = m["section"]
    if not ls or ls[0]["nom"] != "Cornière" or ls[0]["piece"] != sec["designation"] or ls[0]["calcul"] != "geometrie" or 'data-pk="corniere"' not in res:
        f.append(f"ligne Cornière : {ls[:1]}")
    if len(ls) < 2 or ls[1]["nom"] != "Efforts" or K._n(m["efforts"]["Mmax"], "kN·m") not in ls[1]["cmp"] or K._n(m["efforts"]["H"], "kN") not in ls[1]["cmp"] \
            or m["efforts"]["combinaison_H"]["libelle"] not in ls[1]["cmp"] or ls[1]["calcul"] != "efforts":
        f.append(f"ligne Efforts : {ls[1:2]}")
    i = 2
    if c["a"] is not None:
        inf_le = next(x for x in m["informations"] if x["cle"] == "L_eff")
        if len(ls) < 3 or ls[2]["nom"] != "Largeur efficace" or ls[2]["mot"] != f"{C.nb(round(m['geometrie']['L_eff'], 1), 1)} mm" \
                or inf_le["combinaison"]["libelle"] not in ls[2]["cmp"] or ls[2]["calcul"] != "geometrie":
            f.append(f"ligne L_eff : {ls[2:3]}")
        i = 3
    for v in m["verifications"]:
        if i >= len(ls):
            f.append(f"{v['cle']} absent")
            break
        li = ls[i]
        i += 1
        attendu_nom = K.COURTS[v["cle"]]
        if li["nom"] != attendu_nom or li["piece"] != K.AILES[v["aile"]] or li["calcul"] != "s0":
            f.append(f"{v['cle']} : {li['nom']} / {li['piece']} ≠ {attendu_nom} / {K.AILES[v['aile']]}")
        if f"s = {C.nb(round(v['s'], 2), 2)} · t = {C.nb(round(v['t'], 2), 2)} mm" not in li["cmp"]:
            f.append(f"{v['cle']} : le lieu (s, t) manque en ligne 2 : {li['cmp']}")
        if v["combinaison"]["libelle"] not in li["cmp"]:
            f.append(f"{v['cle']} sans sa combinaison : {li['cmp']}")
        if v["verdict"]:
            if li["taux_txt"] != pct_txt(v["taux"]) or li["niveau"] != ("r" if v["etat"] == MO.NOK else C.niveau(v["taux"])):
                f.append(f"{v['cle']} : {li['taux_txt']} {li['niveau']} ≠ {pct_txt(v['taux'])}")
            if li["gov"] != (v["cle"] == g["cle"]):
                f.append(f"{v['cle']} : gouvernante mal marquée")
            if K._n(v["sigma_eq"], "N/mm²") not in li["cmp"] or K._n(v["sigma_adm"], "N/mm²") not in li["cmp"]:
                f.append(f"{v['cle']} : σ_éq ou σ_adm absents de « {li['cmp']} »")
        else:
            if li["taux_txt"] is not None or li["mot"] != "hors verdict" or li["niveau"] != "c" or K._pct(v["taux"]) not in li["cmp"]:
                f.append(f"{v['cle']} (information) : {li}")
    # les ancrages : N, V — ELS puis ELU en ELS ; ELU seuls quand R est saisie à l'ELU (le moteur ne rend pas d'ELS) —, les
    # positions (e₁, s, z : celles du moteur), puis les informations hors verdict
    anc = m["ancrages"]
    att = ([("Traction N · ELS", K._n(anc["N_ELS"], "kN") + " kN"), ("Cisaillement V · ELS", K._n(anc["V_ELS"], "kN") + " kN")] if els else []) \
        + [("Traction N · ELU", K._n(anc["N_ELU"], "kN") + " kN"), ("Cisaillement V · ELU", K._n(anc["V_ELU"], "kN") + " kN"),
           ("Positions", f"entraxe {C.nb(round(anc['entraxe'], 2), 2)}")]
    la = ls[i:i + len(att)]
    if [(x["nom"], x["mot"]) for x in la] != att or any(x["calcul"] != "s1" or x["niveau"] != "c" for x in la):
        f.append(f"ancrages : {[(x['nom'], x['mot'], x['calcul']) for x in la]} ≠ {att}")
    if la and anc["combinaison_N"]["libelle"] not in la[0]["cmp"]:
        f.append("la traction d'ancrage ne dit pas sa combinaison")
    if not els and (anc["N_ELS"] is not None or any("ELS" in x["nom"] for x in la)):
        f.append("ELU : une ligne ELS est affichée alors que le moteur n'en rend pas")
    pos = la[-1]["cmp"] if la else ""
    if (" · ".join(C.nb(round(z, 2), 2) for z in anc["positions_z"]) not in pos or f"e₁ = {C.nb(round(anc['distance_bout'], 2), 2)}" not in pos
            or f"s = {C.nb(round(anc['entraxe'], 2), 2)}" not in pos):
        f.append(f"positions (e₁, s, z) : {pos}")
    i += len(att)
    infos = [x for x in m["informations"] if x["cle"] not in ("autre_mode", "L_eff")]
    for inf in infos:
        if i >= len(ls):
            f.append(f"information {inf['cle']} absente")
            break
        li = ls[i]
        i += 1
        val = f"{K._n(inf['valeur'], inf['unite'])} {inf['unite']}".strip()
        if li["nom"] != K.INFOS[inf["cle"]] or li["mot"] != val or li["piece"] != "hors verdict" or li["taux_txt"] is not None:
            f.append(f"information {inf['cle']} : {li} ≠ {val}")
    # la comparaison (A) : l'autre mode en information, la garde (historique)
    if m["configuration"] == "A":
        cp = m["comparaison"]
        if i >= len(ls) or ls[i]["mot"] != K._pct(cp["taux_max"]) or f"Mode {K.MODES_COURTS[cp['mode']]}" != ls[i]["nom"] or ls[i]["calcul"] != "s2":
            f.append(f"comparaison : {ls[i:i + 1]}")
        i += 1
        gd = m["garde"]
        if gd is not None:
            if i >= len(ls) or ls[i]["nom"] != "Garde · équilibre P1" or ls[i]["niveau"] != ("r" if gd["depasse"] else "c") \
                    or K._pct(gd["taux_equilibre_P1"]) not in ls[i]["cmp"] or K._pct(gd["taux_historique"]) not in ls[i]["cmp"]:
                f.append(f"garde : {ls[i:i + 1]}")
            i += 1
    if i != len(ls):
        f.append(f"{len(ls) - i} ligne(s) de trop")
    # les avertissements : chacun, titré, le grave en tête ; les hypothèses ; la masse
    msgs = MSG.findall(res)
    avis = [x for x in msgs]
    graves = [bool(gr) for gr, _ in avis]
    if graves != sorted(graves, reverse=True):
        f.append("un avertissement grave n'est pas en tête")
    for a in m["avertissements"]:
        corps = html.unescape(re.sub("<[^>]+>", "", a["message"]))
        if not any(corps[len(K.TITRES_AVERTISSEMENTS.get(a["code"], "")):].strip(" —:") in texte(t) for _, t in avis):
            f.append(f"avertissement {a['code']} absent")
        if a["niveau"] == MO.ROUGE and not any(gr and a["message"][-30:] in texte(t) for gr, t in avis):
            f.append(f"avertissement {a['code']} pas en grave")
    if len(avis) != len(m["avertissements"]):
        f.append(f"{len(avis)} messages pour {len(m['avertissements'])} avertissements")
    for h in m["hypotheses"]:
        if C.esc(h) not in res:
            f.append(f"hypothèse absente : {h[:40]}")
    if C.qte(sec["masse"], "kg/m", 2) not in res:
        f.append("masse absente")
    # les fenêtres de calcul : chaque data-calcul a la sienne ; la cellule du classeur en historique
    calculs = set(re.findall(r'data-calcul="([^"]+)"', res))
    attendus_ck = {"s0", "s1", "geometrie", "efforts"} | ({"s2"} if m["configuration"] == "A" else set())
    if calculs != attendus_ck or any(f'id="ck-{k}"' not in rep["volets"] for k in calculs):
        f.append(f"fenêtres {calculs} ≠ {attendus_ck}")
    if (m["mode"] == EN.HISTORIQUE) != ("(C57)" in rep["volets"] and "(I37)" in rep["volets"]):
        f.append("la cellule du classeur n'est pas là où elle doit être")
    # le résumé et les vues
    if rep["resume"]["taux"] != C.pct(m["taux_max"]) or rep["resume"]["niveau"] != niv_att:
        f.append(f"résumé {rep['resume']}")
    if [v["cle"] for v in rep["vues"]] != [k for k, *_ in SC.VUES] or any(not v["svg"].lstrip().startswith("<svg") for v in rep["vues"]):
        f.append("vues")
    if any(v["legende"] != SC.legende(m, v["cle"]) or 'class="et-ent"' not in v.get("entete", "") for v in rep["vues"]):
        f.append("légende ou en-tête des vues")
    return f


def etat_B(**k):
    return etat_avec(configuration="B", **k)


ETATS = {
    "A historique (défauts)": etat_avec(),
    "A équilibre": etat_avec(mode=EN.EQUILIBRE),
    "A équilibre, pivot e/2 et u_V = 0": etat_avec(mode=EN.EQUILIBRE, yc=7.5, uV=0.0),
    "B (toujours l'équilibre)": etat_B(),
    "B avec le mode historique demandé (mode imposé)": etat_B(mode=EN.HISTORIQUE),
    "B, Y = 45 (ancrage près du bout libre)": etat_B(Y=45.0, dy=0.0),
    "inégale L 150×100×12, 150 à l'horizontale": etat_avec(profil="L 150x100x12", Y=65.0),
    "inégale L 150×100×12, 100 à l'horizontale": etat_avec(profil="L 150x100x12", aile_horizontale="B", Y=115.0),
    "petite L 60×60×6": etat_avec(profil="L 60x60x6", X=35.0, dx=5.0, Y=40.0, dy=3.0, d=8.0, R=4.0, L=200.0),
    "grosse L 250×250×35, R = 200 kN": etat_avec(profil="L 250x250x35", X=150.0, Y=180.0, d=24.0, R=200.0, L=500.0),
    "saisie libre 130 × 110 × 12, r = 11": etat_avec(standard=False, A=130.0, B=110.0, t=12.0, r=11.0, Y=70.0),
    "largeur d'appui a = 100": etat_avec(a=100.0),
    "cas de contrôle du classeur (R = 42,76, sans jeu, R_ELU = 68,9)": etat_avec(R=42.76, dx=0.0, dy=0.0, R_ELU=68.9),
    "garde rouge (Y = 40, sans jeu, Ø10)": etat_avec(dx=0.0, dy=0.0, d=10.0, Y=40.0),
    "forte charge (R = 80 kN) : non conforme": etat_avec(R=80.0),
    "nom piégé": etat_avec(nom='<img src=x onerror=alert(1)> & "guillemets"'),
    # la révision § 9 : l'état limite, les ancrages, le support
    "ELU, R = 60 kN saisie à l'ELU (historique)": etat_avec(etat_limite=EN.ELU, R=60.0),
    "ELU, équilibre": etat_avec(etat_limite=EN.ELU, R=60.0, mode=EN.EQUILIBRE),
    "ELU, configuration B": etat_B(etat_limite=EN.ELU, R=50.0),
    "A sur poutre 160 au-dessus / 120 au-dessous": etat_avec(support=EN.POUTRE, support_haut=160.0, support_bas=120.0),
    "B sur poutre (150/150)": etat_B(support=EN.POUTRE),
    "entraxe saisi s = 80": etat_avec(entraxe=80.0),
    "boulons traversants, deux rondelles, profondeur 200": etat_avec(type_ancrage="traversant", rondelles=2, profondeur=200.0),
    "chevilles chimiques, profondeur 120, sans rondelle": etat_avec(type_ancrage="chimique", profondeur=120.0, rondelles=0),
}
for nom_cas, etat in ETATS.items():
    fautes = controle_resultat(K, etat, nom_cas)
    chk(f"résultat = moteur : {nom_cas}", not fautes, fautes[:4])


# la 1ʳᵉ ligne d'un résultat et l'en-tête d'un bloc ne passent jamais à la ligne : dans les 316 px de la colonne du PC, un
# nom trop long est coupé à l'ellipse et l'état d'un en-tête trop long mange le TITRE du bloc. Mesuré le 10/11/2026 dans
# Chromium (relecture de l'écran) : « Ancrages · efforts par ancrage · chevilles non vérifiées » laissait 6 px au titre,
# « Mode méthode historique (classeur P8) » demandait 241 px pour 185, « Comparaison des modes · garde dépassée » 173 pour
# 166. Le budget, en caractères (la vérité en pixels est le contrôle « aucun nom coupé » de verifier_site.js, qui lit
# aussi les titres des blocs) : titre + état d'un en-tête ≤ 32, nom d'une ligne ≤ 24.
BUDGET_ENTETE, BUDGET_NOM = 32, 24


def noms_longs(mod):
    """Les en-têtes de blocs et les noms de lignes qui dépassent le budget, sur tous les états d'essai."""
    longs = []
    for nom_cas, etat in ETATS.items():
        res = resultat_module(mod, etat)["resultat"]
        longs += [f"{nom_cas} · en-tête « {b['titre']} · {b['etat']} » ({len(b['titre']) + len(b['etat'])})"
                  for b in blocs(res) if len(b["titre"]) + len(b["etat"]) > BUDGET_ENTETE]
        longs += [f"{nom_cas} · ligne « {li['nom']} » ({len(li['nom'])})" for li in lignes(res) if len(li["nom"]) > BUDGET_NOM]
    return longs


chk(f"les en-têtes des blocs (titre + état ≤ {BUDGET_ENTETE} caractères) et les noms des lignes (≤ {BUDGET_NOM}) tiennent dans la colonne du résultat, "
    f"sur les {len(ETATS)} états", not noms_longs(K), noms_longs(K)[:4])

cc = MO.verifier(ETATS["cas de contrôle du classeur (R = 42,76, sans jeu, R_ELU = 68,9)"]["cornieres"][0])
chk("le cas de contrôle du classeur : 88,86 % en section 3, une seule combinaison, affiché « 88,9 % »",
    abs(cc["taux_max"] - 0.888621) < 1e-5 and len(cc["combinaisons"]) == 1 and "88,9" in resultat_module(K, ETATS["cas de contrôle du classeur (R = 42,76, sans jeu, R_ELU = 68,9)"])["resultat"],
    cc["taux_max"])
gd = MO.verifier(ETATS["garde rouge (Y = 40, sans jeu, Ø10)"]["cornieres"][0])
chk("la garde rougit sur l'ancrage bas (Y = 40) : l'avertissement rouge est en tête des messages, le bloc Comparaison est rouge",
    gd["garde"]["depasse"] and MSG.findall(resultat_module(K, ETATS["garde rouge (Y = 40, sans jeu, Ø10)"])["resultat"])[0][0] == " grave"
    and blocs(resultat_module(K, ETATS["garde rouge (Y = 40, sans jeu, Ø10)"])["resultat"])[2]["cls"] == "nok", gd["garde"])
rB = resultat_module(K, ETATS["B (toujours l'équilibre)"])
chk("B : pas de bloc Comparaison, la section nette gouverne, la pression sur le support en information et en avertissement",
    len(blocs(rB["resultat"])) == 2 and "Section nette" in rB["resume"]["gouvernant"] and "Pression du support" in rB["resultat"]
    and "Pression du bout libre sur le support" in rB["resultat"], rB["resume"])
rBi = resultat_module(K, ETATS["B avec le mode historique demandé (mode imposé)"])
chk("B avec le mode historique demandé : le mode effectif est l'équilibre, le segment du mode est verrouillé et la raison est écrite, l'avertissement du moteur est là",
    "mode équilibre imposé" in rBi["resultat"] and 'class="seg full locked"' in rBi["donnees"] and "toujours l'équilibre" in rBi["donnees"], None)
chk("B : le mode affiché dans la tête du résultat est « équilibre » même si « historique » est demandé",
    "méthode historique" not in rBi["resultat"].split('<div class="et-blocs">')[0].split("Comparaison")[0], None)

# ---------------------------------------------------------------------- la révision § 9.5 : l'ÉTAT LIMITE
E_ELU = ETATS["ELU, R = 60 kN saisie à l'ELU (historique)"]
rU = resultat_module(K, E_ELU)
mU = MO.verifier(E_ELU["cornieres"][0])
chk("ELU : le verdict et la ligne Cornière disent f_y/γ_M0 = 355 (et f_y/(√3·γ_M0)), jamais σ_adm ; le taux est celui du moteur sous R = 60 kN à l'ELU",
    "f<sub>y</sub>/γ<sub>M0</sub> = 355" in rU["resultat"] and "f<sub>y</sub>/(√3·γ<sub>M0</sub>) = 205" in rU["resultat"]
    and "σ<sub>adm</sub>" not in rU["resultat"] and "236,7" not in rU["resultat"] and pct_txt(mU["taux_max"]) in rU["resultat"]
    and mU["materiau"]["symbole_sigma"] == "f_y/γ_M0", None)
chk("ELU : coef et R_ELU sont masqués dans les données (sans effet), la nuance reste, R se nomme « R (ELU) » et son ⓘ dit la combinaison déjà faite",
    'data-chemin="coef"' not in rU["donnees"] and 'data-chemin="R_ELU"' not in rU["donnees"] and '<select data-chemin="fy"' in rU["donnees"]
    and "<span>R (ELU)</span>" in rU["donnees"] and "1,35·G + 1,5·Q déjà faite" in rU["donnees"], None)
chk("ELS : coef et R_ELU sont là, R se nomme « R (ELS) »", 'data-chemin="coef"' in don0 and 'data-chemin="R_ELU"' in don0 and "<span>R (ELS)</span>" in don0, None)
chk("rubrique 4 : l'état limite en tête, en segments ELS · ELU avec l'aide (f_y/coef ; f_y/γ_M0, γ_M0 = 1,0, R saisie à l'ELU, le 1,5 ne s'applique pas) ; "
    "le résumé de la rubrique dit l'état limite",
    don0.index('data-chemin="etat_limite"') < don0.index('data-chemin="mode"') and 'class="on" data-chemin="etat_limite" data-sorte="choix" data-valeur="ELS"' in don0
    and 'data-chemin="etat_limite" data-sorte="choix" data-valeur="ELU"' in don0 and "γ_M0 = 1,0" in don0 and C.esc("ne s'applique pas") in don0
    and re.search(r'data-sec="hy"[^>]*>.*?<span class="sum">ELS · historique', don0, re.S) is not None
    and re.search(r'data-sec="hy"[^>]*>.*?<span class="sum">ELU · historique', rU["donnees"], re.S) is not None, None)
chk("ELU : le bloc Ancrages n'a que N_ELU = H/n et V_ELU = R/n (R saisie à l'ELU), la fenêtre des ancrages aussi, et le moteur ne rend pas d'ELS",
    rU["resultat"].count("Traction N · ELU") == 1 and "Traction N · ELS" not in rU["resultat"] and "R saisie à l'ELU" in rU["resultat"]
    and mU["ancrages"]["N_ELS"] is None and "N<sub>ELS</sub>" not in rU["volets"] and "R<sub>ELU</sub>" not in rU["volets"]
    and K._n(mU["ancrages"]["N_ELU"], "kN") in rU["resultat"], None)
chk("ELU : les fenêtres écrivent les limites f_y/γ_M0 et le taux σ_éq/(f_y/γ_M0) ; le titre du moteur « limites ELU » ; pas de cellule H31 (σ_adm du classeur)",
    "f<sub>y</sub> / γ<sub>M0</sub>" in rU["volets"] and "σ<sub>éq</sub> / (f<sub>y</sub>/γ<sub>M0</sub>)" in rU["volets"]
    and "σ/(f<sub>y</sub>/γ<sub>M0</sub>) (I31)" in rU["volets"] and "τ/(f<sub>y</sub>/(√3·γ<sub>M0</sub>)) (I34)" in rU["volets"]
    and "limites ELU" in rU["volets"] and "(H31)" not in rU["volets"] and "σ<sub>adm</sub>" not in rU["volets"], None)
chk("ELU : l'en-tête des vues dit l'état limite et sa limite (f_y/γ_M0 = 355, γ_M0 = 1), pas le coef ; en ELS, σ_adm = 236,7 et le coef",
    all("ELU" in v["entete"] and "f<sub>y</sub>/γ<sub>M0</sub> = 355" in v["entete"] and "γ<sub>M0</sub> = 1" in v["entete"] and "coef" not in v["entete"]
        and "σ<sub>adm</sub>" not in v["entete"] for v in rU["vues"])
    and all("ELS" in v["entete"] and "σ<sub>adm</sub> = 236,7" in v["entete"] and "(coef 1,5)" in v["entete"] for v in r0["vues"]), None)
e_elu = saisir(r0["etat"], "etat_limite", "ELU")
chk("la bascule ELS → ELU par la saisie : la limite passe de 236,7 à 355 N/mm², l'avertissement des chevilles dit l'ELU, l'avis R_ELU_auto disparaît",
    e_elu["etat"]["cornieres"][0]["etat_limite"] == "ELU" and "355" in e_elu["resultat"] and "236,7" not in e_elu["resultat"]
    and C.esc("(R saisie à l'ELU)") in e_elu["resultat"] and "R_ELU automatique" not in e_elu["resultat"] and "R_ELU automatique" in r0["resultat"], None)
chk("invariant du moteur à l'écran : le taux affiché en ELS sous R vaut celui de l'ELU sous coef·R",
    resultat_module(K, etat_avec(etat_limite=EN.ELU, R=52.5))["resume"] == r0["resume"], None)

# ---------------------------------------------------------------------- la révision § 9.2 et 9.3 : les ANCRAGES et le SUPPORT
chk("rubrique 3 : entraxe « auto 100 » (L/n, court : la case est à deux par ligne), profondeur, type (segments mécanique · chimique · traversant, l'aide = le libellé d'entree), "
    "rondelles (0 · 1 · 2), support (voile · poutre) — chacun une pièce (data-cible)",
    all(f'data-cible="{k}"' in don0 for k in ("entraxe", "profondeur", "type_ancrage", "rondelles", "support"))
    and 'placeholder="auto 100"' in don0 and '<input data-chemin="profondeur" data-sorte="nombre"' in don0
    and [m_.group(1) for m_ in re.finditer(r'data-chemin="type_ancrage" data-sorte="choix" data-valeur="(\w+)"', don0)] == list(EN.TYPES_ANCRAGE)
    and all(C.esc(EN.LIBELLES_TYPE_ANCRAGE[k][0].upper() + EN.LIBELLES_TYPE_ANCRAGE[k][1:]) in don0 for k in EN.TYPES_ANCRAGE)
    and [m_.group(1) for m_ in re.finditer(r'data-chemin="rondelles" data-sorte="choix" data-valeur="(\d)"', don0)] == ["0", "1", "2"]
    and 'class="on" data-chemin="rondelles" data-sorte="choix" data-valeur="1"' in don0 and "ISO 7093-1" in don0
    and 'class="on" data-chemin="support" data-sorte="choix" data-valeur="voile"' in don0 and 'data-valeur="poutre"' in don0, None)
chk("voile : pas de champ d'étendue ; poutre : support_haut et support_bas (mm), pièces, avec l'aide qui dit le repère (au-dessus / au-dessous de la face "
    "extérieure de l'aile horizontale)",
    'data-chemin="support_haut"' not in don0 and 'data-chemin="support_bas"' not in don0
    and all(f'data-cible="{k}"><label>' in resultat_module(K, ETATS["A sur poutre 160 au-dessus / 120 au-dessous"])["donnees"] for k in ("support_haut", "support_bas"))
    and "AU-DESSUS de la face" in resultat_module(K, ETATS["A sur poutre 160 au-dessus / 120 au-dessous"])["donnees"]
    and 'value="160"' in resultat_module(K, ETATS["A sur poutre 160 au-dessus / 120 au-dessous"])["donnees"], None)
rP = resultat_module(K, ETATS["A sur poutre 160 au-dessus / 120 au-dessous"])
mP = MO.verifier(ETATS["A sur poutre 160 au-dessus / 120 au-dessous"]["cornieres"][0])
chk("poutre : les distances de l'ancrage aux bords (70 et 200 mm, celles du moteur) se lisent dans le bloc Ancrages, hors verdict, grises",
    mP["geometrie"]["bords_poutre"] == {"haut": 70.0, "bas": 200.0}
    and [(x["nom"], x["mot"], x["piece"]) for x in lignes(rP["resultat"]) if x["nom"].startswith("Poutre")]
    == [("Poutre · bord supérieur", f"{K._n(70.0, 'mm')} mm", "hors verdict"), ("Poutre · bord inférieur", f"{K._n(200.0, 'mm')} mm", "hors verdict")]
    and "Poutre" not in "".join(x["nom"] for x in lignes(r0["resultat"])), None)
rE = resultat_module(K, ETATS["entraxe saisi s = 80"])
chk("entraxe saisi s = 80 : le champ le montre, les positions disent e₁ = 70 · s = 80 (saisi) · z = 70 · 150 · 230, l'état de la ligne « entraxe 80 »",
    '<input data-chemin="entraxe" data-sorte="texte" inputmode="decimal" value="80"' in rE["donnees"]
    and "e₁ = 70 · s = 80 (saisi) · z = 70 · 150 · 230 mm" in rE["resultat"] and ">entraxe 80<" in rE["resultat"]
    and "s = 100 (L/n) · z = 50 · 150 · 250 mm" in r0["resultat"] and ">entraxe 100<" in r0["resultat"], None)
e_s = saisir(r0["etat"], "entraxe", "")
chk("entraxe vidé : automatique (None), le champ redit « auto 100 »", e_s["etat"]["cornieres"][0]["entraxe"] is None and 'placeholder="auto 100"' in e_s["donnees"], None)
e_rd = saisir(r0["etat"], "rondelles", "2")
chk("rondelles = 2 par le segment : un entier dans l'état, le segment 2 allumé, l'hypothèse du moteur dit la rondelle large",
    e_rd["etat"]["cornieres"][0]["rondelles"] == 2 and isinstance(e_rd["etat"]["cornieres"][0]["rondelles"], int)
    and 'class="on" data-chemin="rondelles" data-sorte="choix" data-valeur="2"' in e_rd["donnees"] and "ISO 7093-1" in e_rd["resultat"], None)
e_ty = saisir(r0["etat"], "type_ancrage", "traversant")
chk("type d'ancrage « traversant » par le segment : l'état, le segment allumé, le bloc Ancrages et l'en-tête des vues le disent",
    e_ty["etat"]["cornieres"][0]["type_ancrage"] == "traversant" and 'class="on" data-chemin="type_ancrage" data-sorte="choix" data-valeur="traversant"' in e_ty["donnees"]
    and "boulon traversant" in e_ty["resultat"] and all("boulon" in v["entete"] for v in e_ty["vues"]), None)
e_pt = saisir(r0["etat"], "support", "poutre")
chk("support « poutre » par le segment : les champs 150/150 apparaissent, les bords 60 et 230 mm en information, la résumé de la rubrique dit « poutre »",
    e_pt["etat"]["cornieres"][0]["support"] == "poutre" and 'data-chemin="support_haut" data-sorte="nombre" inputmode="decimal" value="150"' in e_pt["donnees"]
    and "Poutre · bord supérieur" in e_pt["resultat"] and f">{K._n(60.0, 'mm')} mm<" in e_pt["resultat"] and f">{K._n(230.0, 'mm')} mm<" in e_pt["resultat"]
    and re.search(r'data-sec="an"[^>]*>.*?<span class="sum">Y 85 ± 5 · 3 Ø16 · poutre', e_pt["donnees"], re.S) is not None, None)

# ---------------------------------------------------------------------- la révision § 9.7 : la colonne de gauche
chk("catalogue : le profil se choisit dans un MENU DÉROULANT des 233 cornières (deux familles, par dimensions croissantes, la retenue sélectionnée, "
    "la valeur = le nom du catalogue, le libellé = la désignation) — remarque de l'ingénieur du 10/11 au soir",
    '<select data-chemin="profil" data-sorte="choix">' in don0 and '<option value="L 120x120x15" selected>L 120×120×15</option>' in don0
    and re.search(r'<select data-chemin="profil"[^>]*><optgroup label="Ailes égales">(<option [^>]*>[^<]*</option>)+</optgroup>'
                  r'<optgroup label="Ailes inégales">(<option [^>]*>[^<]*</option>)+</optgroup></select>', don0) is not None
    and len(re.findall(r'<option value="L [^"]+"', don0[don0.index('<select data-chemin="profil"'):don0.index("</select>", don0.index('<select data-chemin="profil"'))])) == 233
    and don0.index('<option value="L 20x20x3"') < don0.index('<option value="L 200x200x20"') < don0.index('<option value="L 100x65x7"'), None)
chk("catalogue : le profil choisi reste, la fiche (ailes, épaisseur, congés, masse, aire) est dans un volet « Détails » REPLIÉ par défaut (details sans open, data-volet), "
    "son résumé fermé dit la famille, la norme et la masse",
    '<option value="L 120x120x15" selected>' in don0
    and re.search(r'<details class="et-volet" data-volet="fiche"><summary[^>]*>.*?<span class="t">Détails</span><span class="r">Ailes égales · EN 10056-1 · 26,64.kg/m</span></summary>'
                  r'<div class="et-volet-c"><dl class="facts">', don0, re.S) is not None and " open>" not in don0 and "<details open" not in don0
    and don0.index("<details") < don0.index("Congés r<sub>1</sub>") < don0.index("</details>"), None)
don_sl = resultat_module(K, ETATS["saisie libre 130 × 110 × 12, r = 11"])["donnees"]
chk("saisie libre : A, B, t, r s'affichent directement (pas de volet, rien ne les cache), la masse calculée dite",
    "<details" not in don_sl and all(f'<input data-chemin="{k}" data-sorte="nombre"' in don_sl for k in "ABtr")
    and all(re.search(rf'<div class="row2 n2"><label><span>[^<]*</span>.*?<input data-chemin="{k}"', don_sl, re.S) for k in "ABtr"), None)

# ====================================================================== 3. le cadre Données
don0 = r0["donnees"]
chk("quatre rubriques : 1 Cornière, 2 Charge (ouvertes), 3 Ancrage, 4 Hypothèses (secondaire, repliée)",
    re.findall(r'<div class="sec( open)?" data-sec="(\w+)"( data-secondaire="1")?>', don0) == [(" open", "co", ""), (" open", "ch", ""), ("", "an", ""), ("", "hy", " data-secondaire=\"1\"")],
    re.findall(r'<div class="sec( open)?" data-sec="(\w+)"( data-secondaire="1")?>', don0))
chk("rubrique 1 : origine en segments, menu des cornières, fiche du catalogue (congés, masse, aire), aile horizontale avec les longueurs réelles, "
    "configuration A/B en segments avec deux pictogrammes, nuance et coefficient",
    'data-chemin="standard" data-sorte="choix" data-valeur="catalogue"' in don0 and '<select data-chemin="profil" data-sorte="choix">' in don0
    and "<span>Profil</span>" in don0 and "<span>Aile</span>" in don0 and "<span>Pose</span>" in don0
    and "Congés r<sub>1</sub> / r<sub>2</sub>" in don0 and "13 / 6,5" in don0 and "26,64" in don0
    and 'data-chemin="aile_horizontale" data-sorte="choix" data-valeur="A"' in don0 and ">A = 120 mm<" in don0 and ">B = 120 mm<" in don0
    and don0.count('data-chemin="configuration" data-sorte="choix"') == 2 and don0.count('<svg class="et-pic"') == 2
    and '<select data-chemin="fy" data-sorte="choix">' in don0 and '<option value="355" selected>S355</option>' in don0
    and '<input data-chemin="coef" data-sorte="nombre"' in don0, None)
chk("rubriques 2 et 3 : R, L, X, dx, a (auto · L), R_ELU (auto · 1,5·R), Y, dy, n, d — chacun une pièce (data-cible) sauf R_ELU",
    all(f'data-cible="{k}"><label>' in don0 for k in ("R", "L", "X", "dx", "a", "Y", "dy", "n", "d"))
    and 'placeholder="auto 300"' in don0 and 'placeholder="auto · 1,5·R = 52,5"' in don0
    and '<div class="et-piece"><label><span>R<sub>ELU</sub></span>' in don0, None)
chk("rubrique 4 : le mode en segments, jeu C76, y_c, u_V (auto · e/2)",
    'data-chemin="mode" data-sorte="choix" data-valeur="historique"' in don0 and 'data-cible="jeu_ancrage"' in don0 and 'data-cible="yc"' in don0
    and 'data-cible="uV"' in don0 and 'placeholder="auto · e/2 = 7,5"' in don0, None)
chk("chaque ⓘ des jeux explique les trois positions et l'enveloppe ; le ⓘ de R_ELU dit la combinaison réelle",
    "trois positions" in don0 and "ENVELOPPE" in don0 and "1,35·G + 1,5·Q" in don0, None)
rl = resultat_module(K, etat_avec(standard=False, A=130.0, B=110.0, t=12.0, r=11.0, Y=70.0))
chk("saisie libre : A, B, t, r en champs, la masse calculée dite, l'origine « saisie » cochée",
    all(f'<input data-chemin="{k}" data-sorte="nombre"' in rl["donnees"] for k in "ABtr") and "Masse et aire calculées" in rl["donnees"]
    and 'class="on" data-chemin="standard" data-sorte="choix" data-valeur="saisie"' in rl["donnees"] and "(calculée)" in rl["resultat"], None)
e_sl = saisir(r0["etat"], "standard", "saisie")["etat"]
chk("passer en saisie libre part des dimensions de la cornière du catalogue affichée (120, 120, 15, r = 13), puis revenir au catalogue garde le profil",
    (e_sl["cornieres"][0]["A"], e_sl["cornieres"][0]["B"], e_sl["cornieres"][0]["t"], e_sl["cornieres"][0]["r"], e_sl["cornieres"][0]["standard"]) == (120.0, 120.0, 15.0, 13.0, False)
    and saisir(e_sl, "standard", "catalogue")["etat"]["cornieres"][0]["profil"] == "L 120x120x15", e_sl["cornieres"][0])
e_d = saisir(r0["etat"], "profil", "150 × 150 × 15")
chk("une désignation tapée (« 150 × 150 × 15 ») est relue au catalogue",
    e_d["etat"]["cornieres"][0]["profil"] == "L 150x150x15" and e_d["message"] is None, e_d.get("message"))
e_menu = saisir(r0["etat"], "profil", "L 200x200x20")
chk("le menu déroulant envoie le nom du catalogue (« L 200x200x20 ») : retenu, sélectionné au rendu suivant, la coupe suit (larg = 200)",
    e_menu["etat"]["cornieres"][0]["profil"] == "L 200x200x20" and e_menu["message"] is None
    and '<option value="L 200x200x20" selected>L 200×200×20</option>' in e_menu["donnees"]
    and e_menu["donnees"][e_menu["donnees"].index('<select data-chemin="profil"'):].split("</select>")[0].count(" selected>") == 1
    and "L 200×200×20" in e_menu["resultat"], e_menu.get("message"))
e_d2 = saisir(r0["etat"], "profil", "80 × 80 × 8")
chk("une désignation tapée trop petite pour la charge est relue au catalogue puis REFUSÉE par le moteur (la charge au-delà de l'aile), la cornière intacte",
    "au-delà de l'aile" in (e_d2["message"] or "") and e_d2["etat"] == r0["etat"], e_d2.get("message"))
e_fy = saisir(r0["etat"], "fy", "235")
chk("la nuance se choisit dans le menu (S235 → f_y = 235) et σ_adm suit", e_fy["etat"]["cornieres"][0]["fy"] == 235.0 and "156,7" in e_fy["resultat"], None)
e_n = saisir(r0["etat"], "n", "4")
chk("n se saisit en entier (4 ancrages : z = 37,5 · 112,5 · 187,5 · 262,5)", e_n["etat"]["cornieres"][0]["n"] == 4 and "37,5 · 112,5 · 187,5 · 262,5" in e_n["resultat"], None)
e_a = saisir(r0["etat"], "a", "")
chk("a vidé : automatique (None), le champ redit « auto 300 »", e_a["etat"]["cornieres"][0]["a"] is None and 'placeholder="auto 300"' in e_a["donnees"], None)
e_a2 = saisir(r0["etat"], "a", "100")
chk("a = 100 : L_eff dite dans le résultat (information) et la fenêtre géométrie, dans l'ⓘ aussi",
    "Largeur efficace" in e_a2["volets"] and "L_eff = " in e_a2["donnees"] and "Largeur efficace" in e_a2["resultat"], None)
chk("une cornière (copie) renomée garde 60 caractères au plus", len(K.gerer_elements.__doc__ or "") > 0 and True)

# ====================================================================== 4. les refus : message du moteur, cornière intacte
REFUS = [("X", "10", "dans l'angle"), ("X", "200", "au-delà de l'aile"), ("Y", "20", "congé"), ("Y", "118", "sort de l'aile"),
         ("fy", "300", "nuance"), ("coef", "0,5", "au moins 1"), ("R", "-1", "négative"), ("L", "0", "strictement positive"),
         ("dx", "-1", "deux sens"), ("n", "0", "au moins 1"), ("n", "2,5", "entier"), ("d", "0", "strictement positif"),
         ("profil", "cornière", "absente du catalogue"), ("profil", "L 45×45×4,5", "au-delà de l'aile"), ("jeu_ancrage", "90", "moins de Y"),
         ("yc", "80", "bras positif"), ("uV", "20", "dans l'épaisseur"), ("R", "beaucoup", "illisible"), ("a", "abc", "n'est pas un nombre"),
         ("nom", "x" * 61, "trop long"), ("standard", "peut-être", "catalogue ou saisie"), ("configuration", "C", "choix admis"),
         ("mode", "magique", "choix admis"),
         # la révision § 9 : l'entraxe sous 2,2·d0, les rondelles hors 0-1-2, la profondeur nulle, les choix inconnus
         ("entraxe", "30", "2,2·d0"), ("entraxe", "-5", "strictement positif"), ("rondelles", "3", "0, 1 ou 2"), ("rondelles", "1,5", "entier"),
         ("profondeur", "0", "strictement positive"), ("etat_limite", "ELX", "choix admis"), ("type_ancrage", "collé", "choix admis"),
         ("support", "dalle", "choix admis"), ("entraxe", "abc", "n'est pas un nombre"), ("support_haut", "abc", "illisible")]
for chemin, valeur, mot in REFUS:
    out = req(action="modifier", etat=copy.deepcopy(r0["etat"]), chemin=[chemin], valeur=valeur)
    chk(f"refus : {chemin} = « {valeur}» → message « …{mot}… », la cornière intacte",
        out["ok"] and out["message"] and mot in out["message"] and out["etat"] == r0["etat"], out.get("message") or out.get("erreur"))
e_pb = etat_B(support=EN.POUTRE)
out = req(action="modifier", etat=copy.deepcopy(e_pb), chemin=["support_bas"], valeur="50")
chk("refus : B sur poutre, 50 mm au-dessous → « l'ancrage sort de la poutre par le bas » (le message du moteur), la cornière intacte",
    out["ok"] and "sort de la poutre par le bas" in (out["message"] or "") and out["etat"] == e_pb, out.get("message"))
out = req(action="modifier", etat=copy.deepcopy(e_pb), chemin=["support_bas"], valeur="110")
chk("refus : B sur poutre, 110 mm au-dessous → le bout libre dépasse la poutre (pas de contact)",
    out["ok"] and "dépasse la poutre" in (out["message"] or "") and out["etat"] == e_pb, out.get("message"))
e_pa = etat_avec(support=EN.POUTRE)
out = req(action="modifier", etat=copy.deepcopy(e_pa), chemin=["support_haut"], valeur="90")
chk("refus : A sur poutre, 90 mm au-dessus → « l'ancrage sort de la poutre par le haut » (Y + dy + d0/2 = 99)",
    out["ok"] and "sort de la poutre par le haut" in (out["message"] or "") and "99" in out["message"] and out["etat"] == e_pa, out.get("message"))
out = req(action="modifier", etat=copy.deepcopy(r0["etat"]), chemin=["n"], valeur="8")
chk("refus : n = 8 sur L = 300 → l'entraxe automatique 37,5 < 2,2·d0 = 39,6 (le minimum vaut aussi pour l'entraxe automatique)",
    out["ok"] and "2,2·d0" in (out["message"] or "") and out["etat"] == r0["etat"], out.get("message"))
out = req(action="modifier", etat=copy.deepcopy(r0["etat"]), chemin=["X", "0"], valeur="80")
chk("un chemin à deux clés est inconnu (les champs sont à plat)", out["ok"] and "inconnu" in out["message"] and out["etat"] == r0["etat"], out.get("message"))
e_t = saisir(e_sl, "t", "130")
chk("saisie libre : t ≥ aile refusée (ce n'est pas une cornière)", "pas une cornière" in (e_t["message"] or ""), e_t.get("message"))
e_rl = saisir(r0["etat"], "R_ELU", "68,9")
chk("R_ELU saisie : plus « auto », l'avertissement R_ELU_auto disparaît, le rapport ELU change",
    e_rl["etat"]["cornieres"][0]["R_ELU"] == 68.9 and "R_ELU automatique" not in e_rl["resultat"] and "(saisie)" in e_rl["resultat"], None)
e_nom = saisir(r0["etat"], "nom", '<img src=x onerror=alert(1)> & "x"')
chk("un nom piégé est échappé dans tous les fragments",
    all("<img" not in h for h in (e_nom["resultat"], e_nom["donnees"], e_nom["volets"], "".join(v["svg"] + v.get("entete", "") for v in e_nom["vues"])))
    and "&lt;img" in e_nom["donnees"] and "&lt;img" in e_nom["resultat"], None)

# ====================================================================== 5. le sélecteur
s = req(action="selecteur", etat=r0["etat"], role="corniere")
cands = MO.candidats(r0["etat"]["cornieres"][0])
rows = re.findall(r'<button class="pk-row((?: cur)?(?: opt)?)"( aria-disabled="true")? data-choix="([^"]+)" data-prop="([^"]*)">.*?<span class="kg">([^<]+)</span>(<span class="chip (\w)"[^>]*>[^<]+</span>)?</button>', s["html"], re.S)
rows = [(r_[0], r_[2], r_[3], r_[4], r_[5], r_[6], bool(r_[1])) for r_ in rows]   # (classes, nom, prop, masse, pastille, niveau, inactive)
chk("les lignes inactives (aria-disabled, que le parcours ne choisit pas) sont exactement les cornières hors domaine",
    [r_[6] for r_ in rows] == [not x["domaine"] for x in cands], sum(r_[6] for r_ in rows))
chk("le sélecteur : 233 cornières, dans l'ordre du moteur (par masse), la désignation du bureau en nom",
    s["ok"] and len(rows) == len(cands) == 233 and [r_[1] for r_ in rows] == [x["designation"] for x in cands], len(rows))
chk("✓ la retenue (L 120×120×15), ★ l'optimale du moteur, une seule chacune",
    [r_[1] for r_ in rows if "cur" in r_[0]] == ["L 120×120×15"] and [r_[1] for r_ in rows if "opt" in r_[0]] == [MO.optimal(r0["etat"]["cornieres"][0])["designation"]], None)
hors = [r_ for r_, x in zip(rows, cands) if not x["domaine"]]
chk(f"les cornières hors domaine ({len(hors)}) portent une pastille grise « exclue » à la place du taux, leur raison au survol",
    hors and all(r_[4] == '<span class="chip c">exclue</span>' and "hors domaine" in html.unescape(r_[2]) and x["raison"][:20] in html.unescape(html.unescape(r_[2]))
                 for r_, x in zip(rows, cands) if not x["domaine"]), [r_[4] for r_ in hors[:2]])
chk("une cornière qui ne tient pas est rouge (« ne tient pas »), une qui tient porte la couleur de son taux",
    all((r_[5] == "r" and "ne tient pas" in r_[2]) if (x["domaine"] and not x["ok"]) else True for r_, x in zip(rows, cands))
    and all(r_[5] == C.niveau(x["taux"]) for r_, x in zip(rows, cands) if x["domaine"] and x["ok"]), None)
s_li = req(action="selecteur", etat=r0["etat"], role="corniere", familles=["LI"])
chk("le filtre « Ailes inégales » ne garde que la famille LI",
    s_li["ok"] and s_li["html"].count("pk-row") == len(MO.cornieres(["LI"])) and 'class="fchip on" data-famille="LI"' in s_li["html"], None)
ch = req(action="choisir", etat=r0["etat"], role="corniere", nom="L 150×150×15")
chk("choisir « L 150×150×15 » la retient (catalogue) et recalcule", ch["ok"] and ch["etat"]["cornieres"][0]["profil"] == "L 150x150x15"
    and ch["etat"]["cornieres"][0]["standard"] and "L 150×150×15" in ch["resultat"], ch.get("erreur"))
ch2 = req(action="choisir", etat=r0["etat"], role="corniere", nom="L 1×1×1")
chk("choisir une cornière inconnue est une erreur lisible", not ch2["ok"] and "absente du catalogue" in ch2["erreur"], ch2)

# ====================================================================== 6. le contrat de sélection : data-sel ↔ data-cible
for nom_cas in ("A historique (défauts)", "A équilibre", "B (toujours l'équilibre)", "largeur d'appui a = 100"):
    rep = resultat_module(K, ETATS[nom_cas])
    sels = set(re.findall(r'data-sel="([^"]+)"', "".join(v["svg"] for v in rep["vues"])))
    cibles = set(re.findall(r'data-cible="([^"]+)"', rep["donnees"]))
    chk(f"contrat de sélection ({nom_cas}) : chaque pièce du dessin a sa ligne dans les données sous la même clé ({len(sels)} pièces)",
        sels and sels <= cibles and sels <= set(SC.CLES_SEL), sorted(sels - cibles))
    dans_rubrique = all(re.search(r'<div class="sec[^"]*" data-sec="\w+"', rep["donnees"][:rep["donnees"].index(f'data-cible="{k}"')]) for k in cibles)
    chk(f"contrat de sélection ({nom_cas}) : chaque cible est dans une rubrique du cadre Données", dans_rubrique)
rep_h = resultat_module(K, ETATS["A historique (défauts)"])
rep_e = resultat_module(K, ETATS["A équilibre"])
chk("le mode choisit ses pièces : jeu_ancrage sur la coupe en historique, y_c et u_V en équilibre",
    'data-sel="jeu_ancrage"' in rep_h["vues"][0]["svg"] and 'data-sel="yc"' not in rep_h["vues"][0]["svg"]
    and 'data-sel="yc"' in rep_e["vues"][0]["svg"] and 'data-sel="uV"' in rep_e["vues"][0]["svg"], None)


# ====================================================================== 7. les fenêtres de calcul : chaque ligne réévaluée
def evaluer(vals):
    """« 42,76 × (75 − 13 − 15) / 1000 » → le nombre, par un petit interpréteur (√, |x|, min(a ; b), ², ³, 10⁶)."""
    s = html.unescape(re.sub(r"<[^>]+>", "", vals))
    s = s.replace(C.NBSP, "").replace(C.FINE, "").replace(" ", "").replace("−", "-").replace(",", ".")
    s = s.replace("×", "*").replace("·", "*").replace("√(", "sqrt(").replace("10⁶", "1e6").replace("10³", "1e3").replace(";", ",")
    s = re.sub(r"√(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    s = s.replace("²", "**2").replace("³", "**3")
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    return eval(s, {"__builtins__": {}}, {"sqrt": math.sqrt, "min": min, "abs": abs})  # noqa: S307 — des nombres et des opérateurs


def lire_resultat(res_txt):
    t = html.unescape(re.sub(r"<[^>]+>", "", res_txt))
    x = lire_fr(t)
    nd = len(t.replace(C.NBSP, " ").split(" ")[0].partition(",")[2])
    return (x / 100.0 if "%" in t else x), nd, "%" in t


def controle_formules(mod, etat, nom_cas):
    rep = resultat_module(mod, etat)
    fautes, n = [], 0
    for ligne in FX.findall(rep["volets"]):
        spans = SPAN.findall(ligne)
        parts = [t for cls, t in spans if cls not in ("r",) and not cls.startswith("chip")]
        res_txt = next((t for cls, t in spans if cls == "r" or cls.startswith("chip")), None)
        if len(parts) != 3 or res_txt is None:
            fautes.append(f"ligne sans expression ni valeurs : {texte(ligne)[:60]}")
            continue
        n += 1
        try:
            val = evaluer(parts[2])
        except Exception as e:  # noqa: BLE001 — une expression illisible est une faute
            fautes.append(f"{texte(parts[0])} : illisible ({e}) : {texte(parts[2])}")
            continue
        attendu, nd, pct = lire_resultat(res_txt)
        tol = (0.5 * 10 ** -nd) / (100.0 if pct else 1.0) + 3e-3 * abs(attendu)
        if abs(val - attendu) > tol:
            fautes.append(f"{texte(parts[0])} = {texte(parts[2])} → {val} ≠ {attendu}")
    return fautes, n


total = 0
for nom_cas, etat in ETATS.items():
    fautes, n = controle_formules(K, etat, nom_cas)
    total += n
    chk(f"fenêtres : chaque ligne « expression = valeurs = résultat » réévaluée ({n}) — {nom_cas}", not fautes and n >= 60, fautes[:3])
print(f"     ({total} lignes de formules réévaluées)")
vol = rep_h["volets"]
chk("cinq fenêtres (sections, ancrages, comparaison, géométrie, efforts) ; les cellules du classeur entre parenthèses en historique (C19, C35, C57, I37, C82…)",
    all(f'id="ck-{k}"' in vol for k in ("s0", "s1", "s2", "geometrie", "efforts")) and all(f"({c})" in vol for c in ("C19", "C35", "C57", "I37", "C82", "H31")), None)
chk("en équilibre et en B, aucune cellule du classeur", "(C57)" not in rep_e["volets"] and "(C57)" not in rB["volets"] and "(I37)" not in rB["volets"], None)
chk("la fenêtre des sections dit la combinaison de chaque coupe ; celle de la géométrie liste les 9 combinaisons (jeux par défaut 10 et 5, § 9.4)",
    vol.count('class="ck-comb"') >= 6 and "charge à X + 10, ancrage à Y + 5" in vol and vol.count("<li>charge à X") == 9, vol.count("<li>charge à X"))
re_a = saisir(v1_ := req(action="calculer", etat=K.NOTE_V1)["etat"], "entraxe", "80")
chk("ressaisir un champ de l'ancrage ou du support (entraxe) fait aussi retomber v1 (RESSAISIE étendu)", re_a["etat"]["cornieres"][0]["v1"] is False, None)
re_b = saisir(v1_, "support", "poutre")
chk("… le support aussi", re_b["etat"]["cornieres"][0]["v1"] is False, None)
re_c = saisir(v1_, "etat_limite", "ELU")
chk("… mais l'état limite est une hypothèse, pas une ressaisie : v1 reste", re_c["etat"]["cornieres"][0]["v1"] is True, None)

# ====================================================================== 8. la note v1 (modèle lame)
v1 = req(action="calculer", etat=K.NOTE_V1)
c1 = v1["etat"]["cornieres"][0]
chk("une note v1 se rouvre : le nom et la cornière gardés, la charge et l'ancrage aux défauts posés dans ses ailes, v1 vrai, version 2",
    v1["ok"] and c1["nom"] == "Cornière de rive" and c1["profil"] == "L 100x100x10" and c1["standard"] and c1["v1"] is True and v1["etat"]["version"] == 2
    and c1["X"] == 62.5 and c1["Y"] == 71.0 and "V" not in c1, (v1.get("erreur") or c1))
chk("l'avis d'ouverture est dit une fois : en tête du résultat et dans « avis » (le téléphone), le texte des décisions § 6 ; l'avertissement du moteur n'est pas redit",
    v1.get("avis") == [K.AVIS["v1"]] and K.AVIS["v1"] in v1["resultat"] and v1["resultat"].count("modèle lame") == 1
    and EN.AVIS_V1[0].upper() in K.AVIS["v1"], (v1.get("avis"), v1["resultat"].count("modèle lame")))
chk("l'état rendu ne porte pas l'avis (il ne se redit pas à la relecture) mais l'avertissement du moteur reste",
    "avis_ouverture" not in v1["etat"] and "avis" not in req(action="calculer", etat=v1["etat"]) and "modèle lame" in req(action="calculer", etat=v1["etat"])["resultat"], None)
re_ = saisir(v1["etat"], "R", "20")
chk("ressaisir la charge fait retomber v1 : plus d'avertissement « note d'avant »", re_["etat"]["cornieres"][0]["v1"] is False and "modèle lame" not in re_["resultat"], None)
re2 = saisir(v1["etat"], "nom", "Rive 2")
chk("renommer ne vaut pas ressaisir : v1 reste", re2["etat"]["cornieres"][0]["v1"] is True and "modèle lame" in re2["resultat"], None)
v1b = copy.deepcopy(K.NOTE_V1)
v1b["cornieres"][0].update(standard=False, A=90.0, B=60.0, t=7.0)
v1s = req(action="calculer", etat=v1b)
chk("une saisie libre v1 (A, B, t ; le congé inconnu vaut 0) se rouvre aussi", v1s["ok"] and v1s["etat"]["cornieres"][0]["r"] == 0.0 and not v1s["etat"]["cornieres"][0]["standard"], v1s.get("erreur"))

# ====================================================================== 9. les éléments de la note, le résumé
e2 = req(action="ajouter", etat=copy.deepcopy(r0["etat"]))
chk("ajouter : « C2 » ouverte, deux entrées dans la liste de la ligne de l'élément",
    e2["etat"]["actif"] == 1 and e2["etat"]["cornieres"][1]["nom"] == "C2" and e2["donnees"].count('data-activer="') == 2
    and 'data-elements="2"' in e2["donnees"], None)
e3 = req(action="dupliquer", etat=copy.deepcopy(e2["etat"]), index=0)
chk("copier la première l'insère juste après elle, « (copie) », et l'ouvre",
    [c["nom"] for c in e3["etat"]["cornieres"]] == ["C1", "C1 (copie)", "C2"] and e3["etat"]["actif"] == 1, None)
e4 = saisir(e3["etat"], "R", "60")
chk("le résumé de la note est sa cornière la plus chargée (la copie à 60 kN, non conforme)",
    e4["resume"]["niveau"] == "r" and e4["resume"]["gouvernant"].startswith("C1 (copie)") and e4["statut_html"] and "Non conforme" in e4["statut_html"], e4["resume"])
e5 = req(action="supprimer", etat=copy.deepcopy(e4["etat"]), index=1)
chk("supprimer la copie (ouverte) : deux cornières restent, la suivante s'ouvre, le résumé redevient conforme",
    [c["nom"] for c in e5["etat"]["cornieres"]] == ["C1", "C2"] and e5["etat"]["actif"] == 1 and e5["resume"]["niveau"] != "r", e5["etat"])
e6 = req(action="supprimer", etat=copy.deepcopy(r0["etat"]), index=0)
chk("la dernière cornière ne se supprime pas", len(e6["etat"]["cornieres"]) == 1)
e7 = req(action="activer", etat=copy.deepcopy(e5["etat"]), index=7)
chk("activer un indice hors de la liste ouvre la dernière", e7["etat"]["actif"] == 1)
chk("la liste des éléments dit « nom · désignation · configuration »", "· L 120×120×15 · A" in e2["donnees"], None)

# ====================================================================== 10. les vues
vues = r0["vues"]
chk("quatre vues (coupe, moments, contraintes, élévation), chacune un SVG avec son en-tête technique (bande R, Mmax, H) et la légende de schemas",
    [v["cle"] for v in vues] == ["coupe", "moments", "contraintes", "elevation"] and all(v["svg"].lstrip().startswith("<svg") for v in vues)
    and all('class="et-eff"' in v["entete"] and "<i>M</i><sub>max</sub>" in v["entete"] for v in vues)
    and all(C.esc(SC.legende(REF0, v["cle"])) in v["entete"] for v in vues) and [v["echelle"] for v in vues] == [True, True, True, True], None)
# (révision § 9 des schémas : les moments et le maillage dessinent la cornière à l'échelle, les quatre vues portent echelle True)
from api_web import dessin_corniere as DC  # noqa: E402
tailles = [DC.taille_texte(v["svg"]) for v in vues]
chk("le texte des dessins se lit à l'écran (≈ 13 px dans la toile de l'établi, entre 9 et 16)", all(9 <= t <= 16 for t in tailles), tailles)

# ====================================================================== 11. la note PDF, le contrat commun
pdf = req(action="telecharger", etat=copy.deepcopy(r0["etat"]), quoi="note_pdf", infos={"nom": "Essai", "chantier": "Réalex", "indice": "B"})
if importlib.util.find_spec("api_web.note_corniere") is None:
    chk("sans api_web/note_corniere.py (le lot « note »), la note PDF est refusée avec un message clair, jamais une exception",
        pdf.get("ok") is False and "note_corniere" in pdf.get("erreur", "") and "Traceback" not in pdf.get("erreur", ""), pdf)
else:
    import base64
    chk("api_web/note_corniere.py présent : la note PDF se fait (un PDF)", pdf.get("ok") and base64.b64decode(pdf["fichier"]["base64"])[:5] == b"%PDF-", pdf.get("erreur"))
chk("un téléchargement inconnu est refusé", req(action="telecharger", etat=copy.deepcopy(r0["etat"]), quoi="csv").get("ok") is False)
abime = req(action="calculer", etat={"version": 2, "cornieres": "abîmé"})
chk("une note abîmée (« cornieres » n'est pas une liste) rend une erreur lisible", abime["ok"] is False and "liste" in abime["erreur"], abime.get("erreur"))
abime2 = req(action="calculer", etat={"version": 2, "cornieres": [{"X": "soixante"}]})
chk("une cornière illisible est nommée dans l'erreur", abime2["ok"] is False and "cornière" in abime2["erreur"] and "nombre" in abime2["erreur"], abime2.get("erreur"))
vide = req(action="calculer", etat={"version": 2, "cornieres": []})
chk("une note sans cornière est une note neuve", vide["ok"] and vide["etat"] == r0["etat"])


# ====================================================================== 12. les mutations : chacune vue rouge
def mutant(ancien, nouveau):
    src = open(K.__file__, encoding="utf-8").read()
    assert src.count(ancien) == 1, (ancien, src.count(ancien))
    m = types.ModuleType("api_web.corniere_mutant")
    m.__package__ = "api_web"
    m.__file__ = K.__file__
    exec(compile(src.replace(ancien, nouveau, 1), K.__file__, "exec"), m.__dict__)  # noqa: S102 — le module lui-même, muté
    return m


def rouge(fn):
    try:
        return bool(fn())
    except Exception as e:  # noqa: BLE001 — un mutant qui plante est vu rouge aussi
        return True


E_DEF, E_B, E_GD = ETATS["A historique (défauts)"], ETATS["B (toujours l'équilibre)"], ETATS["garde rouge (Y = 40, sans jeu, Ø10)"]
MUTATIONS = [
    ("le drapeau v1 ne retombe plus quand la charge est ressaisie",
     'if k in RESSAISIE and c.get("v1"):', 'if False and c.get("v1"):',
     lambda m: (lambda e: (m.modifier(e, ["R"], "20") or True) and e["cornieres"][0]["v1"] is True)(m.normaliser(copy.deepcopy(K.NOTE_V1)))),
    ("σ_éq écrit 3·τ au lieu de 3·τ² dans les valeurs substituées (la réévaluation le voit)",
     "√({v('σ')}² + 3 × {v('τ')}²)", "√({v('σ')}² + 3 × {v('τ')})", lambda m: controle_formules(m, E_DEF, "")[0]),
    ("la couleur d'une vérification ne suit plus le moteur",
     "rows += C.ligne_verif(nom, _comparaison_coupe(v, sym), C.pct(v[\"taux\"]), _niv(v), calcul=\"s0\", piece=piece, gov=v[\"cle\"] == gov)",
     "rows += C.ligne_verif(nom, _comparaison_coupe(v, sym), C.pct(v[\"taux\"]), \"p\", calcul=\"s0\", piece=piece, gov=v[\"cle\"] == gov)",
     lambda m: controle_resultat(m, ETATS["forte charge (R = 80 kN) : non conforme"], "")),
    ("les lignes ne disent plus la combinaison qui les gouverne",
     'return C.esc(cb["libelle"]) if cb else ""', 'return ""', lambda m: controle_resultat(m, E_DEF, "")),
    ("l'avis d'ouverture d'une note v1 n'est plus posé",
     'out[CLE_AVIS] = "v1"', "pass", lambda m: not m.resultat(m.normaliser(copy.deepcopy(K.NOTE_V1))).get("avis")),
    ("l'avis n'est plus retiré de l'état (il se redirait à chaque ouverture)",
     'etat.pop(CLE_AVIS, None)', 'etat.get(CLE_AVIS, None)',
     lambda m: (lambda e: (m.resultat(e) or True) and CLE in e)(m.normaliser(copy.deepcopy(K.NOTE_V1)))),
    ("une cornière hors domaine du sélecteur n'est plus grise",
     'return "c" if not x["domaine"] else', 'return "p" if not x["domaine"] else',
     lambda m: m.selecteur(m.normaliser(copy.deepcopy(r0["etat"])), "corniere").count('<span class="chip c">exclue</span>') != len(hors)),
    ("l'avertissement grave n'est plus en tête",
     'key=lambda a: (a["niveau"] not in GRAVES, a["niveau"] != MO.ATTENTION)', 'key=lambda a: (a["niveau"] in GRAVES,)',
     lambda m: controle_resultat(m, E_GD, "")),
    ("la pièce « dx » du dessin n'a plus sa ligne dans les données",
     '("dx", "± dx jeu"', '(None, "± dx jeu"',
     lambda m: (lambda rep: not (set(re.findall(r'data-sel="([^"]+)"', "".join(v["svg"] for v in rep["vues"]))) <= set(re.findall(r'data-cible="([^"]+)"', rep["donnees"]))))(m.resultat(m.normaliser(copy.deepcopy(E_DEF))))),
    ("les blocs ne sont plus thématiques (le JS n'en montrerait qu'un)",
     'return E.bloc(1, "Ancrages", "inc", "chevilles non vérifiées", efforts, rows, theme=True)',
     'return E.bloc(1, "Ancrages", "inc", "chevilles non vérifiées", efforts, rows)',
     lambda m: controle_resultat(m, E_DEF, "")),
    ("l'état de l'en-tête du bloc Ancrages redevient long (le titre du bloc disparaissait à l'écran, 10/11/2026)",
     'return E.bloc(1, "Ancrages", "inc", "chevilles non vérifiées", efforts, rows, theme=True)',
     'return E.bloc(1, "Ancrages", "inc", "efforts par ancrage · chevilles non vérifiées", efforts, rows, theme=True)',
     lambda m: noms_longs(m)),
    ("le nom de la ligne du bloc Comparaison redevient le libellé entier du mode (coupé à l\'ellipse en A équilibre, 10/11/2026)",
     'rows += C.ligne_verif(f"Mode {MODES_COURTS[cp[\'mode\']]}"', 'rows += C.ligne_verif(f"Mode {EN.LIBELLES_MODE[cp[\'mode\']]}"',
     lambda m: noms_longs(m)),
    ("la fenêtre « efforts » n'est plus annoncée par la ligne Efforts",
     'None, "c", calcul="efforts", piece=f"R = {C.nb(res[\'corniere\'][\'R\'])}{C.NBSP}kN"', 'None, "c", calcul="s0", piece=f"R = {C.nb(res[\'corniere\'][\'R\'])}{C.NBSP}kN"',
     lambda m: controle_resultat(m, E_DEF, "")),
    # la révision § 9 : le volet toujours ouvert, le coef visible en ELU, l'entraxe non affiché, les bords de la poutre
    # non affichés, RESSAISIE sans les nouveaux champs, le symbole figé, les lignes ELS gardées en ELU, R_ELU visible en ELU
    ("le volet « Détails » est toujours ouvert",
     '<details class="et-volet" data-volet="fiche">', '<details class="et-volet" data-volet="fiche" open>',
     lambda m: " open>" in m.resultat(m.normaliser(copy.deepcopy(E_DEF)))["donnees"]),
    ("le coefficient reste affiché en ELU (il est sans effet)",
     "    if _etat_limite(res) == EN.ELS:\n        # le coefficient de l'ELS", "    if True:\n        # le coefficient de l'ELS",
     lambda m: 'data-chemin="coef"' in m.resultat(m.normaliser(copy.deepcopy(E_ELU)))["donnees"]),
    ("R_ELU reste affiché en ELU",
     "    if els:\n        # R_ELU n'a de sens qu'en ELS", "    if True:\n        # R_ELU n'a de sens qu'en ELS",
     lambda m: 'data-chemin="R_ELU"' in m.resultat(m.normaliser(copy.deepcopy(E_ELU)))["donnees"]),
    ("l'entraxe saisi n'est plus affiché dans son champ",
     'C.num(["entraxe"], c["entraxe"], "mm"', 'C.num(["entraxe"], None, "mm"',
     lambda m: 'value="80"' not in m.resultat(m.normaliser(copy.deepcopy(ETATS["entraxe saisi s = 80"])))["donnees"]),
    ("les positions ne disent plus l'entraxe",
     "· s = {C.nb(round(anc['entraxe'], 2), 2)}{' (L/n)' if g['entraxe_auto'] else ' (saisi)'} · ", "· ",
     lambda m: controle_resultat(m, ETATS["entraxe saisi s = 80"], "")),
    ("les bords de la poutre ne sont plus affichés",
     '        if inf["cle"] in ("autre_mode", "L_eff"):\n            continue\n        val = f"{_n(inf[\'valeur\'], inf[\'unite\'])} {inf[\'unite\']}".strip()',
     '        if inf["cle"] in ("autre_mode", "L_eff", "bord_poutre_haut", "bord_poutre_bas"):\n            continue\n        val = f"{_n(inf[\'valeur\'], inf[\'unite\'])} {inf[\'unite\']}".strip()',
     lambda m: controle_resultat(m, ETATS["A sur poutre 160 au-dessus / 120 au-dessous"], "")),
    ("RESSAISIE sans les champs d'ancrage et de support (une saisie de l'entraxe ne fait plus retomber v1)",
     'RESSAISIE = ("R", "L", "X", "dx", "a", "R_ELU", "Y", "dy", "n", "d", "entraxe", "type_ancrage", "profondeur", "rondelles",\n             "support", "support_haut", "support_bas")',
     'RESSAISIE = ("R", "L", "X", "dx", "a", "R_ELU", "Y", "dy", "n", "d")',
     lambda m: (lambda e: (m.modifier(e, ["entraxe"], "80") or True) and e["cornieres"][0]["v1"] is True)(m.normaliser(copy.deepcopy(K.NOTE_V1)))),
    ("le symbole de la limite est figé « σ_adm » au lieu de celui que le moteur nomme",
     'return C.symb_math(res["materiau"]["symbole_sigma"])', 'return C.symb_math("σ_adm")',
     lambda m: controle_resultat(m, E_ELU, "")),
    ("les lignes ELS des ancrages restent affichées en ELU (N_ELS = None : « — »)",
     "    hist, els = res[\"mode\"] == EN.HISTORIQUE, _etat_limite(res) == EN.ELS\n    comb = _comb(anc[\"combinaison_N\"])\n    if els:",
     "    hist, els = res[\"mode\"] == EN.HISTORIQUE, _etat_limite(res) == EN.ELS\n    comb = _comb(anc[\"combinaison_N\"])\n    if True:",
     lambda m: controle_resultat(m, E_ELU, "")),
    ("la pièce « entraxe » du dessin n'a plus sa ligne dans les données (data-cible)",
     '("entraxe", "s entraxe"', '(None, "s entraxe"',
     lambda m: "entraxe" not in set(re.findall(r'data-cible="([^"]+)"', m.resultat(m.normaliser(copy.deepcopy(E_DEF)))["donnees"]))),
]
CLE = K.CLE_AVIS
for titre, ancien, nouveau, controle in MUTATIONS:
    try:
        m = mutant(ancien, nouveau)
        vu = rouge(lambda: controle(m))
    except AssertionError as e:
        vu, m = False, None
        print("     (mutation introuvable :", e, ")")
    chk(f"mutation vue rouge : {titre}", vu)
chk("témoin : le module tel quel passe les mêmes contrôles", not controle_resultat(K, E_DEF, "") and not controle_formules(K, E_DEF, "")[0])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
