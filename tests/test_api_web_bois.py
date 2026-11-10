# -*- coding: utf-8 -*-
"""L'API du site pour les poutres bois refaites sur l'Eurocode 5 (api_web/bois.py, lot Q du 10/11/2026), vérifiée en
Python ordinaire : CE QUE LE SITE AFFICHE = CE QUE LE MOTEUR CALCULE.

  1. LA NOTE NEUVE : version 2, une poutre, quatre rubriques (la dernière secondaire), CONFORME, ★ sur une section plus
     légère ; la réponse complète.
  2. LE RÉSULTAT FACE AU MOTEUR, sur 17 états (la note neuve, G seule qui gouverne l'ELU puis toute la note, le
     déversement, des entretoises, un lamellé-collé, l'appui qui gouverne, la classe de service 3, sur mesure,
     surfacique, sans appui, sans Q, contre-flèche, effet système, k_c,90 ramené à 1, près de 100 %, très au-delà) :
     chaque nombre RELU dans le HTML
     (le verdict, chaque ligne et sa comparaison, les combinaisons, les en-têtes des blocs, le pied) est celui du
     moteur à l'arrondi affiché ; la couleur, la ligne qui gouverne, la combinaison nommée, les avertissements, les
     hypothèses, ce qui n'est pas couvert, le résumé de la note.
  3. CHAQUE CHAMP RELIÉ : modifier → l'état (au bon endroit) → le calcul ; les effets de bord (standard ↔ sur mesure,
     le bois qui change de type, la nature de Q et ses verrous, les champs qui apparaissent).
  4. LES REFUS : le message du moteur (français, le champ nommé), la poutre intacte.
  5. LE SÉLECTEUR : les sections du type du bois par masse, ✓, ★, rouges quand elles ne tiennent pas ; le choix.
  6. UNE ANCIENNE NOTE (version 1) : les noms gardés, des poutres neuves, l'avis dit une fois et retiré de l'état.
  7. L'ÉCHAPPEMENT d'un nom piégé dans tous les fragments.
  8. LES FENÊTRES DE CALCUL : sept, chaque ligne « symbole = expression = valeurs = résultat » RÉÉVALUÉE à partir des
     valeurs écrites (un interpréteur du test) ; le tableau par combinaison ; les pastilles d'hypothèses ; le pied.
  9. LES VUES : trois, leur en-tête technique (la bande M_d, V_d, R_d du moteur), les cotes et les valeurs des dessins.
 10. LES ÉLÉMENTS de la note, son résumé.
 11. Des MUTATIONS du module, chacune vue rouge par ces contrôles.

    /usr/bin/python3.12 tests/test_api_web_bois.py
"""
import ast
import copy
import html
from html.parser import HTMLParser
import json
import math
import os
import re
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import bois as B  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import routeur as R  # noqa: E402
from bois.poutre import donnees as D  # noqa: E402
from bois.poutre import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="bois", **k))))


def saisir(etat, chemin, valeur):
    out = req(action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def etat_avec(**champs):
    """Une note neuve dont la poutre reçoit ``champs`` (relue par le moteur : hors domaine, ValueError)."""
    e = B._note(**champs)
    p = e["poutres"][0]
    if p["standard"] and ("section" in champs or "classe" in champs):
        p["b"] = p["h"] = None
    if "charges__nature" in champs and champs["charges__nature"] != "autre":
        p["charges"]["duree"] = p["charges"]["psi2"] = None
    return B.normaliser(e)


def V(res, cle):
    return next(v for v in res["verifications"] if v["cle"] == cle)


# ====================================================================== lecture du HTML rendu
ESPACES = (C.NBSP, C.FINE)
NOMBRE = r"-?\d[\d ]*(?:,\d+)?"
LIGNE = re.compile(r'<div class="el vc(?P<cls>[^"]*)"(?: data-calcul="(?P<calcul>[^"]*)" title="[^"]*")?><div class="l1"><b>(?P<nom>[^<]*)</b>(?P<l1>.*?)</div>'
                   r'<div class="l2"><span class="cmp">(?P<cmp>.*?)</span>(?P<jauge>(?:<div class="bar">.*?</div>)?)</div></div>', re.S)
PCT = re.compile(r'<span class="pct (\w)"(?: title="[^"]*")?>([^<]+)</span>')
MOT = re.compile(r'<span class="et-mot (\w)">([^<]+)</span>')
PIECE = re.compile(r'<(?:button|span) class="prof"[^>]*>(?:<span class="leger" title="(?P<leger>[^"]*)">.*?</span>)?<span>(?P<piece>.*?)</span>', re.S)
BLOC = re.compile(r'<section class="et-bloc (\w+)" data-theme="(\d)"><div class="et-bloc-h"><button[^>]*>.*?<span class="n">([^<]*)</span><span class="e">([^<]*)</span>'
                  r'.*?<div class="et-bloc-ef">(.*?)</div><div class="els">', re.S)
MSG = re.compile(r'<div class="msg( grave)?">(.*?)</div>', re.S)
FX = re.compile(r'<div class="fx">(.*?)</div>', re.S)
SPAN = re.compile(r'<span(?: class="(s|r|chip \w+)")?>(.*?)</span>', re.S)
VERDICT = re.compile(r'<div class="verdict" data-calcul="([^"]+)"[^>]*><span class="chip xl (\w)"(?: title="[^"]*")?><small>η</small>([^<]+)</span><div class="gov"><b>([^<]*)</b>(.*?)</div></div>', re.S)


def texte(frag):
    return html.unescape(re.sub(r"<[^>]+>", " ", frag))


def sans_indices(frag):
    """Un fragment sans ses indices (« σ<sub>c,90,d</sub> » porte un 90 qui n'est pas une valeur) ni ses balises."""
    return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<sub>.*?</sub>", "", frag)))


def lire_fr(s):
    """« 14,77 » → (14.77, 2 décimales)."""
    t = s.replace(C.NBSP, "").replace(C.FINE, "").replace(" ", "").replace("−", "-")
    return float(t.replace(",", ".")), len(t.partition(",")[2])


def affiche(texte_nombre, exact):
    """Le nombre écrit est la valeur exacte à l'arrondi de ses décimales (demi vers l'extérieur ou non)."""
    val, nd = lire_fr(texte_nombre)
    return abs(val - exact) <= 0.5 * 10 ** -nd + 1e-9


def pct_txt(t):
    return C._taux_txt(C.pct(t))[0]


def lignes(frag):
    out = []
    for m in LIGNE.finditer(frag):
        p = PIECE.search(m.group("l1"))
        t = PCT.search(m.group("l1"))
        w = MOT.search(m.group("l1"))
        out.append({"nom": html.unescape(m.group("nom")), "calcul": m.group("calcul"), "piece": texte(p.group("piece")).strip() if p else "",
                    "leger": html.unescape(p.group("leger") or "") if p else "", "pk": 'data-pk="section"' in m.group("l1"),
                    "taux_txt": t.group(2) if t else None, "niveau": t.group(1) if t else (w.group(1) if w else None),
                    "mot": w.group(2) if w else None, "cmp": m.group("cmp"), "gov": " gov" in m.group("cls"),
                    "largeur": re.search(r'style="width:([0-9.]+)%"', m.group("jauge")).group(1) if m.group("jauge") else None})
    return out


def blocs(frag):
    return [{"cls": c, "theme": int(i), "titre": html.unescape(t), "etat": html.unescape(e), "efforts": ef} for c, i, t, e, ef in BLOC.findall(frag)]


def resultat_module(mod, etat):
    """Le résultat rendu par un module (le vrai ou un mutant), l'état relu comme le routeur le fait."""
    return mod.resultat(mod.normaliser(copy.deepcopy(etat)))


COMPARAISON_ELU = re.compile(rf"= ({NOMBRE}) (≤|>) (?:[^=]*= )?({NOMBRE}) N/mm²")
COMPARAISON_ELS = re.compile(rf"= ({NOMBRE})(?: \(L/({NOMBRE})\))? (≤|>) L/(\d+) = ({NOMBRE}) mm")
NOMS_COURTS = {"flexion": "Flexion", "cisaillement": "Cisaillement", "appui": "Appui", "fleche_inst": "Flèche instantanée", "fleche_fin": "Flèche finale nette"}


def fautes_comparaison(v, r, cmp_html, cas):
    """La comparaison écrite d'une vérification face au moteur : la valeur, le signe, la limite (et « L/n »)."""
    f, t = [], sans_indices(cmp_html)
    signe = "≤" if v["taux"] <= 1 else ">"
    if v["groupe"] == "ELU":
        m = COMPARAISON_ELU.search(t)
        if not m:
            return [f"{cas} {v['cle']} : comparaison illisible « {t} »"]
        if not affiche(m.group(1), v["valeur"]) or not affiche(m.group(3), v["limite"]) or m.group(2) != signe:
            f.append(f"{cas} {v['cle']} : « {t} » ≠ {v['valeur']} {signe} {v['limite']}")
    else:
        m = COMPARAISON_ELS.search(t)
        if not m:
            return [f"{cas} {v['cle']} : comparaison illisible « {t} »"]
        fl = r["fleches"]
        n, rapport = (fl["n_inst"], fl["rapport_inst"]) if v["cle"] == "fleche_inst" else (fl["n_fin"], fl["rapport_fin"])
        if not affiche(m.group(1), v["valeur"]) or not affiche(m.group(5), v["limite"]) or m.group(3) != signe or int(m.group(4)) != n:
            f.append(f"{cas} {v['cle']} : « {t} » ≠ {v['valeur']} {signe} L/{n} = {v['limite']}")
        if (m.group(2) is None) != (rapport is None) or (rapport is not None and not affiche(m.group(2), rapport)):
            f.append(f"{cas} {v['cle']} : L/n « {m.group(2)} » ≠ {rapport}")
    return f


def controle_resultat(mod, etat, cas):
    """Tout ce que le résultat rendu par ``mod`` a de différent du moteur, pour l'état ``etat``."""
    f = []
    en = mod.normaliser(copy.deepcopy(etat))
    p = en["poutres"][en["actif"]]
    r = MO.verifier(p)
    rep = mod.resultat(en)
    res = rep["resultat"]
    # ---- la tête, le verdict
    tete = re.search(r'<div class="et-rp"><b>(.*?)</b><span>(.*?)</span></div>', res)
    attendu_tete = f"{r['geometrie']['designation']} · {p['classe']} · L {C.fr(p['L'], 2)}{C.NBSP}m"
    if not tete or html.unescape(tete.group(1)) != (p["nom"] or "Poutre") or html.unescape(tete.group(2)) != attendu_tete:
        f.append(f"{cas} : tête « {tete and tete.groups()} » ≠ « {attendu_tete} »")
    vd = VERDICT.search(res)
    g = V(r, r["gouvernant"]["cle"])
    niv_note = "r" if r["statut"] == "nok" else C.niveau(r["taux_max"])
    if not vd:
        return f + [f"{cas} : pas de verdict"]
    if vd.group(1) != g["cle"] or vd.group(2) != niv_note or html.unescape(vd.group(3)) != html.unescape(pct_txt(r["taux_max"])) or html.unescape(vd.group(4)) != NOMS_COURTS[g["cle"]]:
        f.append(f"{cas} : verdict {vd.groups()[:4]} ≠ ({g['cle']}, {niv_note}, {pct_txt(r['taux_max'])}, {NOMS_COURTS[g['cle']]})")
    f += fautes_comparaison(g, r, vd.group(5), cas + " verdict")
    comb = next((c for c in r["combinaisons"] if c["cle"] == g["combinaison"]), None)
    dit = texte(vd.group(5))
    if (comb and f"{comb['nom']} · {comb['libelle_duree']}" not in dit) or (comb is None and "ELS caractéristique" not in dit):
        f.append(f"{cas} : le verdict ne nomme pas sa combinaison ({dit})")
    # ---- les blocs et leurs lignes
    bl, ls = blocs(res), lignes(res)
    if [(b["theme"], b["titre"]) for b in bl] != [(0, "ELU · résistance"), (1, "ELS · flèches")]:
        f.append(f"{cas} : blocs {[(b['theme'], b['titre']) for b in bl]}")
        return f
    for b, groupe in zip(bl, ("ELU", "ELS")):
        vs = [v for v in r["verifications"] if v["groupe"] == groupe]
        t = max(v["taux"] for v in vs)
        nok = any(v["taux"] > 1 for v in vs)
        m = re.match(rf"(Conforme|Non conforme) · ({NOMBRE}) %$", b["etat"])
        if not m or (m.group(1) == "Non conforme") != nok or b["cls"] != ("nok" if nok else "ok") or not affiche(m.group(2), C.pct(t, 1)):
            f.append(f"{cas} : en-tête du bloc {groupe} « {b['cls']} · {b['etat']} » ≠ {100 * t}")
    # les combinaisons en tête du bloc ELU : k_mod, M_d, V_d
    cb = re.findall(rf'<span class="n" title="([^"]*)">([^<]*)</span><span>k<sub>mod</sub> ({NOMBRE})</span><span>M<sub>d</sub> ({NOMBRE}) kN·m</span>'
                    rf'<span>V<sub>d</sub> ({NOMBRE}) kN</span>', bl[0]["efforts"])
    if len(cb) != len(r["combinaisons"]) or any(html.unescape(x[1]) != c["nom"] or html.unescape(x[0]) != c["libelle_duree"] or not affiche(x[2], c["kmod"])
                                                 or not affiche(x[3], c["M_d"]) or not affiche(x[4], c["V_d"]) for x, c in zip(cb, r["combinaisons"])):
        f.append(f"{cas} : combinaisons affichées {cb} ≠ {[(c['nom'], c['kmod'], c['M_d'], c['V_d']) for c in r['combinaisons']]}")
    co, fl = r["coefficients"], r["fleches"]
    els = sans_indices(bl[1]["efforts"])
    m = re.search(rf"k = ({NOMBRE}) · ψ = ({NOMBRE})", els)
    if not m or not affiche(m.group(1), co["kdef"]) or not affiche(m.group(2), co["psi2"]) or (("contre-flèche" in els) != (fl["w_c"] > 0)):
        f.append(f"{cas} : en-tête ELS « {els} » ≠ k_def {co['kdef']}, ψ₂ {co['psi2']}")
    attendues = [v for v in r["verifications"]]
    verifs = [li for li in ls if li["taux_txt"] is not None]
    if [li["nom"] for li in verifs] != [NOMS_COURTS[v["cle"]] for v in attendues]:
        f.append(f"{cas} : lignes {[li['nom'] for li in verifs]}")
        return f
    for li, v in zip(verifs, attendues):
        niv = "r" if v["taux"] > 1 else C.niveau(v["taux"])
        if html.unescape(li["taux_txt"]) != html.unescape(pct_txt(v["taux"])) or li["niveau"] != niv or li["calcul"] != v["cle"]:
            f.append(f"{cas} {v['cle']} : {li['taux_txt']} {li['niveau']} ≠ {pct_txt(v['taux'])} {niv}")
        if li["gov"] != (v["cle"] == r["gouvernant"]["cle"]):
            f.append(f"{cas} {v['cle']} : ligne gouvernante mal marquée")
        if float(li["largeur"]) != max(0, min(C.pct(v["taux"]), 100)):
            f.append(f"{cas} {v['cle']} : jauge {li['largeur']}")
        f += fautes_comparaison(v, r, li["cmp"], cas)
        c = next((c for c in r["combinaisons"] if c["cle"] == v["combinaison"]), None)
        if c is not None and not texte(li["cmp"]).rstrip().endswith("· " + c["nom"]):
            f.append(f"{cas} {v['cle']} : la ligne ne nomme pas sa combinaison « {texte(li['cmp'])} » ({c['nom']})")
        if li["pk"] != (v["cle"] == "flexion"):
            f.append(f"{cas} {v['cle']} : sélecteur mal posé")
    fx = verifs[0]
    if fx["piece"] != r["geometrie"]["designation"]:
        f.append(f"{cas} : la pièce de la flexion « {fx['piece']} » n'est pas la section")
    if ("k crit" in texte(fx["cmp"])) != (r["deversement"]["kcrit"] < 1):
        f.append(f"{cas} : k_crit dit à tort (ou tu) sur la ligne de flexion")
    opt = MO.optimal(p)
    doit = bool(opt) and not (p["standard"] and opt["nom"] == p["section"]) and (opt["A"] < r["geometrie"]["A"] or r["statut"] == "nok")
    if bool(fx["leger"]) != doit or (doit and (opt["designation"] not in fx["leger"] or f"({C.pc(C.pct(opt['taux']))} %)" not in fx["leger"])):
        f.append(f"{cas} : étoile « {fx['leger']} » ≠ {opt and opt['designation']}")
    non_ver = [li for li in ls if li["mot"] == "non vérifié"]
    if bool(non_ver) != (not r["appui"]["verifie"]):
        f.append(f"{cas} : appui non vérifié mal dit")
    # ---- les avertissements, les hypothèses, ce qui n'est pas couvert, le pied
    msgs = [texte(m_[1]).replace("  ", " ") for m_ in MSG.findall(res)]
    for a in r["avertissements"]:
        cle_txt = re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", C.symb_math(a["message"]))))
        if not any(re.sub(r"\s+", "", m_) == cle_txt for m_ in msgs):
            f.append(f"{cas} : avertissement « {a['code']} » absent du résultat")
    if len(msgs) != len(r["avertissements"]):
        f.append(f"{cas} : {len(msgs)} message(s) pour {len(r['avertissements'])} avertissement(s)")
    plat = re.sub(r"\s+", "", texte(res))
    for h in r["hypotheses"]:
        if re.sub(r"\s+", "", texte(C.symb_math(h))) not in plat:
            f.append(f"{cas} : hypothèse absente « {h[:40]} »")
    if "Non couvert par ce module" not in res or any(C.symb_math(x) not in res for x in r["non_couvert"]):
        f.append(f"{cas} : ce qui n'est pas couvert n'est pas écrit sous le résultat")
    pied = re.search(rf'<div class="pfoot"><div class="mass"><span>Masse · ({NOMBRE}) m de pièce</span><b>({NOMBRE}) kg · ({NOMBRE}) kg/m</b></div>'
                     rf'<div class="mass"><span>Poids propre</span><b>({NOMBRE}) kN/m( \(non compté\))?</b></div>', res)
    if not pied or not affiche(pied.group(1), r["geometrie"]["longueur"] / 1000) or not affiche(pied.group(2), r["masse"]) \
            or not affiche(pied.group(3), r["masse_lineique"]) or not affiche(pied.group(4), r["charges"]["poids_propre"]) \
            or bool(pied.group(5)) != (not r["charges"]["poids_propre_compte"]):
        f.append(f"{cas} : pied « {pied and pied.groups()} » ≠ {r['masse']}, {r['masse_lineique']}, {r['charges']['poids_propre']}")
    # ---- le résumé de la note (une seule poutre)
    rs = rep["resume"]
    if len(en["poutres"]) == 1 and (rs["taux"] != C.pct(r["taux_max"]) or rs["niveau"] != niv_note or rs["statut"] != C.STATUTS[niv_note][1]
                                    or rs["gouvernant"] != f"{p['nom'] or 'Poutre'} · {NOMS_COURTS[g['cle']].lower()}"):
        f.append(f"{cas} : résumé {rs}")
    return f


# ====================================================================== 1. une note neuve
print("=== 1. La note neuve")
r0 = req(action="nouveau")
e0 = r0["etat"]
chk("une note neuve se calcule : version 2, une poutre « P1 » aux défauts de la conception, la réponse complète",
    r0["ok"] and e0 == {"version": 2, "poutres": [MO.nouvelle_poutre("P1")], "actif": 0}
    and all(k in r0 for k in ("resultat", "volets", "vues", "phrase", "statut_html", "resume", "donnees", "elements", "etat")), r0.get("erreur"))
REF0 = MO.verifier(e0["poutres"][0])
chk("elle s'ouvre CONFORME (56 %, la flèche finale nette gouverne), en vert pâle, et le résumé de la note le dit",
    r0["resume"] == {"taux": 56, "niveau": "p", "statut": "Conforme", "gouvernant": "P1 · flèche finale nette"} and REF0["statut"] == "ok", r0["resume"])
secs = re.findall(r'<div class="sec( open)?" data-sec="(\w+)"( data-secondaire="1")?><div class="sechd" data-replier="1"><span class="n">(\d)</span><span class="tt">([^<]*)</span>', r0["donnees"])
chk("quatre rubriques numérotées : 1 Poutre et 2 Charges marquées ouvertes (l'établi compact n'ouvre que la première à la première ouverture, comme pour "
    "les cornières), 3 Appuis et maintien, 4 Hypothèses (secondaire) repliées",
    [(bool(o), k, bool(s), n, html.unescape(t)) for o, k, s, n, t in secs]
    == [(True, "po", False, "1", "Poutre"), (True, "ch", False, "2", "Charges"), (False, "ap", False, "3", "Appuis et maintien"), (False, "hy", True, "4", "Hypothèses")], secs)
chk("la colonne de gauche est la colonne COMPACTE : la ligne de l'élément (nom, liste, copier, supprimer), puis les rubriques",
    '<div class="et-arbre et-arbre-c" data-poutre="0"><div class="et-elsel" data-elements="1">' in r0["donnees"] and 'class="et-elt-nom"' in r0["donnees"]
    and r0["elements"] == "" and r0["phrase"] == "" and r0["statut_html"] == "")
l0 = lignes(r0["resultat"])
chk("la ligne « Flexion » porte la section (elle ouvre le sélecteur) et l'étoile d'une section plus légère qui tient (38 × 235, 95 %)",
    l0[0]["nom"] == "Flexion" and l0[0]["piece"] == "75 × 225" and l0[0]["pk"] and l0[0]["leger"] == "La plus légère qui tient : 38 × 235 (95 %)", l0[0])
chk("cinq lignes, chacune ouvre SA fenêtre ; le verdict ouvre celle de la vérification qui gouverne ; le bouton de chaque bloc « tout le calcul »",
    [li["calcul"] for li in l0] == ["flexion", "cisaillement", "appui", "fleche_inst", "fleche_fin"] and 'class="verdict" data-calcul="fleche_fin"' in r0["resultat"]
    and all(f'class="et-tout" data-calcul="s{i}"' in r0["resultat"] for i in (0, 1)) and all(f'id="ck-{k}"' in r0["volets"] for k in ("flexion", "cisaillement", "appui", "fleche_inst", "fleche_fin", "s0", "s1")))

# ====================================================================== 2. le résultat face au moteur
print("\n=== 2. Le résultat face au moteur, sur des états variés")
ETATS = {
    "la note neuve": e0,
    "G seule gouverne (G = 3, Q = 0,5)": etat_avec(charges__G=3.0, charges__Q=0.5),
    "G seule gouverne toute la note (2,5 m, G = 7, Q = 1)": etat_avec(L=2.5, charges__G=7.0, charges__Q=1.0),
    "déversement (45 × 245 sur 5 m, aux appuis)": etat_avec(section="45x245", L=5.0, maintien__type="appuis"),
    "entretoises à 1,25 m": etat_avec(section="45x245", L=5.0, maintien__type="entretoises", maintien__entraxe=1.25),
    "lamellé-collé GL24h 120 × 400 sur 8 m": etat_avec(classe="GL24h", section="120x400", L=8.0, charges__G=3.0, charges__Q=4.0),
    "l'appui gouverne (a = 40 mm)": etat_avec(section="100x300", L=3.0, charges__G=8.0, charges__Q=12.0, appui__a=40.0),
    "classe de service 3": etat_avec(service=3),
    "sur mesure 62,5 × 187,5": etat_avec(standard=False, b=62.5, h=187.5),
    "surfacique 2 et 3 kN/m² à 0,60 m": etat_avec(charges__mode="surfacique", charges__g=2.0, charges__q=3.0, charges__entraxe=0.6),
    "sans vérification d'appui, sans poids propre": etat_avec(appui__verifier=False, charges__poids_propre=False),
    "sans charge variable": etat_avec(charges__Q=0.0),
    "contre-flèche de 20 mm (poutre cambrée)": etat_avec(contre_fleche=20.0),
    "effet système, entraxe 0,90 m": etat_avec(systeme=True, charges__mode="surfacique", charges__entraxe=0.9),
    "appuis rapprochés (k_c,90 = 1)": etat_avec(section="100x300", L=0.65),
    "près de 100 % (Q = 2,06 kN/m)": etat_avec(charges__Q=2.06),
    "très au-delà (12 m, 20 + 30 kN/m)": etat_avec(L=12.0, charges__G=20.0, charges__Q=30.0),
}
for nom_cas, etat in ETATS.items():
    fautes = controle_resultat(B, etat, nom_cas)
    chk(f"le résultat = le moteur — {nom_cas}", not fautes, fautes[:3])
rg = resultat_module(B, ETATS["G seule gouverne (G = 3, Q = 0,5)"])
lg = lignes(rg["resultat"])
chk("G seule qui gouverne l'ELU se LIT : la ligne de flexion finit par « · G seule », le bloc ELU est rouge (118,2 %) ; le verdict, lui, est la flèche "
    "finale (153 %, « ELS caractéristique »)",
    texte(lg[0]["cmp"]).rstrip().endswith("· G seule") and blocs(rg["resultat"])[0]["cls"] == "nok" and "118,2" in blocs(rg["resultat"])[0]["etat"]
    and "ELS caractéristique" in texte(VERDICT.search(rg["resultat"]).group(5)) and VERDICT.search(rg["resultat"]).group(1) == "fleche_fin", lg[0]["cmp"])
rg2 = resultat_module(B, ETATS["G seule gouverne toute la note (2,5 m, G = 7, Q = 1)"])
chk("G seule qui gouverne TOUTE la note (flexion, 106,4 %) : le verdict dit « G seule · permanent » et ouvre la fenêtre de la flexion",
    "G seule · permanent" in texte(VERDICT.search(rg2["resultat"]).group(5)) and VERDICT.search(rg2["resultat"]).group(1) == "flexion"
    and rg2["resume"] == {"taux": 106, "niveau": "r", "statut": "Non conforme", "gouvernant": "P1 · flexion"}, (texte(VERDICT.search(rg2["resultat"]).group(5)), rg2["resume"]))
rd = resultat_module(B, ETATS["déversement (45 × 245 sur 5 m, aux appuis)"])
chk("le déversement se lit sur la ligne de flexion (« k_crit 0,398 ») et dans la rubrique 3 (l_ef = 4 990 mm, λ_rel,m = 1,584)",
    "k crit  0,398" in texte(lignes(rd["resultat"])[0]["cmp"]) and f"4{C.FINE}990{C.NBSP}mm" in rd["donnees"] and "1,584" in rd["donnees"], texte(lignes(rd["resultat"])[0]["cmp"]))
rp = resultat_module(B, ETATS["près de 100 % (Q = 2,06 kN/m)"])
chk("près de 100 % : la bande verte foncée (95 à 100 %) — verdict « f », statut « Conforme, attention »",
    rp["resume"]["niveau"] == "f" and rp["resume"]["statut"] == "Conforme, attention" and 95 <= rp["resume"]["taux"] <= 100, rp["resume"])
rt = resultat_module(B, ETATS["très au-delà (12 m, 20 + 30 kN/m)"])
chk("un taux de plus de 999 % s'écrit « > 999 % » par les briques (la valeur exacte au survol), rien n'est plafonné par le module",
    "&gt; 999 %" in rt["resultat"] and rt["resume"]["taux"] > 999 and 'title="' in VERDICT.search(rt["resultat"]).group(0), rt["resume"])
r100 = resultat_module(B, etat_avec(charges__Q=2.14))
t100 = MO.verifier(B.normaliser(etat_avec(charges__Q=2.14))["poutres"][0])["taux_max"]
chk(f"un taux juste au-dessus de 100 % ({100 * t100:.2f} %) est ROUGE et s'écrit 101 % (C.pct garde la bande), jamais « 100 % »",
    1.0 < t100 < 1.005 and r100["resume"]["niveau"] == "r" and r100["resume"]["taux"] == 101 and "Non conforme" in blocs(r100["resultat"])[1]["etat"], (t100, r100["resume"]))

# ====================================================================== 3. chaque champ relié
print("\n=== 3. Chaque champ relié : modifier → l'état → le calcul")
P = lambda e: e["poutres"][e["actif"]]  # noqa: E731
CHAMPS = [  # (chemin, saisie, lecture dans l'état, valeur attendue, ce que le calcul doit montrer)
    ("L", "4,5", lambda p: p["L"], 4.5, lambda r: r["geometrie"]["L_mm"] == 4500.0),
    ("classe", "C30", lambda p: p["classe"], "C30", lambda r: r["materiau"]["fm_k"] == 30.0),
    ("charges/G", "1,2", lambda p: p["charges"]["G"], 1.2, lambda r: r["charges"]["G"] == 1.2),
    ("charges/Q", "1,5", lambda p: p["charges"]["Q"], 1.5, lambda r: r["charges"]["q"] == 1.5),
    ("charges/mode", "surfacique", lambda p: p["charges"]["mode"], "surfacique", lambda r: r["charges"]["entraxe"] == 0.4),
    ("charges/nature", "E", lambda p: (p["charges"]["nature"], p["charges"]["duree"], p["charges"]["psi2"]), ("E", "long", 0.8), lambda r: r["combinaisons"][1]["kmod"] == 0.7),
    ("charges/poids_propre", False, lambda p: p["charges"]["poids_propre"], False, lambda r: r["charges"]["g"] == 0.6),
    ("appui/a", "60", lambda p: p["appui"]["a"], 60.0, lambda r: r["appui"]["l_ef"] == 90.0),
    ("appui/debord", "50", lambda p: p["appui"]["debord"], 50.0, lambda r: r["appui"]["l_ef"] == 160.0),
    ("appui/verifier", False, lambda p: p["appui"]["verifier"], False, lambda r: len(r["verifications"]) == 4),
    ("maintien/type", "appuis", lambda p: p["maintien"]["type"], "appuis", lambda r: r["deversement"]["l_ef"] == 0.9 * 4000 + 450),
    ("service", "2", lambda p: p["service"], 2, lambda r: r["coefficients"]["kdef"] == 0.8),
    ("gamma_G", "1,2", lambda p: p["gamma_G"], 1.2, lambda r: abs(r["combinaisons"][0]["p_d"] - 1.2 * r["charges"]["g"]) < 1e-12),
    ("gamma_Q", "1,3", lambda p: p["gamma_Q"], 1.3, lambda r: abs(r["combinaisons"][1]["p_d"] - 1.35 * r["charges"]["g"] - 1.3 * 0.8) < 1e-12),
    ("systeme", True, lambda p: p["systeme"], True, lambda r: r["coefficients"]["ksys"] == 1.1),
    ("lim_inst", "500", lambda p: p["lim_inst"], 500, lambda r: r["fleches"]["lim_inst"] == 8.0),
    ("lim_fin", "350", lambda p: p["lim_fin"], 350, lambda r: abs(r["fleches"]["lim_fin"] - 4000 / 350) < 1e-9),
    ("contre_fleche", "5", lambda p: p["contre_fleche"], 5.0, lambda r: abs(r["fleches"]["w_net_fin"] - r["fleches"]["w_fin"] + 5) < 1e-9),
    ("nom", "Gîte du séjour", lambda p: p["nom"], "Gîte du séjour", lambda r: True),
]
mal = []
for chemin, saisie, lire, attendu, calcul in CHAMPS:
    out = saisir(e0, chemin, saisie)
    p = P(out["etat"])
    if out["message"] or lire(p) != attendu or not calcul(MO.verifier(p)) or controle_resultat(B, out["etat"], chemin):
        mal.append((chemin, out["message"], lire(p), controle_resultat(B, out["etat"], chemin)[:1]))
chk(f"{len(CHAMPS)} champs : la saisie arrive au bon endroit de l'état, le moteur recalcule, et l'écran rendu est encore celui du moteur", not mal, mal[:3])
# les champs qui n'existent que dans un certain état : sur mesure, surfacique, entretoises, nature « autre »
E_MES, E_SURF, E_ENT, E_AUTRE = etat_avec(standard=False), etat_avec(charges__mode="surfacique"), etat_avec(maintien__type="entretoises"), etat_avec(charges__nature="autre")
CHAMPS_D_UN_ETAT = [  # (l'état de départ, chemin, saisie, lecture dans l'état, valeur attendue, ce que le calcul doit montrer)
    (E_MES, "b", "62,5", lambda p: p["b"], 62.5, lambda r: r["geometrie"]["A"] == 62.5 * 225),
    (E_MES, "h", "187,5", lambda p: p["h"], 187.5, lambda r: r["geometrie"]["W"] == 75 * 187.5 ** 2 / 6),
    (E_SURF, "charges/g", "2,5", lambda p: p["charges"]["g"], 2.5, lambda r: abs(r["charges"]["G"] - 2.5 * 0.4) < 1e-12),
    (E_SURF, "charges/q", "3", lambda p: p["charges"]["q"], 3.0, lambda r: abs(r["charges"]["q"] - 3.0 * 0.4) < 1e-12),
    (E_SURF, "charges/entraxe", "0,6", lambda p: p["charges"]["entraxe"], 0.6, lambda r: abs(r["charges"]["G"] - 1.5 * 0.6) < 1e-12 and abs(r["charges"]["Q"] - 2.0 * 0.6) < 1e-12),
    (E_ENT, "maintien/entraxe", "2", lambda p: p["maintien"]["entraxe"], 2.0, lambda r: r["deversement"]["l_ef"] == 2000.0 + 2 * 225),
    (E_AUTRE, "charges/duree", "court", lambda p: p["charges"]["duree"], "court", lambda r: r["combinaisons"][1]["kmod"] == 0.9),
    (E_AUTRE, "charges/psi2", "0,6", lambda p: p["charges"]["psi2"], 0.6, lambda r: r["coefficients"]["psi2"] == 0.6),
]
mal = []
for depart, chemin, saisie, lire, attendu, calcul in CHAMPS_D_UN_ETAT:
    out = saisir(depart, chemin, saisie)
    p = P(out["etat"])
    if out["message"] or lire(p) != attendu or not calcul(MO.verifier(p)) or controle_resultat(B, out["etat"], chemin):
        mal.append((chemin, out["message"], lire(p), controle_resultat(B, out["etat"], chemin)[:1]))
chk(f"{len(CHAMPS_D_UN_ETAT)} champs de plus, ceux qui n'existent que dans un état (b et h sur mesure ; g_s, q_s et l'entraxe en kN/m² ; l'entraxe des entretoises ; "
    "la durée et ψ₂ de la nature « autre ») : reliés de même", not mal, mal[:3])
relies = {c[0] for c in CHAMPS} | {c[1] for c in CHAMPS_D_UN_ETAT} | {"standard"}
champs_donnees = set(re.findall(r'data-chemin="([^"]+)"', "".join(resultat_module(B, e)["donnees"] for e in (
    e0, etat_avec(standard=False), etat_avec(charges__mode="surfacique"), etat_avec(maintien__type="entretoises")))))
attendus = {"nom", "classe", "standard", "b", "h", "L", "charges/mode", "charges/G", "charges/Q", "charges/g", "charges/q", "charges/entraxe", "charges/nature",
            "charges/poids_propre", "charges/duree", "charges/psi2", "appui/a", "appui/debord", "appui/verifier", "maintien/type", "maintien/entraxe", "service",
            "gamma_G", "gamma_Q", "systeme", "lim_inst", "lim_fin", "contre_fleche"}
chk("le cadre Données offre TOUS les champs de la poutre (28 chemins, section standard par le sélecteur) et aucun autre ; chacun est dans une des deux "
    "tables ci-dessus (« standard » : plus bas)", champs_donnees == attendus and relies == attendus,
    (sorted(attendus - champs_donnees), sorted(champs_donnees - attendus), sorted(attendus ^ relies)))
# la saisie surfacique, les entretoises, la section sur mesure : leurs champs n'apparaissent que quand ils jouent
d0 = r0["donnees"]
d_surf = saisir(e0, "charges/mode", "surfacique")
chk("kN/m : G et Q ; kN/m² : g_s, q_s et l'entraxe, avec les charges linéaires qu'ils donnent (0,600 et 0,800 kN/m)",
    'data-chemin="charges/G"' in d0 and 'data-chemin="charges/g"' not in d0 and 'data-chemin="charges/entraxe"' in d_surf["donnees"]
    and 'data-chemin="charges/G"' not in d_surf["donnees"] and f"G = 0,600{C.NBSP}kN/m · Q = 0,800{C.NBSP}kN/m" in d_surf["donnees"])
d_ent = saisir(e0, "maintien/type", "entretoises")
chk("l'entraxe des entretoises n'apparaît que pour ce maintien ; la rubrique dit alors l_ef, λ_rel,m et k_crit",
    'data-chemin="maintien/entraxe"' in d_ent["donnees"] and 'data-chemin="maintien/entraxe"' not in d0 and "λ<sub>rel,m</sub>" in d_ent["donnees"]
    and "pas de déversement" in d0)
d_mes = saisir(e0, "standard", "mesure")
chk("« Sur mesure » part des dimensions de la section affichée (75 × 225) : b et h sur une ligne, plus de bouton de section",
    P(d_mes["etat"])["standard"] is False and (P(d_mes["etat"])["b"], P(d_mes["etat"])["h"]) == (75.0, 225.0) and 'class="row2 n2"' in d_mes["donnees"]
    and 'data-chemin="b"' in d_mes["donnees"] and 'class="bo-pk"' not in d_mes["donnees"] and 'class="bo-pk" data-pk="section"' in d0)
d_b = saisir(d_mes["etat"], "b", "80")
d_std = saisir(saisir(d_b["etat"], "h", "240")["etat"], "standard", "standard")
chk("retour à « Standard » depuis 80 × 240 : la section standard la plus proche (100 × 200), b et h la suivent",
    P(d_std["etat"])["standard"] is True and P(d_std["etat"])["section"] == "100x200" and (P(d_std["etat"])["b"], P(d_std["etat"])["h"]) == (100.0, 200.0), P(d_std["etat"]))
# le bois qui change de type
d_gl = saisir(e0, "classe", "GL24h")
pg = P(d_gl["etat"])
chk("du massif au lamellé-collé : la section 75 × 225 GARDE ses dimensions, en « sur mesure » (elle n'existe pas en lamellé) ; l'AVIS le dit une fois, "
    "en tête du résultat et dans « avis », et n'est pas gardé dans l'état",
    not d_gl["message"] and pg["classe"] == "GL24h" and pg["standard"] is False and (pg["b"], pg["h"]) == (75.0, 225.0) and pg["section"] == "80x200"
    and len(d_gl.get("avis", [])) == 1 and "75 × 225" in d_gl["avis"][0] and "lamellé-collé" in d_gl["avis"][0] and C.esc(d_gl["avis"][0]) in d_gl["resultat"]
    and B.CLE_AVIS not in d_gl["etat"] and "avis" not in req(action="calculer", etat=d_gl["etat"]), (pg, d_gl.get("avis")))
d_c30 = saisir(e0, "classe", "C30")
chk("dans le même type (C24 → C30), rien d'autre ne bouge et aucun avis n'est dit",
    P(d_c30["etat"])["standard"] is True and P(d_c30["etat"])["section"] == "75x225" and "avis" not in d_c30)
d_200 = saisir(etat_avec(section="100x200"), "classe", "GL28c")
chk("une section que les deux types proposent (100 × 200) reste standard quand le bois change de type, sans avis",
    P(d_200["etat"])["standard"] is True and P(d_200["etat"])["section"] == "100x200" and "avis" not in d_200, P(d_200["etat"]))
d_retour = saisir(d_gl["etat"], "classe", "C24")
chk("retour au massif depuis le lamellé sur mesure 75 × 225 : la section reste sur mesure (rien n'est deviné), la réserve redevient 75 × 225",
    P(d_retour["etat"])["standard"] is False and P(d_retour["etat"])["section"] == "75x225" and "avis" not in d_retour, P(d_retour["etat"]))
menu = re.search(r'<select data-chemin="classe" data-sorte="choix">(.*?)</select>', d0, re.S).group(1)
groupes = re.findall(r'<optgroup label="([^"]*)">(.*?)</optgroup>', menu, re.S)
chk("le menu « Bois » est groupé : d'abord les 12 classes du bois massif (EN 338), puis les 6 du lamellé-collé (EN 14080)",
    [html.unescape(g_[0]) for g_ in groupes] == ["Bois massif résineux · EN 338", "Lamellé-collé résineux · EN 14080"]
    and [re.findall(r'value="([^"]+)"', g_[1]) for g_ in groupes] == [[c for c in D.CLASSES if c.startswith("C")], [c for c in D.CLASSES if c.startswith("GL")]], groupes)
# la nature de Q et ses verrous
chk("nature « A · habitation » : la ligne discrète dit « moyen terme (1 semaine à 6 mois) · k_mod = 0,8 · ψ₂ = 0,3 » ; la durée et ψ₂ sont LUS et verrouillés",
    "moyen terme (1 semaine à 6 mois) · k<sub>mod</sub> = 0,8 · ψ<sub>2</sub> = 0,3" in d0
    and re.search(r'<select data-chemin="charges/duree" data-sorte="choix" disabled>', d0) and re.search(r'<input data-chemin="charges/psi2"[^>]*value="0,3"[^>]* disabled>', d0))
d_autre = saisir(e0, "charges/nature", "autre")
chk("nature « autre » : la durée et ψ₂ partent des derniers (moyen terme, 0,3) et se DÉVERROUILLENT",
    (P(d_autre["etat"])["charges"]["duree"], P(d_autre["etat"])["charges"]["psi2"]) == ("moyen", 0.3)
    and 'data-chemin="charges/duree" data-sorte="choix">' in d_autre["donnees"] and not re.search(r'<input data-chemin="charges/psi2"[^>]* disabled>', d_autre["donnees"])
    and "à saisir dans les Hypothèses" in d_autre["donnees"])
d_inst = saisir(saisir(d_autre["etat"], "charges/duree", "instantane")["etat"], "charges/psi2", "0,5")
chk("… et jouent : instantané → k_mod 1,1 ; ψ₂ = 0,5 ; revenir à « neige » les refixe (court terme, 0)",
    MO.verifier(P(d_inst["etat"]))["combinaisons"][1]["kmod"] == 1.1 and P(d_inst["etat"])["charges"]["psi2"] == 0.5
    and (P(saisir(d_inst["etat"], "charges/nature", "neige")["etat"])["charges"]["duree"], P(saisir(d_inst["etat"], "charges/nature", "neige")["etat"])["charges"]["psi2"]) == ("court", 0.0))
d_sans = saisir(e0, "appui/verifier", False)
ls_sans = lignes(d_sans["resultat"])
chk("l'appui retiré de la note : la ligne « Appui » reste, grise, « non vérifié », sans fenêtre ; l'avertissement le dit",
    [(li["nom"], li["mot"], li["calcul"]) for li in ls_sans if li["mot"]] == [("Appui", "non vérifié", None)] and "n'est pas vérifiée" in d_sans["resultat"]
    and 'id="ck-appui"' not in d_sans["volets"], [(li["nom"], li["mot"]) for li in ls_sans])
chk("la case du poids propre porte sa valeur (0,070 kN/m) ; la rubrique 1 dit A, W, I et la masse au mètre",
    f"Poids propre compté · 0,070{C.NBSP}kN/m" in d0 and f"A = 168,8{C.NBSP}cm² · W = 632,8{C.NBSP}cm³ · I = 7{C.FINE}119{C.NBSP}cm⁴ · 7,09{C.NBSP}kg/m" in d0)
aides = re.findall(r'<span class="aide" title="([^"]+)">', d0)
chk("les explications sont dans les ⓘ (au moins 14), dont l'hypothèse du maintien : « appuis … maintenus en torsion »",
    len(aides) >= 14 and any("maintenus en torsion" in html.unescape(a) for a in aides), len(aides))

# ====================================================================== 4. les refus
print("\n=== 4. Les refus : le message du moteur, la poutre intacte")
REFUS = [
    ("L", "0,2", ("portée L", "0,5 et 30 m", "0,2")), ("L", "quatre", ("Nombre illisible",)), ("L", "", ("Nombre illisible",)), ("L", "1e400", ("Nombre illisible",)),
    ("charges/Q", "-3", ("charge variable Q", "0 et 200")), ("charges/G", "300", ("charge permanente G",)), ("appui/a", "10", ("longueur d'appui a", "20 et 600")),
    ("appui/a", "4000", ("longueur d'appui",)), ("gamma_G", "0,9", ("γ_G", "1 et 1,5")), ("gamma_Q", "2,5", ("γ_Q",)), ("contre_fleche", "-1", ("contre-flèche",)),
    ("charges/psi2", "0,7", ("ψ₂", "habitation", "autre")), ("charges/duree", "court", ("durée", "court terme", "autre")), ("service", "4", ("classe de service",)),
    ("service", "1,5", ("Choix illisible",)), ("lim_inst", "275", ("flèche instantanée",)), ("classe", "D30", ("classe de résistance",)),
    ("charges/nature", "vent", ("nature",)), ("maintien/type", "libre", ("maintien",)), ("charges/mode", "ponctuelle", ("mode",)),
    ("systeme", "oui", ("vrai ou faux",)), ("b", "80", ("largeur b", "section standard")), ("section", "80x160", ("80 × 160", "bois massif")),
    ("standard", "peut-être", ("standard ou sur mesure",)), ("inconnu", "1", ("Champ inconnu",)), ("charges/inconnu", "1", ("Champ inconnu",)),
    ("nom", "x" * 61, ("Nom trop long", "61")),
]
mal = []
for chemin, valeur, mots in REFUS:
    out = req(action="modifier", etat=copy.deepcopy(e0), chemin=chemin.split("/"), valeur=valeur)
    if not out.get("ok") or not out.get("message") or out["etat"] != e0 or not all(m in out["message"] for m in mots) or re.search(r"\d\.\d", out["message"]):
        mal.append((chemin, valeur, out.get("message") or out.get("erreur")))
chk(f"{len(REFUS)} saisies hors domaine : chacune refusée avec un message qui nomme le champ (sans nombre « à point »), la note intacte", not mal, mal[:4])
m_L = req(action="modifier", etat=copy.deepcopy(e0), chemin=["L"], valeur="0,2")["message"]
chk("le message : « Saisie refusée : la portée L doit rester entre 0,5 et 30 m (reçu : 0,2) »", m_L == "Saisie refusée : la portée L doit rester entre 0,5 et 30 m (reçu : 0,2)", m_L)
e_ent = saisir(e0, "maintien/type", "entretoises")["etat"]
m_e = req(action="modifier", etat=copy.deepcopy(e_ent), chemin=["maintien", "entraxe"], valeur="9")
chk("un entraxe d'entretoises plus grand que la portée est refusé (9 m sur 4 m), le champ nommé",
    m_e["etat"] == e_ent and "entraxe des entretoises" in m_e["message"] and "portée" in m_e["message"], m_e["message"])
m_gl = req(action="modifier", etat=copy.deepcopy(e0), chemin=["classe"], valeur="GL99")
chk("un refus pendant un changement de bois ne laisse ni la poutre modifiée ni un avis en attente", m_gl["etat"] == e0 and "avis" not in m_gl and m_gl["message"], m_gl.get("message"))
chk("les champs ne portent pas de bornes locales (data-min, data-max) : c'est le moteur qui refuse et dit le domaine", "data-min" not in d0 and "data-max" not in d0)
m_psi = req(action="modifier", etat=copy.deepcopy(e0), chemin=["charges", "psi2"], valeur="0,3")
chk("réécrire un champ verrouillé avec sa valeur lue (ψ₂ = 0,3 en habitation) ne refuse ni ne change rien", m_psi["ok"] and not m_psi["message"] and m_psi["etat"] == e0)

# ====================================================================== 5. le sélecteur
print("\n=== 5. Le sélecteur de sections")
ROW = re.compile(r'<button class="pk-row(?P<cls>[^"]*)" data-choix="(?P<nom>[^"]+)" data-prop="(?P<prop>[^"]*)"><span class="mk">(?P<mk>.*?)</span>'
                 r'<span class="nm">(?P<nm>[^<]*)</span><span class="kg">(?P<kg>[^<]*)</span>(?:<span class="chip (?P<niv>\w)"(?: title="[^"]*")?>(?P<taux>[^<]*)</span>)?</button>', re.S)


def fautes_selecteur(mod, etat, cas):
    f = []
    en = mod.normaliser(copy.deepcopy(etat))
    p = en["poutres"][en["actif"]]
    cands = MO.candidats_sections(p)
    h = mod.selecteur(en, "section")
    rows = [m.groupdict() for m in ROW.finditer(h)]
    if len(rows) != len(cands):
        return [f"{cas} : {len(rows)} lignes pour {len(cands)} sections"]
    for row, c in zip(rows, cands):
        niv = "r" if not c["ok"] else C.niveau(c["taux"])
        if html.unescape(row["nom"]) != c["designation"] or not affiche(row["kg"], c["masse"]) or row["niv"] != niv \
                or html.unescape(row["taux"]) != html.unescape(pct_txt(c["taux"])):
            f.append(f"{cas} {c['nom']} : {row['nom']} {row['kg']} {row['niv']} {row['taux']} ≠ {c['masse']} {niv} {pct_txt(c['taux'])}")
        if (" cur" in row["cls"]) != c["retenue"] or (" opt" in row["cls"]) != c["optimal"]:
            f.append(f"{cas} {c['nom']} : ✓ ou ★ mal posé ({row['cls']})")
        prop = html.unescape(row["prop"])
        if NOMS_COURTS[c["gouvernant"]].lower() not in prop or ("ne tient pas" in prop) != (not c["ok"]):
            f.append(f"{cas} {c['nom']} : propriétés « {texte(prop)[:60]} »")
    masses = [lire_fr(row["kg"])[0] for row in rows]
    if masses != sorted(masses):
        f.append(f"{cas} : pas triées par masse")
    if f"{len(cands)} sections" not in h or ("aucune ne tient" in h) != (not any(c["ok"] for c in cands)):
        f.append(f"{cas} : sous-titre")
    return f


sel0 = req(action="selecteur", etat=e0, role="section")
fs = fautes_selecteur(B, e0, "massif")
chk("note neuve : 26 sections du bois massif par masse, chacune au taux du moteur (rouge quand elle ne tient pas), ✓ sur 75 × 225, ★ sur 38 × 235",
    sel0["ok"] and not fs and "Section · bois massif C24" in sel0["html"] and re.search(r'pk-row opt" data-choix="38 × 235"', sel0["html"])
    and re.search(r'pk-row cur" data-choix="75 × 225"', sel0["html"]) and "kg/m · taux" in sel0["html"] and "la plus légère qui tient" in sel0["html"], fs[:3])
fs = fautes_selecteur(B, ETATS["lamellé-collé GL24h 120 × 400 sur 8 m"], "lamellé") + fautes_selecteur(B, ETATS["sur mesure 62,5 × 187,5"], "sur mesure") \
    + fautes_selecteur(B, ETATS["très au-delà (12 m, 20 + 30 kN/m)"], "aucune")
chk("lamellé-collé : les 67 sections du lamellé ; sur mesure : aucune retenue ; rien ne tient : pas d'étoile, et le sous-titre le dit", not fs, fs[:3])
ch = req(action="choisir", etat=copy.deepcopy(e0), role="section", nom="38 × 235")
chk("choisir « 38 × 235 » : la section est retenue (standard), le calcul suit (95 %), l'étoile disparaît de la ligne de flexion",
    ch["ok"] and P(ch["etat"])["section"] == "38x235" and (P(ch["etat"])["b"], P(ch["etat"])["h"]) == (38.0, 235.0) and ch["resume"]["taux"] == 95
    and lignes(ch["resultat"])[0]["leger"] == "" and not controle_resultat(B, ch["etat"], "choisie"), ch.get("erreur"))
ch2 = req(action="choisir", etat=copy.deepcopy(d_mes["etat"]), role="section", nom="45 × 220")
chk("choisir depuis une section sur mesure la remet en standard", ch2["ok"] and P(ch2["etat"])["standard"] is True and P(ch2["etat"])["section"] == "45x220")
ch3 = req(action="choisir", etat=copy.deepcopy(e0), role="section", nom="120 × 400")
chk("une section d'un autre type est refusée par le moteur (erreur lisible)", ch3["ok"] is False and "n'existe pas en bois massif" in ch3["erreur"], ch3.get("erreur"))

# ====================================================================== 6. une ancienne note
print("\n=== 6. Une note du module d'avant (version 1)")
v1 = req(action="calculer", etat=B.NOTE_V1)
chk("une note de version 1 se rouvre : les NOMS gardés (« Solive du séjour », « Chevêtre »), des poutres NEUVES, la poutre ouverte gardée, version 2",
    v1["ok"] and v1["etat"]["version"] == 2 and [p["nom"] for p in v1["etat"]["poutres"]] == ["Solive du séjour", "Chevêtre"] and v1["etat"]["actif"] == 1
    and all(dict(p, nom="P1") == MO.nouvelle_poutre("P1") for p in v1["etat"]["poutres"]), v1.get("erreur"))
chk("l'avis d'ouverture est dit UNE fois : en tête du résultat et dans « avis » (le téléphone) — « Note d'avant le 10/11/2026 … les données sont à ressaisir »",
    v1.get("avis") == [B.AVIS_V1] and C.esc(B.AVIS_V1) in v1["resultat"] and v1["resultat"].index("Note d&#x27;avant") < v1["resultat"].index('class="verdict"')
    and "refait sur l'Eurocode 5" in B.AVIS_V1 and "à ressaisir" in B.AVIS_V1, v1.get("avis"))
v1b = req(action="calculer", etat=v1["etat"])
chk("l'état rendu ne porte pas l'avis : il ne se redit pas à la relecture", B.CLE_AVIS not in v1["etat"] and "avis" not in v1b and "Note d&#x27;avant" not in v1b["resultat"])
v1_mod = req(action="modifier", etat=copy.deepcopy(B.NOTE_V1), chemin=["L"], valeur="5")
chk("une saisie envoyée avec l'ancienne note : relue d'abord (noms gardés), la saisie s'applique, l'avis est dit", v1_mod["ok"] and P(v1_mod["etat"])["L"] == 5.0
    and P(v1_mod["etat"])["nom"] == "Chevêtre" and v1_mod.get("avis") == [B.AVIS_V1])
long_ = copy.deepcopy(B.NOTE_V1)
long_["poutres"][0]["nom"] = "N" * 75
long_["poutres"][1] = "illisible"
v1l = req(action="calculer", etat=long_)
chk("un nom de plus de 60 caractères est coupé et l'avis le dit ; une poutre illisible devient « P2 » (jamais une erreur)",
    v1l["ok"] and [p["nom"] for p in v1l["etat"]["poutres"]] == ["N" * 60, "P2"] and v1l["avis"] == [B.AVIS_V1 + B.AVIS_NOM], v1l.get("erreur"))
sans_version = req(action="calculer", etat={"poutres": [{"nom": "Sans version"}]})
chk("une note sans numéro de version est prise pour une ancienne note (nom gardé, avis)", sans_version["ok"] and sans_version["etat"]["poutres"][0]["nom"] == "Sans version"
    and "avis" in sans_version)
futur = req(action="calculer", etat={"version": 3, "poutres": [{"nom": "Demain"}]})
chk("une note d'une version PLUS RÉCENTE est refusée (rien n'est deviné)", futur["ok"] is False and "plus récente" in futur["erreur"], futur.get("erreur"))
abime = req(action="calculer", etat={"version": 2, "poutres": "abîmé"})
chk("une note abîmée (« poutres » n'est pas une liste) rend une erreur lisible", abime["ok"] is False and "liste" in abime["erreur"], abime.get("erreur"))
abime2 = req(action="calculer", etat={"version": 2, "poutres": [{"nom": "Faîtière", "L": "quatre"}]})
chk("une poutre illisible d'une note d'aujourd'hui est nommée dans l'erreur", abime2["ok"] is False and "Faîtière" in abime2["erreur"] and "portée L" in abime2["erreur"], abime2.get("erreur"))
vide = req(action="calculer", etat={"version": 2, "poutres": []})
chk("une note sans poutre est une note neuve", vide["ok"] and vide["etat"] == e0)

# ====================================================================== 7. l'échappement
print("\n=== 7. L'échappement d'un nom piégé")
PIEGE = '<img src=x onerror=alert(1)>"\'&'
pg_ = saisir(e0, "nom", PIEGE)
fragments = {"résultat": pg_["resultat"], "fenêtres": pg_["volets"], "données": pg_["donnees"], "dessins": "".join(v["svg"] + v["entete"] for v in pg_["vues"]),
             "sélecteur": req(action="selecteur", etat=pg_["etat"], role="section")["html"]}
chk("un nom piégé est gardé tel quel dans l'état et ÉCHAPPÉ dans tous les fragments (résultat, fenêtres, données, dessins, sélecteur)",
    P(pg_["etat"])["nom"] == PIEGE and not any("<img" in h for h in fragments.values()) and "&lt;img src=x" in pg_["resultat"] and "&lt;img src=x" in pg_["volets"]
    and "&lt;img src=x" in pg_["donnees"], [k for k, h in fragments.items() if "<img" in h])


class Balises(HTMLParser):
    """Les balises d'un fragment, lues comme un navigateur les lit : leurs noms et leurs attributs (le TEXTE échappé
    d'un nom piégé — « onerror=… » — n'en est pas un)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fautes = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "iframe", "object", "embed", "link", "img", "style"):
            self.fautes.append(f"<{tag}>")
        for nom, val in attrs:
            if nom.startswith("on"):
                self.fautes.append(f"{tag} {nom}=")
            if nom in ("href", "src", "xlink:href", "action", "formaction") or re.search(r"(?:https?:)?//|javascript:", val or "", re.I) and nom != "xmlns":
                self.fautes.append(f"{tag} {nom}={val}")


def fautes_balises(h):
    b = Balises()
    b.feed(h)
    b.close()
    return b.fautes


dangers = {k: fautes_balises(h) for k, h in dict(fragments, neuve="".join((r0["resultat"], r0["volets"], r0["donnees"]))).items()}
chk("aucun fragment ne porte de script, de gestionnaire « on…= », de lien ni de ressource externe (les balises lues une à une, le nom piégé compris)",
    not any(dangers.values()) and fautes_balises('<p onclick="x">') and fautes_balises('<a href="https://exemple.be">x</a>') and fautes_balises("<script>1</script>"),
    {k: v[:3] for k, v in dangers.items() if v})


# ====================================================================== 8. les fenêtres de calcul : chaque ligne réévaluée
print("\n=== 8. Les fenêtres de calcul")
def evaluer(vals):
    """« 5 × 0,6695 × 4 000⁴ / (384 × 11 000 × 71 191 406) » → le nombre, par un petit interpréteur (√, min(a ; b), ², ³, ⁴, ^, 10⁶)."""
    s = html.unescape(re.sub(r"<[^>]+>", "", vals))
    s = s.replace(C.NBSP, "").replace(C.FINE, "").replace(" ", "").replace("−", "-")
    s = s.replace("10⁹", "1e9").replace("10⁶", "1e6").replace("10³", "1e3")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s).replace(";", ",")
    s = s.replace("×", "*").replace("√(", "sqrt(").replace("^", "**").replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    return eval(s, {"__builtins__": {}}, {"sqrt": math.sqrt, "min": min})  # noqa: S307 — des nombres et des opérateurs


def lire_resultat(res_txt):
    t = html.unescape(re.sub(r"<[^>]+>", "", res_txt))
    x, nd = lire_fr(re.match(r"[-−]?[\d   ]*\d(?:,\d+)?", t.replace("> ", "")).group(0))
    return (x / 100.0 if "%" in t else x), nd, "%" in t


def controle_formules(mod, etat, cas):
    """Les fautes des fenêtres : chaque ligne à trois parties réévaluée, chaque ligne à deux parties (valeur lue) dite."""
    rep = resultat_module(mod, etat)
    fautes, n, lues = [], 0, 0
    for ligne in FX.findall(rep["volets"]):
        spans = SPAN.findall(ligne)
        parts = [t for cls, t in spans if cls not in ("r",) and not cls.startswith("chip")]
        res_txt = next((t for cls, t in spans if cls == "r" or cls.startswith("chip")), None)
        if res_txt is None or len(parts) not in (2, 3):
            fautes.append(f"{cas} : ligne mal formée « {texte(ligne)[:60]} »")
            continue
        if len(parts) == 2:
            lues += 1
            continue
        n += 1
        try:
            val = evaluer(parts[2])
        except Exception as e:  # noqa: BLE001 — une expression illisible est une faute
            fautes.append(f"{cas} {texte(parts[0])} : illisible ({e}) : {texte(parts[2])}")
            continue
        attendu, nd, pct = lire_resultat(res_txt)
        tol = (0.5 * 10 ** -nd) / (100.0 if pct else 1.0) + 2e-3 * abs(attendu)
        if abs(val - attendu) > tol:
            fautes.append(f"{cas} {texte(parts[0])} = {texte(parts[2])} → {val} ≠ {attendu}")
    return fautes, n, lues


total = 0
for nom_cas, etat in ETATS.items():
    fautes, n, lues = controle_formules(B, etat, nom_cas)
    total += n
    chk(f"fenêtres : chaque ligne « expression = valeurs = résultat » réévaluée ({n}, et {lues} valeurs lues) — {nom_cas}", not fautes and n >= 60, fautes[:3])
print(f"     ({total} lignes de formules réévaluées)")
vol = r0["volets"]


def fenetre(volets, ident):
    m = re.search(rf'<div class="flyout calcul" id="ck-{ident}">(.*?)(?=<div class="flyout calcul"|$)', volets, re.S)
    return m.group(1) if m else ""


def lignes_tableau(volets, ident):
    return re.findall(r'<tr class="(gov)?"><td class="nm">(.*?)</td><td class="fo">(.*?)</td><td class="n v">(.*?)</td><td class="n">(.*?)</td><td class="n">(.*?)</td></tr>',
                      fenetre(volets, ident), re.S)


def fautes_tableaux(mod, etat, cas):
    """Le tableau de chaque fenêtre face au moteur : une ligne par combinaison (celle qui gouverne mise en avant), ses
    nombres, ses taux ; « tout le calcul » : toutes les vérifications du bloc."""
    f = []
    en = mod.normaliser(copy.deepcopy(etat))
    r = MO.verifier(en["poutres"][en["actif"]])
    vol_ = mod.resultat(en)["volets"]
    for v in r["verifications"]:
        rows = lignes_tableau(vol_, v["cle"])
        det = v["detail"] or [dict(v, combinaison=None)]
        if len(rows) != len(det):
            f.append(f"{cas} {v['cle']} : {len(rows)} ligne(s) de tableau pour {len(det)}")
            continue
        for row, d in zip(rows, det):
            val, lim = re.search(rf"= ({NOMBRE})", sans_indices(row[3])), re.search(rf"= ({NOMBRE})", sans_indices(row[4]))
            if not val or not lim or not affiche(val.group(1), d["valeur"]) or not affiche(lim.group(1), d["limite"]) or pct_txt(d["taux"]) not in row[5]:
                f.append(f"{cas} {v['cle']} : tableau « {texte(row[3])} / {texte(row[4])} / {texte(row[5])} » ≠ {d['valeur']} / {d['limite']} / {d['taux']}")
            if bool(row[0]) != (d["combinaison"] == v["combinaison"]):
                f.append(f"{cas} {v['cle']} : la combinaison qui gouverne n'est pas celle mise en avant")
            if d["combinaison"] and (d["nom"] not in texte(row[1]) or D.LIBELLES_DUREE[d["duree"]] not in texte(row[1])):
                f.append(f"{cas} {v['cle']} : la ligne ne nomme pas sa combinaison")
            if html.unescape(re.sub(r"<[^>]+>", "", row[2])) != v["enonce"].replace("_", ""):
                f.append(f"{cas} {v['cle']} : énoncé « {texte(row[2])} » ≠ « {v['enonce']} »")
    n_elu = sum(len(v["detail"]) for v in r["verifications"] if v["groupe"] == "ELU")
    if len(lignes_tableau(vol_, "s0")) != n_elu or len(lignes_tableau(vol_, "s1")) != 2:
        f.append(f"{cas} : « tout le calcul » : {len(lignes_tableau(vol_, 's0'))} + {len(lignes_tableau(vol_, 's1'))} lignes")
    return f


ft = [x for nom_cas, etat in ETATS.items() for x in fautes_tableaux(B, etat, nom_cas)]
chk("le tableau de chaque fenêtre : une ligne par combinaison (valeur, limite, taux du moteur), celle qui gouverne mise en avant, l'énoncé du moteur ; "
    "« tout le calcul » : toutes les vérifications du bloc", not ft, ft[:3])
hyp = re.search(r'<div class="ck-hyp">(.*?)</div><div class="ck-body">', fenetre(vol, "flexion"), re.S).group(1)
chk("les hypothèses en pastilles : le bois et sa norme, la classe de service, k_mod des DEUX combinaisons, k_def, γ_M, la nature de Q et ψ₂",
    all(m in texte(hyp) for m in ("C24", "EN 338:2016, tableau 1", "Classe de service", "intérieur chauffé", "0,6 (G seule, permanent)", "0,8 (G + Q, moyen terme)",
                                 "A · habitation")) and "k<sub>def</sub> = 0,6" in hyp and "γ<sub>M</sub> = 1,3" in hyp and "ψ<sub>2</sub> = 0,3" in hyp, texte(hyp))
titres = [html.unescape(re.sub(r"<[^>]+>", " ", t)).strip() for t in re.findall(r'<div class="ck-t">(.*?)</div>', fenetre(vol, "flexion"))]
chk("le détail de la fenêtre « flexion », groupé : Charges ; Combinaison (la gouvernante d'abord, dite) ; Section ; Déversement ; Flexion de chaque combinaison",
    [re.sub(r"\s+", " ", t) for t in titres] == ["Charges", "Combinaison G + Q moyen terme · gouverne", "Combinaison G seule permanent", "Section", "Déversement",
                                                "Flexion G + Q · moyen terme · gouverne", "Flexion G seule · permanent"], titres)
titres_s0 = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)).strip()).split(" ")[0] for t in re.findall(r'<div class="ck-t">(.*?)</div>', fenetre(vol, "s0"))]
titres_s1 = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)).strip()) for t in re.findall(r'<div class="ck-t">(.*?)</div>', fenetre(vol, "s1"))]
chk("« tout le calcul » ELU : charges, combinaisons, section, déversement, flexion, cisaillement, appui (dans l'ordre du moteur) ; ELS : charges, section, flèches",
    titres_s0 == ["Charges", "Combinaison", "Combinaison", "Section", "Déversement", "Flexion", "Flexion", "Cisaillement", "Cisaillement", "Appui", "Compression", "Compression"]
    and titres_s1 == ["Charges", "Section", "Flèche instantanée", "Flèche finale nette"], (titres_s0, titres_s1))
chk("le pied de chaque fenêtre : les hypothèses du moteur et ce qui n'est pas couvert (charges concentrées, vibrations, feu…)",
    vol.count('class="ck-autres bo-pied"') == 7 and all(C.symb_math(x) in fenetre(vol, "s0") for x in REF0["non_couvert"])
    and "k<sub>v</sub>" in fenetre(vol, "s0") and "k_v" not in vol + r0["resultat"]
    and "maintenus en torsion" in fenetre(vol, "flexion"))
chk("les symboles s'écrivent avec leurs indices (σ_m,d, k_c,90, w_net,fin, E_0,mean, λ_rel,m) et chaque taux en pastille de couleur",
    all(s in vol for s in ("σ<sub>m,d</sub>", "k<sub>c,90</sub>", "w<sub>net,fin</sub>", "E<sub>0,mean</sub>", "f<sub>c,90,d</sub>"))
    and "λ<sub>rel,m</sub>" in rd["volets"] and '<span class="chip p">45 %</span>' in vol)

# ====================================================================== 9. les vues
print("\n=== 9. Les vues")
vues = r0["vues"]
chk("trois vues (schéma et efforts, déformée, section), chacune un SVG sous son en-tête technique, aucune clé qui passerait pour une section (js/compact.js)",
    [(v["cle"], v["titre"], v["court"]) for v in vues] == [("schema", "Schéma et efforts", "Schéma et efforts"), ("deformee", "Déformée", "Déformée"), ("section", "Section", "Section")]
    and all(v["svg"].lstrip().startswith("<svg") and 'class="et-ent"' in v["entete"] for v in vues) and not any(re.fullmatch(r"[a-z]+\d+", v["cle"]) for v in vues))


def fautes_entete(mod, etat, cas):
    f = []
    en = mod.normaliser(copy.deepcopy(etat))
    r = MO.verifier(en["poutres"][en["actif"]])
    for v in mod.resultat(en)["vues"]:
        bande = re.findall(rf"<span><i>(\w)</i><sub>(\w+)</sub><b>({NOMBRE})</b><small>([^<]*)</small></span>", v["entete"])
        ef = r["efforts"]
        if [(s, i, u) for s, i, _, u in bande] != [("M", "d", "kN·m"), ("V", "d", "kN"), ("R", "d", "kN")] \
                or not all(affiche(x[2], val) for x, val in zip(bande, (ef["M_d"], ef["V_d"], ef["R"]))):
            f.append(f"{cas} {v['cle']} : bande {bande} ≠ {ef['M_d']}, {ef['V_d']}, {ef['R']}")
        t = texte(v["entete"]).replace(C.NBSP, " ")          # (une quantité y est d'un seul tenant : ses espaces sont insécables)
        ch_, ap_ = r["charges"], r["appui"]
        for mot in (r["geometrie"]["designation"] + " mm", r["materiau"]["classe"], f"L = {C.fr(r['geometrie']['L'], 2)} m", f"g = {C.fr(ch_['g'], 2)} kN/m",
                    f"q = {C.fr(ch_['q'], 2)} kN/m", f"combinaison {ef['nom']}", f"a = {C.nb(ap_['a'])} mm", r["coefficients"]["libelle_nature"],
                    f"classe de service {r['coefficients']['service']}", r["deversement"]["libelle"]):
            if mot not in t:
                f.append(f"{cas} {v['cle']} : l'en-tête ne dit pas « {mot} »")
        for lie in (f"a{C.NBSP}={C.NBSP}{C.nb(ap_['a'])}{C.NBSP}mm", f"L{C.NBSP}={C.NBSP}{C.fr(r['geometrie']['L'], 2)}{C.NBSP}m"):
            if lie not in html.unescape(v["entete"]):
                f.append(f"{cas} {v['cle']} : « {lie} » peut se couper en fin de ligne")
        if C.esc(v["legende"]) not in v["entete"]:
            f.append(f"{cas} {v['cle']} : légende absente de l'en-tête")
    return f


fe = [x for nom_cas, etat in ETATS.items() for x in fautes_entete(B, etat, nom_cas)]
chk("l'en-tête de chaque vue : la section, le bois, la portée, g et q, la nature de Q, la classe de service, la combinaison ELU qui gouverne, l'appui, le "
    "maintien (chaque quantité d'un seul tenant) ; la bande M_d, V_d, R_d du moteur ; la légende", not fe, fe[:3])


def textes_svg(svg):
    return [html.unescape(re.sub(r"<[^>]+>", "", t)) for t in re.findall(r"<text[^>]*>(.*?)</text>", svg, re.S)]


def fautes_dessins(mod, etat, cas):
    """Ce que les dessins écrivent face au moteur : la portée, l'appui, les charges, R_d, V_d, M_d, la combinaison ; les
    flèches, leurs limites et leur L/n ; b, h, les contraintes et leurs limites ; le platelage, les lamelles."""
    f = []
    en = mod.normaliser(copy.deepcopy(etat))
    r = MO.verifier(en["poutres"][en["actif"]])
    vs = {v["cle"]: v for v in mod.resultat(en)["vues"]}
    geo, ch, ef, fl, ap = r["geometrie"], r["charges"], r["efforts"], r["fleches"], r["appui"]
    t = textes_svg(vs["schema"]["svg"])
    attendus = [f"L = {C.fr(geo['L_mm'], 0)}", f"a = {C.nb(ap['a'])} mm", f"Rd = {C.fr(ef['R'], 2)} kN", f"Md = {C.fr(ef['M_d'], 2)} kN·m", f"combinaison {ef['nom']}"]
    attendus += [f"g = {C.fr(ch['g'], 2)} kN/m"] if ch["g"] > 0 else []
    attendus += [f"q = {C.fr(ch['q'], 2)} kN/m"] if ch["q"] > 0 else []
    for a in attendus:
        if not any(a in x for x in t) and ef["M_d"] >= 0.1:
            f.append(f"{cas} schéma : « {a} » absent")
    if ef["V_d"] >= 0.1 and (f"+{C.fr(ef['V_d'], 2)}" not in t or f"−{C.fr(ef['V_d'], 2)}" not in t):
        f.append(f"{cas} schéma : ±V_d absent")
    t = " | ".join(textes_svg(vs["deformee"]["svg"]))
    for a in (f"winst = {C.fr(fl['w_inst'], 2)} mm", f"L/{fl['n_inst']} = {C.fr(fl['lim_inst'], 1)} mm", f"L/{fl['n_fin']} = {C.fr(fl['lim_fin'], 1)} mm",
              f"wnet,fin = {C.fr(fl['w_net_fin'], 2)} mm") + ((f"= L/{C.fr(fl['rapport_inst'], 0)}",) if fl["rapport_inst"] else ()):
        if a not in t and fl["w_inst"] >= 0.1:
            f.append(f"{cas} déformée : « {a} » absent")
    svg = vs["section"]["svg"]
    t = " | ".join(textes_svg(svg))
    vm, vt = V(r, "flexion"), V(r, "cisaillement")
    for a in (f"b = {C.nb(geo['b'])}", f"h = {C.nb(geo['h'])}", f"σm,d = {C.fr(vm['valeur'], 2)}", f"kcrit·fm,d = {C.fr(vm['limite'], 2)}", f"τd = {C.fr(vt['valeur'], 2)}",
              f"fv,d = {C.fr(vt['limite'], 2)}", r["materiau"]["classe"]):
        if a not in t and vt["valeur"] >= 0.1:
            f.append(f"{cas} section : « {a} » absent")
    continu = r["deversement"]["type"] == "continu"
    if ("maintien continu" in t) != continu or (f'fill="{mod.DB.PLATELAGE}"' in svg) != continu:
        f.append(f"{cas} section : le platelage du maintien continu mal posé")
    lamelle = r["materiau"]["type"] == "lamelle"
    if ("lamelles horizontales" in t) != lamelle:
        f.append(f"{cas} section : lamelles mal dites")
    return f


fdz = [x for nom_cas, etat in ETATS.items() for x in fautes_dessins(B, etat, nom_cas)]
chk("les dessins écrivent les nombres du moteur : L, a, g, q, R_d, ±V_d, M_d et la combinaison ; w_inst, w_net,fin, leurs limites, leur L/n ; b, h, σ_m,d, "
    "k_crit·f_m,d, τ_d, f_v,d ; le platelage du seul maintien continu, les lamelles du seul lamellé-collé", not fdz, fdz[:3])
svg_gl = resultat_module(B, ETATS["lamellé-collé GL24h 120 × 400 sur 8 m"])["vues"][2]["svg"]
chk("la section d'un lamellé-collé 120 × 400 montre ses 10 lamelles horizontales (9 plans de collage), pas les cernes d'un bois massif",
    len(re.findall(rf'<line [^>]*stroke="{B.DB.VEINE}" stroke-width="0.9"', svg_gl)) == 9 and "<circle" not in svg_gl and "<circle" in vues[2]["svg"])
grand = resultat_module(B, etat_avec(standard=False, b=400.0, h=2500.0, L=3.0, appui__a=600.0, appui__debord=500.0))["vues"][0]["svg"]
chk("une poutre très haute sur une portée courte (400 × 2 500 sur 3 m, a = 600, débord 500) : le schéma est réduit et le DIT ; la note neuve ne dit rien",
    "dessin réduit pour la lisibilité" in grand and "la hauteur de la poutre" in grand and "la longueur d'appui" in grand and "le débord" in grand
    and "dessin réduit" not in vues[0]["svg"])
ids = [re.search(r'<svg id="([^"]+)"', v["svg"]).group(1) for v in vues]
chk("chaque dessin a son identifiant, propre à la poutre ouverte (deux dessins d'une page ne se mêlent pas)", ids == ["bo0_schema", "bo0_deformee", "bo0_section"], ids)

# ====================================================================== 9 bis. l'écran n'écrit aucune valeur de norme
print("\n=== 9 bis. Aucune formule, aucune valeur de norme écrite dans l'écran")
NORME = re.compile(r"\d+,\d+|\bL/\d|\b\d+\s?(?:mm|cm|m|kN|ans|mois|jours?)\b|\d\s(?:%|N/mm²)|[=≤≥<>]\s?\d|\d\s?[·×]\s?[A-Za-zγψσλ]")
SYMBOLES_A_CHIFFRE = ("E_0,05", "E_0,mean", "ψ<sub>2</sub>", "ψ_2", "L/2")          # des NOMS (et la mi-portée du dessin), pas des valeurs
HORS_TEXTES = {"NOTE_V1", "PARCOURS", "AVIS_V1", "AVIS_NOM", "DECIMALES", "ESPACES"}  # des saisies d'essai, une date, des précisions d'affichage


def valeurs_de_norme(source):
    """Les textes d'un fichier de l'écran qui portent un nombre de norme (« 30 mm », « 0,9·L », « L/300 », « 10 % »…) :
    chaque constante de texte de ses fonctions et de ses affectations, hors docstrings et hors saisies d'essai."""
    arbre = ast.parse(source)
    docs = {id(n.body[0].value) for n in ast.walk(arbre) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    zones = [n for n in arbre.body if isinstance(n, ast.FunctionDef)
             or (isinstance(n, ast.Assign) and not any(getattr(t, "id", None) in HORS_TEXTES for t in n.targets))]
    out = []
    for z in zones:
        for n in ast.walk(z):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                t = n.value
                for s_ in SYMBOLES_A_CHIFFRE:
                    t = t.replace(s_, "")
                if NORME.search(t):
                    out.append((n.lineno, n.value[:80]))
    return out


SRC_ECRAN = {f: open(os.path.join(RACINE, "api_web", f), encoding="utf-8").read() for f in ("bois.py", "dessin_bois.py")}
vn = {f: valeurs_de_norme(s) for f, s in SRC_ECRAN.items()}
chk("api_web/bois.py et api_web/dessin_bois.py n'écrivent AUCUN nombre de norme dans leurs textes (ni « 30 mm », ni « 0,9·L », ni « L/300 », ni "
    "« 1,35 ») : les aides ⓘ, les formules et les hypothèses sont celles du moteur", not any(vn.values()), vn)
chk("le lecteur de ce contrôle voit bien une valeur remise à la main (« de 30 mm au plus », « l_ef = 0,9·L + 2·h », « L/300 à L/500 », « 1,35 »)",
    all(valeurs_de_norme(f'def f():\n    return "{t}"\n') for t in ("de 30 mm au plus", "l_ef = 0,9·L + 2·h", "L/300 à L/500", "(6.10 : 1,35)", "majorés de 10 %")))
aides_ecran = [html.unescape(a) for a in re.findall(r'<span class="aide" title="([^"]+)">', "".join(resultat_module(B, e)["donnees"] for e in (
    e0, etat_avec(standard=False), etat_avec(charges__mode="surfacique"), etat_avec(maintien__type="entretoises"))))]
chk(f"chaque aide du moteur ({len(MO.AIDES)}) est posée derrière un ⓘ du cadre Données (celle de la nature de Q et les deux verrous sont complétés par l'écran)",
    all(any(a.startswith(v) for a in aides_ecran) for v in MO.AIDES.values()) and len(MO.AIDES) == 23, [k for k, v in MO.AIDES.items() if not any(a.startswith(v) for a in aides_ecran)])

# le harnais du site charge les modules de l'écran avec le python3 du SYSTÈME (3.11) : une f-string imbriquée à la
# manière de Python 3.12 (les mêmes guillemets dedans et dehors) y est une erreur de syntaxe
FICHIERS_MODULE = [os.path.join(RACINE, *f.split("/")) for f in ("api_web/bois.py", "api_web/dessin_bois.py", "bois/poutre/__init__.py", "bois/poutre/donnees.py",
                                                                 "bois/poutre/entree.py", "bois/poutre/moteur.py", "bois/poutre/diagrammes.py")]
LIRE = "import ast, sys\nfor f in sys.argv[1:]:\n    ast.parse(open(f, encoding='utf-8').read(), f)\nprint(sys.version_info[:2])"
for py in ("/usr/bin/python3.11", "/usr/bin/python3.13"):
    if not os.path.exists(py):
        print(f"     ({py} absent de cette machine : la lecture par cette version n'est pas jouée)")
        continue
    lu = subprocess.run([py, "-c", LIRE] + FICHIERS_MODULE, capture_output=True, text=True, timeout=60)
    chk(f"les sept fichiers du module se LISENT avec {os.path.basename(py)} (aucune syntaxe réservée à une version plus récente)",
        lu.returncode == 0, lu.stderr.strip().splitlines()[-1:] if lu.stderr else "")

# ====================================================================== 10. les éléments de la note
print("\n=== 10. Les éléments de la note, son résumé")
e2 = req(action="ajouter", etat=copy.deepcopy(e0))
chk("ajouter : « P2 » ouverte, deux entrées dans la liste de la ligne de l'élément, « nom · section · classe »",
    e2["etat"]["actif"] == 1 and e2["etat"]["poutres"][1]["nom"] == "P2" and e2["donnees"].count('data-activer="') == 2 and 'data-elements="2"' in e2["donnees"]
    and "· 75 × 225 · C24" in e2["donnees"])
e3 = req(action="dupliquer", etat=copy.deepcopy(e2["etat"]), index=0)
chk("copier la première l'insère juste après elle, « (copie) », et l'ouvre", [p["nom"] for p in e3["etat"]["poutres"]] == ["P1", "P1 (copie)", "P2"] and e3["etat"]["actif"] == 1)
e4 = saisir(e3["etat"], "charges/Q", "5")
chk("le résumé de la note est sa poutre la plus chargée (la copie à Q = 5 kN/m, non conforme) ; l'état de la note s'affiche dès deux poutres",
    e4["resume"]["niveau"] == "r" and e4["resume"]["gouvernant"].startswith("P1 (copie)") and "Non conforme" in e4["statut_html"], e4["resume"])
e5 = req(action="supprimer", etat=copy.deepcopy(e4["etat"]), index=1)
chk("supprimer la copie (ouverte) : deux poutres restent, le résumé redevient conforme", [p["nom"] for p in e5["etat"]["poutres"]] == ["P1", "P2"] and e5["resume"]["niveau"] == "p")
chk("la dernière poutre ne se supprime pas ; activer un indice hors de la liste ouvre la dernière",
    len(req(action="supprimer", etat=copy.deepcopy(e0), index=0)["etat"]["poutres"]) == 1 and req(action="activer", etat=copy.deepcopy(e5["etat"]), index=7)["etat"]["actif"] == 1)
chk("un téléchargement est refusé proprement (pas encore de note PDF pour ce module)", req(action="telecharger", etat=copy.deepcopy(e0), quoi="note_pdf").get("ok") is False)


# ====================================================================== 11. les mutations : chacune vue rouge
print("\n=== 11. Les mutations du module : chacune vue rouge")
def mutant(ancien, nouveau):
    src = open(B.__file__, encoding="utf-8").read()
    assert src.count(ancien) == 1, (ancien, src.count(ancien))
    m = types.ModuleType("api_web.bois_mutant")
    m.__package__ = "api_web"
    m.__file__ = B.__file__
    m.SOURCE = src.replace(ancien, nouveau, 1)
    exec(compile(m.SOURCE, B.__file__, "exec"), m.__dict__)  # noqa: S102 — le module lui-même, muté
    return m


def rouge(fn):
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 — un mutant qui plante est vu rouge aussi
        return True


E_G, E_D = ETATS["G seule gouverne (G = 3, Q = 0,5)"], ETATS["déversement (45 × 245 sur 5 m, aux appuis)"]


def modifier_mutant(m, etat, chemin, valeur):
    e = m.normaliser(copy.deepcopy(etat))
    msg = m.modifier(e, chemin.split("/"), valeur)
    return e, msg, m.resultat(e)


MUTATIONS = [
    ("le verdict montre la première vérification, pas celle qui gouverne",
     'v = next(x for x in r["verifications"] if x["cle"] == r["gouvernant"]["cle"])', 'v = r["verifications"][0]', lambda m: controle_resultat(m, e0, "neuve")),
    ("la comparaison écrit la limite à la place de la valeur",
     'return f"{_symb(v[\'symbole\'])} = {_n(v[\'valeur\'], u)} {signe} {lim}{_nu(v[\'limite\'], u)}"',
     'return f"{_symb(v[\'symbole\'])} = {_n(v[\'limite\'], u)} {signe} {lim}{_nu(v[\'limite\'], u)}"', lambda m: controle_resultat(m, e0, "neuve")),
    ("la ligne ne nomme plus la combinaison qui la gouverne (toujours « G + Q »)",
     'return C.esc(c["nom"]) if court else', 'return "G + Q" if court else', lambda m: controle_resultat(m, E_G, "G seule")),
    ("une ligne non conforme reste verte (la couleur du taux seul)",
     'return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])', 'return "p"', lambda m: controle_resultat(m, E_G, "G seule")),
    ("le taux d'une ligne est plafonné à 999 % par le module",
     'rows += C.ligne_verif(COURTS[v["cle"]], _comparaison(r, v) + suite, C.pct(v["taux"]), _niv(v), calcul=v["cle"],',
     'rows += C.ligne_verif(COURTS[v["cle"]], _comparaison(r, v) + suite, C.pct(min(v["taux"], 9.99)), _niv(v), calcul=v["cle"],',
     lambda m: controle_resultat(m, ETATS["très au-delà (12 m, 20 + 30 kN/m)"], "au-delà")),
    ("les valeurs substituées arrondies à deux chiffres (la réévaluation ne retombe plus sur le résultat)",
     "        nd = _nd(x, 4, 4)\n        s = C.nb(round(x, nd), nd)", "        nd = _nd(x, 1, 1)\n        s = C.nb(round(x, nd), nd)", lambda m: controle_formules(m, e0, "neuve")[0]),
    ("le résultat d'une formule pris dans une autre unité (kN·m écrits en N·mm)",
     'return {"parts": parts, "res": _nu(f["resultat"], f["unite"]), "niveau": ""}',
     'return {"parts": parts, "res": _nu(f["resultat"] * (1e6 if f["unite"] == "kN·m" else 1), f["unite"]), "niveau": ""}', lambda m: controle_formules(m, e0, "neuve")[0]),
    ("le tableau d'une fenêtre met en avant la combinaison qui ne gouverne pas",
     "lignes = _lignes_tableau(r, v, lambda vv, c: c == vv[\"combinaison\"])", "lignes = _lignes_tableau(r, v, lambda vv, c: c != vv[\"combinaison\"])",
     lambda m: fautes_tableaux(m, E_G, "G seule")),
    ("le nom de la poutre n'est plus échappé dans la tête du résultat",
     'nom = C.esc(p.get("nom") or "Poutre")', 'nom = p.get("nom") or "Poutre"',
     lambda m: "<img" in m.resultat(m.normaliser(copy.deepcopy(pg_["etat"])))["resultat"] + m.resultat(m.normaliser(copy.deepcopy(pg_["etat"])))["volets"]),
    ("l'avis d'ouverture reste dans l'état (il se redirait à chaque calcul)",
     "cle_avis = etat.pop(CLE_AVIS, None)", "cle_avis = etat.get(CLE_AVIS)",
     lambda m: m.CLE_AVIS in (lambda e: (m.resultat(e), e)[1])(m.normaliser(copy.deepcopy(B.NOTE_V1)))),
    ("une ancienne note perd ses noms", 'nom = p.get("nom") if isinstance(p, dict) else None', "nom = None",
     lambda m: [p["nom"] for p in m.normaliser(copy.deepcopy(B.NOTE_V1))["poutres"]] != ["Solive du séjour", "Chevêtre"]),
    ("une saisie refusée laisse la poutre modifiée",
     '        etat["poutres"][etat["actif"]] = avant\n        if avis_avant is None:', '        if avis_avant is None:',
     lambda m: modifier_mutant(m, e0, "L", "0,2")[0] != e0),
    ("la nature de Q ne refixe plus la durée ni ψ₂ (la saisie « stockage » est refusée, ou garde 0,3)",
     '                cible["duree"] = cible["psi2"] = None           # déduits de la nature (le moteur les remplit)', "                pass",
     lambda m: (lambda e, msg, _: msg or e["poutres"][0]["charges"]["psi2"] != 0.8)(*modifier_mutant(m, e0, "charges/nature", "E"))),
    ("le bois qui change de type prend la section standard la plus proche SANS le dire (dimensions changées en silence)",
     '        if p["standard"]:\n            p["standard"] = False\n            etat[CLE_AVIS] = "section"\n', "",
     lambda m: (lambda e, msg, rep: (e["poutres"][0]["b"], e["poutres"][0]["h"]) != (75.0, 225.0) and "avis" not in rep)(*modifier_mutant(m, e0, "classe", "GL24h"))),
    ("le sélecteur présente les sections dans l'ordre inverse (la plus lourde d'abord)", '"prop": prop(c)} for c in cands]', '"prop": prop(c)} for c in reversed(cands)]',
     lambda m: fautes_selecteur(m, e0, "massif")),
    ("le sélecteur marque ★ la section retenue", '"cur": c["retenue"], "opt": c["optimal"], "prop": prop(c)} for c in cands]',
     '"cur": c["retenue"], "opt": c["retenue"], "prop": prop(c)} for c in cands]', lambda m: fautes_selecteur(m, e0, "massif")),
    ("le sélecteur écrit la masse d'une autre colonne (l'aire)", '"col": C.fr(c["masse"], 2)', '"col": C.fr(c["A"] / 100, 2)', lambda m: fautes_selecteur(m, e0, "massif")),
    ("l'étoile de la ligne de flexion annonce la section retenue",
     "    opt = MO.optimal(p)\n    leger = (", "    opt = next((c for c in MO.candidats_sections(p) if c[\"retenue\"]), None)\n    leger = (", lambda m: controle_resultat(m, e0, "neuve")),
    ("la bande de l'en-tête écrit V_d à la place de M_d", 'bande = [("M", "d", ef["M_d"], "kN·m")', 'bande = [("M", "d", ef["V_d"] * 2, "kN·m")', lambda m: fautes_entete(m, e0, "neuve")),
    ("le pied écrit la masse au mètre à la place de la masse de la pièce", "f'<b>{C.qte(r[\"masse\"], \"kg\", 1)} ·", "f'<b>{C.qte(r[\"masse_lineique\"], \"kg\", 1)} ·",
     lambda m: controle_resultat(m, e0, "neuve")),
    ("les avertissements du moteur ne sont plus rendus", 'messages = "".join(C.message(_message(a)) for a in r["avertissements"])', 'messages = ""',
     lambda m: controle_resultat(m, ETATS["appuis rapprochés (k_c,90 = 1)"], "k_c,90")),
    ("ce qui n'est pas couvert n'est plus écrit sous le résultat",
     '    return f\'<div class="et-hyps">{lignes}{non}</div>\'', '    return f\'<div class="et-hyps">{lignes}</div>\'', lambda m: controle_resultat(m, e0, "neuve")),
    ("k_crit n'est plus dit sur la ligne de flexion quand la poutre déverse", 'if v["cle"] == "flexion" and r["deversement"]["kcrit"] < 1:', "if False:",
     lambda m: controle_resultat(m, E_D, "déversement")),
    ("une aide récrite à la main dans l'écran, avec sa valeur de norme (« 30 mm »)", 'A["appui/debord"]))',
     '"Le débord augmente la longueur efficace, de 30 mm au plus."))', lambda m: valeurs_de_norme(m.SOURCE)),
]
TEMOIN = ("témoin : un commentaire changé", "# la plus légère qui tient (l'étoile de la ligne « Flexion », qui ouvre le sélecteur)", "# la plus légère qui tient (témoin)")
survivants = []
for titre, ancien, nouveau, controle in MUTATIONS:
    m = mutant(ancien, nouveau)
    if not rouge(lambda controle=controle, m=m: controle(m)):
        survivants.append(titre)
chk(f"{len(MUTATIONS)} mutations d'affichage (verdict, comparaison, combinaison nommée, couleur, plafond, valeurs substituées, unité d'un résultat, tableau, échappement, "
    "avis, noms d'une ancienne note, refus, nature de Q, changement de type, sélecteur × 3, étoile, bande, pied, avertissements, non couvert, k_crit, aide récrite) : TOUTES vues rouges",
    not survivants, survivants)
mt = mutant(TEMOIN[1], TEMOIN[2])
temoin_rouge = [t for t, _, _, c in MUTATIONS if rouge(lambda c=c: c(mt))]
chk("le témoin (un commentaire changé) n'est vu rouge par AUCUN de ces contrôles : les rouges viennent bien des mutations", not temoin_rouge, temoin_rouge)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
