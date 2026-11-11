# -*- coding: utf-8 -*-
"""L'API du site pour les fondations de grues (api_web/fondation_grue.py ; le moteur : beton/fondation_grue, tenu par
tests/test_fondation_grue_moteur.py), vérifiée en Python ordinaire : CE QUE LE SITE AFFICHE, DESSINE ET EXPORTE = CE
QUE LE MOTEUR CALCULE.

  1. LA NOTE NEUVE : version 1, une fondation à 8 pieux aux valeurs du classeur BASE, six rubriques, la réponse complète.
  2. LE RÉSULTAT FACE AU MOTEUR, sur les quatre classeurs et des états de plus (la paroi de pieux sécants, un pieu
     « PAS OK », un ferraillage « LIMITE », un cisaillement au-delà de la zone 3) : le verdict, chaque ligne (son nom,
     sa pièce, ce qu'elle compare, son taux, sa couleur, l'état du classeur), les en-têtes des blocs, les valeurs que le
     classeur calcule sans les vérifier, les contrôles de géométrie, les écarts — chaque nombre RELU dans le HTML.
  3. CHAQUE CHAMP RELIÉ : modifier → l'état (au bon endroit) → le calcul ; le schéma qui change, « reprendre un
     classeur », un cas de la grue de plus, de moins.
  4. LES REFUS : le message du moteur, la fondation intacte.
  5. L'ÉCHAPPEMENT d'un texte piégé dans tous les fragments, dessins compris.
  6. LES FENÊTRES DE CALCUL : chaque formule du moteur y est, à la lettre (symbole = expression = valeurs = résultat,
     puis la cellule du classeur) ; le tableau ; les écarts ; les cas de la grue ; le contrôle avant export.
  7. LES VUES : le plan (autant de cercles que de pieux, les cotes saisies) et, par poutre ou semelle, ses diagrammes
     (les extremums de l'enveloppe et les valeurs du classeur écrits sont ceux du moteur) ; des SVG bien formés.
  8. LA GÉOMÉTRIE du plan de principe : les pieux là où le schéma les met.
  9. L'EXPORT : le fichier RBAR (les modèles de poutre portent les armatures de la note, zone par zone), son refus
     quand une vérification est PAS OK (et le bouton retiré), le DXF, les deux tableaux CSV.
 10. LES ÉLÉMENTS de la note, son résumé.
 11. Des MUTATIONS de l'écran, de l'export, des dessins et de la géométrie, chacune vue rouge par ces contrôles.

    /usr/bin/python3.12 tests/test_api_web_fondation_grue.py
"""
import copy
import html
import json
import math
import os
import re
import sys
import types
import xml.etree.ElementTree as ET

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.fondation_grue import ecriture as W  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import fondation_grue as FG  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.fondation_grue import entree as EN  # noqa: E402
from beton.fondation_grue import export as X  # noqa: E402
from beton.fondation_grue import geometrie as G  # noqa: E402
from beton.fondation_grue import moteur as MO  # noqa: E402
from beton.fondation_grue import schemas as SC  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:500]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="fondation_grue", **k))))


def note(*saisies, exemple=None):
    """Une note d'une fondation : un classeur repris (ou la note neuve), puis des saisies (chemin, valeur)."""
    e = FG.nouvel_etat()
    if exemple:
        assert FG.modifier(e, ["exemple"], exemple) is None
    for chemin, valeur in saisies:
        msg = FG.modifier(e, chemin.split("/"), valeur)
        assert msg is None, (chemin, valeur, msg)
    return e


ESPACES = str.maketrans({C.NBSP: " ", C.FINE: " "})


def texte(frag):
    """Le texte d'un fragment HTML : sans balises, entités lues, espaces ordinaires."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", frag)).translate(ESPACES).split())


def nombres(s):
    """Les nombres d'un texte à la française (« 1 101,4 », « −3,2 ») : les espaces de milliers sont dans le nombre."""
    out = []
    for m in re.finditer(r"[−-]?\d{1,3}(?: \d{3})+(?:,\d+)?|[−-]?\d+(?:,\d+)?", s):
        out.append(float(m.group(0).replace(" ", "").replace(",", ".").replace("−", "-")))
    return out


def proche(lu, exact, nd):
    """Le nombre lu est l'exact, à l'arrondi affiché (``nd`` décimales)."""
    return abs(lu - exact) <= 0.5 * 10 ** (-nd) + 1e-9


BLOC = re.compile(r'<section class="et-bloc (\w+)" data-theme="(\d+)">(.*?)</section>', re.S)
LIGNE = re.compile(r'<div class="el vc( gov)?( fixe)?"(?: data-calcul="([^"]*)"[^>]*)?><div class="l1"><b>(.*?)</b>'
                   r'(?:<span class="prof"><span>(.*?)</span></span>)?(?:<span class="pct (\w)"[^>]*>(.*?)</span>|<span class="et-mot (\w)">(.*?)</span>)'
                   r'</div><div class="l2"><span class="cmp">(.*?)</span>', re.S)


def blocs_html(frag):
    """[(classe, rang, titre, état, efforts, [lignes], html)] — une ligne : dict(gov, calcul, nom, piece, niv, pct, mot, cmp)."""
    out = []
    for cls, rang, corps in BLOC.findall(frag):
        titre = texte(re.search(r'<span class="n">(.*?)</span>', corps, re.S).group(1))
        etat = texte(re.search(r'<span class="e">(.*?)</span>', corps, re.S).group(1))
        efforts = texte(re.search(r'<div class="et-bloc-ef">(.*?)</div>', corps, re.S).group(1))
        lignes = [dict(gov=bool(g), calcul=html.unescape(calc or ""), nom=texte(nom), piece=texte(piece or ""), niv=n1 or n2, pct=texte(pct or ""),
                       mot=texte(mot or ""), cmp=texte(cmp_)) for g, _, calc, nom, piece, n1, pct, n2, mot, cmp_ in LIGNE.findall(corps)]
        out.append((cls, int(rang), titre, etat, efforts, lignes, corps))
    return out


def niveau_attendu(v):
    return {"warn": "v", "nok": "r"}.get(v["etat"]) or C.niveau(v["taux"])


def fautes_resultat(mod, etat, cas):
    """Le résultat rendu par ``mod`` face au moteur ; rend la liste des fautes (vide : tout est juste)."""
    etat = mod.normaliser(copy.deepcopy(etat))
    rep = mod.resultat(copy.deepcopy(etat))
    f = etat["fondations"][etat["actif"]]
    r = MO.verifier(f)
    h, fautes = rep["resultat"], []

    def faute(quoi, info=""):
        fautes.append(f"{cas} · {quoi} {info}".strip())
    # --- le verdict
    vd = re.search(r'<div class="verdict" data-calcul="([^"]*)"[^>]*><span class="chip xl (\w)"[^>]*><small>η</small>(.*?)</span><div class="gov"><b>(.*?)</b>(.*?)</div></div>', h, re.S)
    if not vd:
        return [f"{cas} · pas de verdict"]
    gouv = max((v for v in r["verifications"] if v["taux"] is not None), key=lambda v: v["taux"])
    if html.unescape(vd.group(1)) != gouv["cle"]:
        faute("le verdict n'ouvre pas la vérification qui gouverne", (vd.group(1), gouv["cle"]))
    if nombres(texte(vd.group(3)))[:1] != [float(C.pct(r["taux_max"]))]:
        faute("le taux du verdict", (texte(vd.group(3)), r["taux_max"]))
    attendu = {"warn": "v", "nok": "r"}.get(r["statut"]) or C.niveau(r["taux_max"])
    if vd.group(2) != attendu:
        faute("la couleur du verdict", (vd.group(2), attendu, r["statut"]))
    suite = texte(vd.group(5))
    if f"état du classeur : {gouv['etat_classeur']}" not in suite:
        faute("l'état du classeur au verdict", suite)
    pas_ok = [v for v in r["verifications"] if v["etat"] != "ok"]
    zones = [i for b in r["blocs"] for i in b["informations"] if i["etat"] in ("warn", "nok") and i.get("zone")]
    if bool(pas_ok or zones) != ("À regarder" in suite):
        faute("« À regarder » au verdict", suite)
    for v in pas_ok:
        if v["etat_classeur"] not in suite.split("À regarder")[-1]:
            faute("« À regarder » ne nomme pas", v["cle"])
    # --- les blocs du classeur
    bl = blocs_html(h)
    if [b[1] for b in bl] != list(range(len(bl))):
        faute("les rangs des blocs", [b[1] for b in bl])
    for i, b in enumerate(r["blocs"]):
        if i >= len(bl):
            faute("bloc absent", b["cle"])
            continue
        cls, _, titre, etat_txt, efforts, lignes, corps = bl[i]
        if cls != b["etat"]:
            faute(f"la couleur du bloc {b['cle']}", (cls, b["etat"]))
        if titre != mod.COURTS_BLOC.get(b["cle"], b["nom"]):
            faute(f"le titre du bloc {b['cle']}", titre)
        taux = [v["taux"] for v in b["verifications"] if v["taux"] is not None]
        if not etat_txt.startswith({"ok": "Conforme", "warn": "À vérifier", "nok": "Non conforme"}[b["etat"]]):
            faute(f"l'état du bloc {b['cle']}", etat_txt)
        if taux and not proche(nombres(etat_txt)[-1], 100 * max(taux), 1):
            faute(f"le taux du bloc {b['cle']}", (etat_txt, max(taux)))
        if texte(html.escape(b["piece"])) not in efforts:
            faute(f"la pièce du bloc {b['cle']}", efforts)
        el = next((e for e in r["elements"] if e["cle"] == b["cle"]), None)
        if el:
            for k in ("M_max", "M_min", "V_max"):
                x = el["dimensionnant"].get(k)
                if x is not None and not any(proche(n, x, 1) for n in nombres(efforts)):
                    faute(f"{k} du bloc {b['cle']}", (efforts, x))
        par_cle = {li["calcul"]: li for li in lignes}
        for v in b["verifications"]:
            li = par_cle.get(v["cle"])
            if li is None:
                faute("ligne absente", v["cle"])
                continue
            if li["nom"] != mod.COURTS.get(v["cle"].rsplit(".", 1)[-1], v["nom"]):
                faute(f"le nom de {v['cle']}", li["nom"])
            if li["gov"] != (v["cle"] == gouv["cle"]):
                faute(f"la marque « gouverne » de {v['cle']}")
            if nombres(li["pct"])[:1] != [float(C.pct(v["taux"]))]:
                faute(f"le taux de {v['cle']}", (li["pct"], v["taux"]))
            if li["niv"] != niveau_attendu(v):
                faute(f"la couleur de {v['cle']}", (li["niv"], niveau_attendu(v), v["etat_classeur"]))
            nd = FG.DECIMALES.get(v["unite"], 2)
            lus = nombres(li["cmp"])
            if len(lus) < 2 or not proche(lus[0], v["valeur"], nd) or not proche(lus[1], v["limite"], nd):
                faute(f"la comparaison de {v['cle']}", (li["cmp"], v["valeur"], v["limite"]))
            if ("≤" in li["cmp"]) != (v["valeur"] <= v["limite"]) or (">" in li["cmp"]) == (v["valeur"] <= v["limite"]):
                faute(f"le signe de la comparaison de {v['cle']}", li["cmp"])
            if (v["etat_classeur"] in li["cmp"]) != (v["etat_classeur"] != "OK"):
                faute(f"l'état du classeur sur la ligne de {v['cle']}", (li["cmp"], v["etat_classeur"]))
            m = re.match(r"(\d+) brins Ø(\d+) – (\d+)", v["piece"])
            if m and not all(x in nombres(li["piece"]) for x in map(float, m.groups())):
                faute(f"la pièce de {v['cle']}", (li["piece"], v["piece"]))
            if not m and "brins" not in v["piece"] and li["piece"] != texte(html.escape(v["piece"])):
                faute(f"la pièce de {v['cle']}", (li["piece"], v["piece"]))
        for k, info in enumerate(b["informations"]):
            if "formule" in info:
                li = par_cle.get(f"{b['cle']}.i{k}")
                if li is None:
                    faute("valeur calculée absente", (b["cle"], info["nom"]))
                    continue
                if not proche(nombres(li["cmp"])[0], info["valeur"], 3):
                    faute(f"la valeur de {info['nom']} ({b['cle']})", (li["cmp"], info["valeur"]))
                if li["niv"] != {"ok": "p", "warn": "v", "nok": "r", "info": "c"}[info["etat"]] or li["mot"].lower() != (info.get("zone") or "valeur").lower():
                    faute(f"l'état de {info['nom']} ({b['cle']})", (li["niv"], li["mot"], info.get("zone")))
            elif info.get("valeur") is not None and not any(proche(n, info["valeur"], 1) for n in nombres(texte(corps))):
                faute(f"l'information {info['nom']} ({b['cle']})", info["valeur"])
            elif info.get("texte") and texte(html.escape(info["texte"])) not in texte(corps):
                faute(f"le texte de l'information {info['nom']} ({b['cle']})")
    # --- les contrôles de géométrie, puis les écarts
    n = len(r["blocs"])
    if r["controles"]:
        if n >= len(bl) or bl[n][2] != "Contrôles de géométrie":
            faute("le bloc des contrôles de géométrie manque")
        else:
            ko = sum(1 for c in r["controles"] if not c["ok"])
            if bl[n][0] != ("nok" if ko else "ok") or len(bl[n][5]) != len(r["controles"]):
                faute("les contrôles de géométrie", (bl[n][0], len(bl[n][5])))
            for c, li in zip(r["controles"], bl[n][5]):
                lus = nombres(li["cmp"])
                if not (proche(lus[0], c["valeur"], 3) and proche(lus[1], c["limite"], 3)) or (li["mot"] == "OK") != c["ok"] or li["niv"] != ("p" if c["ok"] else "r"):
                    faute(f"le contrôle {c['cle']}", (li, c["valeur"], c["limite"]))
        n += 1
    if r["ecarts"]:
        if n >= len(bl) or not bl[n][2].startswith("Écarts"):
            faute("le bloc des écarts manque")
        elif len(bl[n][5]) != len(r["ecarts"]):
            faute("le nombre d'écarts", (len(bl[n][5]), len(r["ecarts"])))
        else:
            for e, li in zip(r["ecarts"], bl[n][5]):
                if not li["nom"].startswith(e["code"] + " ·") and li["nom"] != e["code"]:
                    faute(f"le code de l'écart {e['code']}", li["nom"])
                if texte(html.escape(e["titre"])) not in li["cmp"]:
                    faute(f"le titre de l'écart {e['code']}", li["cmp"])
                if isinstance(e["classeur"], float) and isinstance(e["redemontre"], float):
                    suite = li["cmp"].split("—")[-1]
                    lus = nombres(suite)
                    if "classeur" not in suite or len(lus) < 2 or not proche(lus[0], e["classeur"], 2) or not proche(lus[1], e["redemontre"], 2):
                        faute(f"les deux valeurs de l'écart {e['code']}", (suite, e["classeur"], e["redemontre"]))
                if li["calcul"] != "ecarts":
                    faute(f"l'écart {e['code']} n'ouvre pas la fenêtre des écarts")
        n += 1
    if len(bl) != n:
        faute("un bloc de trop", [b[2] for b in bl])
    # --- le pied : l'export
    ctl = X.controle(r)
    boutons = re.findall(r'data-telecharger="([^"]*)"', h)
    attendus = (["rbar"] if ctl["pret"] else []) + ["dxf", "tableau"] + (["cas"] if f["type"] == EN.SEMELLES8 else [])
    if boutons != attendus:
        faute("les boutons d'export", (boutons, attendus))
    if ("Export vers RBAR bloqué" in h) != (not ctl["pret"]):
        faute("le message « export bloqué »")
    # --- le résumé de la note
    if rep["resume"]["taux"] != C.pct(r["taux_max"]) or rep["resume"]["niveau"] != attendu:
        faute("le résumé de la note", rep["resume"])
    return fautes


# ====================================================================== 1. la note neuve
print("=== 1. La note neuve")
e0 = FG.nouvel_etat()
r0 = req(action="calculer", etat=copy.deepcopy(e0))
chk("la note neuve : version 1, une fondation à 8 pieux sur semelles, aux valeurs du classeur BASE (chantier et grue vides)",
    e0["version"] == 1 and len(e0["fondations"]) == 1 and e0["fondations"][0]["type"] == EN.SEMELLES8 and e0["fondations"][0]["projet"] == ""
    and {k: v for k, v in e0["fondations"][0].items() if k not in ("nom", *EN.CHAMPS_TEXTE)}
    == {k: v for k, v in EN.exemple("BASE").items() if k not in ("nom", *EN.CHAMPS_TEXTE)})
chk("elle se calcule ; la réponse porte le résultat, les fenêtres, trois vues (plan, poutre support, semelle), les données",
    r0.get("ok") and [v["cle"] for v in r0["vues"]] == ["plan", "mv_support", "mv_semelle"] and all(r0[k] for k in ("resultat", "volets", "donnees")), r0.get("erreur"))
secs = re.findall(r'<div class="sec( open)?" data-sec="(\w+)"><div class="sechd" data-replier="1"><span class="n">(\d)</span><span class="tt">(.*?)</span>', r0["donnees"])
chk("six rubriques : Fondation, Matériaux, Géométrie, Tolérances d'exécution, Charges de la grue, Armatures choisies",
    [(s[1], s[3]) for s in secs] == [("fo", "Fondation"), ("ma", "Matériaux"), ("ge", "Géométrie"), ("to", "Tolérances d'exécution"),
                                     ("ch", "Charges de la grue"), ("ar", "Armatures choisies")], secs)
chk("le module est au registre du site, sur l'établi compact, à l'adresse « fondations-grues »", FG.MODULE == "fondation_grue" and FG.PAGE == "fondations-grues")

# ====================================================================== 2. le résultat face au moteur
print("\n=== 2. Le résultat face au moteur")
ETATS = {
    "neuve": e0,
    "classeur BASE": note(exemple="BASE"),
    "classeur Greenwood": note(exemple="GWW"),
    "classeur 4 pieux": note(exemple="VAR4"),
    "classeur 8 pieux primaires": note(exemple="VAR8"),
    "paroi de pieux sécants": note(("cote_b", EN.COTE_PAROI), exemple="GWW"),
    "pieu PAS OK (capacité 40 T)": note(("pieux/capacite", "40"), exemple="GWW"),
    "armatures PAS OK (2Ø20)": note(("support/inf/0/n", "2"), exemple="GWW"),
    "étriers trop espacés": note(("semelle/etriers/pas", "400"), exemple="GWW"),
    "au-delà de la zone 3 (poutre de 60 de large)": note(("support/b", "60"), exemple="GWW"),
    "4 pieux, pieu trop chargé": note(("pieux/capacite", "300"), exemple="VAR4"),
    "8 pieux primaires, béton C20/25": note(("beton", "C20/25"), exemple="VAR8"),
    "géométrie non respectée (entraxe 150)": note(("pieux/entraxe", "150"), exemple="GWW"),
    "un seul cas de la grue": note(*[("charges/cas/1/-", "1")] * 5, exemple="GWW"),
}
statuts = {cas: MO.verifier(e["fondations"][0])["statut"] for cas, e in ETATS.items()}
chk("les états couvrent ok, warn et nok ; un état LIMITE ; un cisaillement au-delà de la zone 3 ; un contrôle de géométrie non respecté",
    set(statuts.values()) == {"ok", "warn", "nok"}
    and any(v["etat_classeur"] == "LIMITE" for e in ETATS.values() for v in MO.verifier(e["fondations"][0])["verifications"])
    and any(i.get("zone") == "> ZONE 3" for e in ETATS.values() for b in MO.verifier(e["fondations"][0])["blocs"] for i in b["informations"])
    and any(not c["ok"] for e in ETATS.values() for c in MO.verifier(e["fondations"][0])["controles"]), statuts)
toutes = [x for cas, e in ETATS.items() for x in fautes_resultat(FG, e, cas)]
chk(f"le verdict, chaque ligne, chaque bloc, les contrôles, les écarts, l'export, le résumé : ceux du moteur ({len(ETATS)} états)", not toutes, toutes[:6])


# ====================================================================== 3. chaque champ relié
print("\n=== 3. Chaque champ relié")
CHAMPS = re.compile(r'<input data-chemin="([^"]+)" data-sorte="(nombre|entier|texte)"[^>]*?value="([^"]*)"')
MENUS = re.compile(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>(.*?)</select>', re.S)


def sortie(etat):
    """Tout ce que le moteur calcule pour la fondation ouverte (sans la fondation elle-même)."""
    return json.dumps({k: v for k, v in MO.verifier(etat["fondations"][0]).items() if k not in ("fondation", "armatures", "pieces")}, sort_keys=True)


def fautes_champs(mod, exemple):
    """Chaque champ numérique du cadre Données : une autre valeur → l'état au chemin dit, le calcul qui suit (ou un refus
    du moteur, l'état intact) ; chaque menu de diamètre : un autre diamètre."""
    fautes = []
    e = mod.normaliser(note(exemple=exemple))
    donnees = mod.resultat(copy.deepcopy(e))["donnees"]
    for chemin, sorte, valeur in CHAMPS.findall(donnees):
        if sorte == "texte":
            continue
        avant = EN.lire(e["fondations"][0], chemin)
        x = avant + 1 if sorte == "entier" else round(avant * 1.1 + 1, 3)
        e2 = copy.deepcopy(e)
        msg = mod.modifier(e2, chemin.split("/"), C.nb(x, 3))
        if msg is None:
            if EN.lire(e2["fondations"][0], chemin) != x:
                fautes.append((exemple, chemin, "valeur non écrite"))
            if sortie(e2) == sortie(e) and chemin not in SANS_EFFET.get(exemple, ()):
                fautes.append((exemple, chemin, "sans effet sur le calcul"))
        elif e2 != e or not msg.startswith("Saisie refusée"):
            fautes.append((exemple, chemin, msg))
    for chemin, options in MENUS.findall(donnees):
        if not chemin.endswith("/d"):
            continue
        avant = EN.lire(e["fondations"][0], chemin)
        autre = next(d for d in EN.DIAMETRES if d != avant and d >= 12)
        e2 = copy.deepcopy(e)
        msg = mod.modifier(e2, chemin.split("/"), str(autre))
        if msg is not None or EN.lire(e2["fondations"][0], chemin) != autre:
            fautes.append((exemple, chemin, msg))
    return fautes


# les champs que le classeur lit sans les faire entrer dans une valeur calculée (constaté, et dit dans l'étude § 5)
SANS_EFFET = {"VAR4": ("poutre/etriers_travee/brins",), "BASE": (), "GWW": (), "VAR8": ()}
fc = [x for ex in ("BASE", "VAR4", "VAR8") for x in fautes_champs(FG, ex)]
chk("chaque champ numérique et chaque menu de diamètre des trois schémas écrit l'état à son chemin et change le calcul (ou est refusé par le moteur)",
    not fc, fc[:6])
e = note(exemple="GWW")
e["fondations"][0]["nom"] = "Grue G1"
avant = copy.deepcopy(e["fondations"][0])
chk("changer de schéma : la fondation repart des valeurs de son classeur ; le nom, le chantier, la grue et les matériaux sont gardés",
    FG.modifier(e, ["acier"], "BE400") is None and FG.modifier(e, ["type"], EN.CADRE4) is None and e["fondations"][0]["type"] == EN.CADRE4
    and [e["fondations"][0][k] for k in ("nom", "projet", "grue", "beton", "acier")] == ["Grue G1", avant["projet"], avant["grue"], avant["beton"], "BE400"]
    and e["fondations"][0]["poutre"] == EN.exemple("VAR4")["poutre"], e["fondations"][0])
e_meme = copy.deepcopy(e)
chk("redire le même schéma ne remet rien à zéro ; un schéma inconnu est refusé",
    FG.modifier(e, ["poutre", "b"], "95") is None and FG.modifier(e, ["type"], EN.CADRE4) is None and e["fondations"][0]["poutre"]["b"] == 95.0
    and "inconnu" in (FG.modifier(e, ["type"], "radier") or ""))
chk("« reprendre un classeur » : toutes ses valeurs, le nom gardé ; le choix vide ne fait rien ; un classeur inconnu est refusé",
    FG.modifier(e, ["exemple"], "VAR8") is None and {k: v for k, v in e["fondations"][0].items() if k != "nom"} == {k: v for k, v in EN.exemple("VAR8").items() if k != "nom"}
    and e["fondations"][0]["nom"] == "Grue G1" and FG.modifier(e, ["exemple"], "") is None and "inconnu" in (FG.modifier(e, ["exemple"], "XYZ") or ""))
e = note(exemple="GWW")
n_cas = len(e["fondations"][0]["charges"]["cas"])
chk("un cas de la grue de plus (« CAS 7 », en service, à zéro), puis de moins ; le dernier ne se retire pas ; au plus douze",
    FG.modifier(e, ["charges", "cas", "+"], "1") is None and e["fondations"][0]["charges"]["cas"][-1] == {"nom": "CAS 7", "etat": "service", "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    and FG.modifier(e, ["charges", "cas", "0", "-"], "1") is None and len(e["fondations"][0]["charges"]["cas"]) == n_cas
    and e["fondations"][0]["charges"]["cas"][0]["nom"] == "CAS 2"
    and "au moins un cas" in (FG.modifier(note(*[("charges/cas/1/-", "1")] * 5, exemple="GWW"), ["charges", "cas", "0", "-"], "1") or "")
    and "Au plus" in (FG.modifier(note(*[("charges/cas/+", "1")] * 6, exemple="GWW"), ["charges", "cas", "+"], "1") or ""))
chk("l'état d'un cas (en service / hors service) et son nom se saisissent ; un état inconnu est refusé",
    FG.modifier(e, ["charges", "cas", "0", "etat"], "hors") is None and e["fondations"][0]["charges"]["cas"][0]["etat"] == "hors"
    and FG.modifier(e, ["charges", "cas", "0", "nom"], "Tempête") is None and e["fondations"][0]["charges"]["cas"][0]["nom"] == "Tempête"
    and (FG.modifier(e, ["charges", "cas", "0", "etat"], "panne") or "").startswith("Saisie refusée"))

# ====================================================================== 4. les refus
print("\n=== 4. Les refus")
REFUS = [("GWW", "e_support", "80", "excentricité de la poutre support"), ("GWW", "chassis", "0", "largeur du châssis"),
         ("GWW", "support/inf/0/d", "17", "Ø parmi"), ("GWW", "support/inf/0/n", "2,5", "entier"), ("GWW", "pieux/D", "abc", "illisible"),
         ("GWW", "beton", "C12/15", "C12/15"), ("GWW", "poutre/b", "90", "Champ inconnu"), ("VAR4", "e", "500", "excentricité"),
         ("VAR8", "e_primaire", "90", "excentricité de la poutre primaire"), ("GWW", "semelle/L", "100", "semelle"),
         ("GWW", "charges/cas/9/A", "5", "Champ inconnu"), ("VAR4", "charges/cas/+", "1", "pas de tableau des cas")]


def fautes_refus(mod):
    out = []
    for ex, chemin, valeur, mot in REFUS:
        e = mod.normaliser(note(exemple=ex))
        avant = copy.deepcopy(e)
        msg = mod.modifier(e, chemin.split("/"), valeur)
        if not msg or mot not in msg or e != avant:
            out.append((ex, chemin, valeur, msg, e == avant))
    return out


fr_ = fautes_refus(FG)
chk(f"{len(REFUS)} saisies hors domaine : refusées avec le message du moteur, la fondation intacte", not fr_, fr_[:4])
chk("par le routeur : le message arrive à l'écran, l'état rendu est celui d'avant",
    (lambda rep, e: rep.get("ok") and "excentricité" in rep.get("message", "") and rep["etat"] == e)(
        req(action="modifier", etat=note(exemple="GWW"), chemin=["e_support"], valeur="80"), note(exemple="GWW")))
abime = {"version": 1, "fondations": [{"type": "semelles8", "chassis": -5}], "actif": 0}
chk("une note abîmée (châssis négatif), une liste qui n'en est pas une, une version plus récente : une erreur lisible, jamais une exception",
    all(rep.get("ok") is False and isinstance(rep.get("erreur"), str) and "Traceback" not in rep["erreur"] for rep in (
        req(action="calculer", etat=abime), req(action="calculer", etat={"version": 1, "fondations": "x"}),
        req(action="calculer", etat={"version": 9, "fondations": [EN.exemple("GWW")]}))))
chk("une note sans fondation est une note neuve ; un indice ouvert hors de la liste ouvre la dernière",
    FG.normaliser({"version": 1}) == FG.nouvel_etat() and FG.normaliser({"version": 1, "fondations": [EN.exemple("GWW")], "actif": 7})["actif"] == 0)

# ====================================================================== 5. l'échappement
print("\n=== 5. L'échappement")
PIEGE = '<img src=x onerror=alert(1)>'


def fautes_echappement(mod):
    e = mod.normaliser(note(exemple="GWW"))
    for chemin in ("nom", "projet", "grue", "fleche"):
        assert mod.modifier(e, [chemin], PIEGE) is None
    assert mod.modifier(e, ["charges", "cas", "0", "nom"], "<b>x</b>&") is None
    rep = mod.resultat(e)
    frags = {"resultat": rep["resultat"], "volets": rep["volets"], "donnees": rep["donnees"], **{"vue " + v["cle"]: v["svg"] + v["entete"] for v in rep["vues"]}}
    propre = {k: h.replace(html.escape(PIEGE), "").replace(html.escape(PIEGE, quote=False), "") for k, h in frags.items()}
    # (le texte piégé doit se LIRE, échappé, là où la fondation est nommée : le résultat, les fenêtres, les données)
    return [k for k, h in frags.items() if "<img" in h or "<b>x</b>" in h or re.search(r"<script|\son[a-z]+\s*=", propre[k], re.I)
            or (k in ("resultat", "volets", "donnees") and html.escape(PIEGE, quote=False) not in h.replace("&#x27;", "'").replace("&quot;", '"'))]


fe = fautes_echappement(FG)
chk("un nom, un chantier, une grue et un nom de cas piégés sont échappés dans le résultat, les fenêtres, les données et les dessins", not fe, fe)
e = note(exemple="GWW")
FG.modifier(e, ["projet"], PIEGE)
f_rbar = FG.telecharger(e, "rbar", {})
chk("dans les fichiers : le JSON RBAR garde le texte tel quel (une donnée), le DXF n'écrit que de l'ASCII",
    json.loads(f_rbar["texte"])["note"]["projet"] == PIEGE and FG.telecharger(e, "dxf", {})["texte"].isascii())


# ====================================================================== 6. les fenêtres de calcul
print("\n=== 6. Les fenêtres de calcul")
def fenetre(volets, ident):
    m = re.search(r'<div class="flyout calcul" id="ck-' + re.escape(html.escape(ident)) + r'">(.*?)(?=<div class="flyout calcul" id="ck-|$)', volets, re.S)
    return m.group(1) if m else None


def ligne_formule(fx):
    """La ligne attendue d'une formule du moteur, en texte : « symbole = expression = valeurs = résultat cellule » —
    l'écriture commune à l'écran et à la note (``ecriture.py`` ; tests/test_fondation_grue_moteur.py relit ce qu'elle
    imprime)."""
    parts, res = W.morceaux(fx)
    return " ".join((" = ".join(texte(C.symb_math(p)) for p in parts) + " = " + res + (" " + fx["cellule"] if fx.get("cellule") else "")).split())


def lignes_fx(frag):
    return [texte(re.sub(r"</span>", " </span>", x)) for x in re.findall(r'<div class="fx">(.*?)</div>', frag, re.S)]


def fautes_fenetres(mod, etat, cas):
    etat = mod.normaliser(copy.deepcopy(etat))
    rep = mod.resultat(copy.deepcopy(etat))
    r = MO.verifier(etat["fondations"][0])
    volets, fautes = rep["volets"], []
    annonces = {html.unescape(x) for x in re.findall(r'data-calcul="([^"]*)"', rep["resultat"])}
    manquent = [a for a in annonces if fenetre(volets, a) is None]
    if manquent:
        fautes.append((cas, "fenêtres annoncées et absentes", manquent))
    for i, b in enumerate(r["blocs"]):
        soll = [ligne_formule(f) for f in b["sollicitations"]]
        tout = fenetre(volets, f"s{i}") or ""
        attendu_tout = list(soll)
        for v in b["verifications"]:
            w = fenetre(volets, v["cle"])
            if w is None:
                fautes.append((cas, "pas de fenêtre", v["cle"]))
                continue
            attendu = soll + [ligne_formule(f) for f in v["formules"]]
            attendu_tout += [ligne_formule(f) for f in v["formules"]]
            if lignes_fx(w) != attendu:
                fautes.append((cas, "formules de " + v["cle"], [x for x in attendu if x not in lignes_fx(w)][:2], [x for x in lignes_fx(w) if x not in attendu][:2]))
            tab = texte(re.search(r"<tbody>(.*?)</tbody>", w, re.S).group(1))
            nd = FG.DECIMALES.get(v["unite"], 2)
            if f"classeur : {v['etat_classeur']} ({v['cellule_etat']})" not in tab or not any(proche(n, v["valeur"], nd) for n in nombres(tab)) \
                    or not any(proche(n, v["limite"], nd) for n in nombres(tab)) or not any(proche(n, float(C.pct(v["taux"])), 0) for n in nombres(tab)):
                fautes.append((cas, "tableau de " + v["cle"], tab))
        for k, info in enumerate(b["informations"]):
            if "formule" in info:
                w = fenetre(volets, f"{b['cle']}.i{k}")
                attendu_tout.append(ligne_formule(info["formule"]))
                if w is None or lignes_fx(w) != soll + [ligne_formule(info["formule"])]:
                    fautes.append((cas, "fenêtre de la valeur calculée", b["cle"], info["nom"]))
        if lignes_fx(tout) != attendu_tout:
            fautes.append((cas, f"tout le calcul du bloc {b['cle']}", len(lignes_fx(tout)), len(attendu_tout)))
    # les écarts : les deux valeurs et le message du moteur
    if r["ecarts"]:
        w = fenetre(volets, "ecarts") or ""
        for e in r["ecarts"]:
            if texte(html.escape(e["message"])) not in texte(w) or texte(html.escape(e["titre"])) not in texte(w) or (e.get("cellule") or "") not in texte(w):
                fautes.append((cas, "écart", e["code"]))
    # le contrôle avant export : une ligne par armature, sa valeur exportée
    w = fenetre(volets, "export") or ""
    for a in r["armatures"]:
        if texte(html.escape(a["texte"])) not in texte(w):
            fautes.append((cas, "contrôle avant export", a["designation"]))
    if ("prêt à partir" in w) != X.controle(r)["pret"]:
        fautes.append((cas, "l'état de l'export"))
    # les cas de la grue : pour chaque position, le maximum de chaque grandeur et sa colonne
    if "tableau_cas" in r:
        w = texte(fenetre(volets, "cas") or "")
        for t in r["tableau_cas"]:
            for k in ("RA", "RB", "V_max", "V_central", "M_max", "M_max_eq"):
                if not any(proche(n, t["max"][k]["valeur"], 1) for n in nombres(w)) or t["max"][k]["libelle"].strip("()") not in w:
                    fautes.append((cas, "cas de la grue", t["sous_cas"], k))
    elif fenetre(volets, "cas") is not None:
        fautes.append((cas, "une fenêtre des cas pour un schéma sans tableau"))
    return fautes


ff = [x for cas in ("neuve", "classeur Greenwood", "classeur 4 pieux", "classeur 8 pieux primaires", "pieu PAS OK (capacité 40 T)") for x in fautes_fenetres(FG, ETATS[cas], cas)]
n_fx = sum(len(b["sollicitations"]) + sum(len(v["formules"]) for v in b["verifications"]) for ex in ("BASE", "VAR4", "VAR8") for b in MO.verifier(EN.exemple(ex))["blocs"])
chk(f"chaque fenêtre annoncée existe ; chaque formule du moteur y est à la lettre, avec sa cellule ({n_fx} formules sur les trois schémas) ; "
    "le tableau, les écarts, le contrôle avant export, les cas de la grue", not ff, ff[:4])
chk("les pastilles d'hypothèses : la méthode du classeur, le béton et σ'b, l'acier, σ_adm et μ, les zones de cisaillement, le classeur de référence",
    all(x in texte(fenetre(r0["volets"], "pieux")) for x in ("contraintes admissibles", "C30/37", "12,96", "BE500", "333,33", "0,1708", "0,56 / 1,12 / 1,68", "BASE")))


# ====================================================================== 7. les vues
print("\n=== 7. Les vues")
def textes_svg(svg):
    racine = ET.fromstring(svg)
    return racine, [("".join(t.itertext())).translate(ESPACES) for t in racine.iter("{http://www.w3.org/2000/svg}text")]


def fr(x, nd=1):
    s = f"{x:.{nd}f}"
    return (s.lstrip("-") if float(s) == 0 else s).replace(".", ",").replace("-", "−")


def fautes_vues(mod, etat, cas, sc=SC):
    etat = mod.normaliser(copy.deepcopy(etat))
    f = etat["fondations"][0]
    r = MO.verifier(f)
    fautes = []
    vues = [dict(cle=c, titre=t, svg=sc.svg(s, "x_" + c, t)) for c, t, _, _, s, _ in sc.vues(r)]
    if [v["cle"] for v in vues] != ["plan"] + ["mv_" + el["cle"] for el in r["elements"]]:
        return [(cas, "les vues", [v["cle"] for v in vues])]
    racine, textes = textes_svg(vues[0]["svg"])
    g = G.plan(f)
    cercles = len(list(racine.iter("{http://www.w3.org/2000/svg}circle")))
    n_paroi = 0
    if g["paroi"]:
        y = g["paroi"]["y0"]
        while y <= g["paroi"]["y1"] + 1e-9:
            n_paroi += 1
            y += 0.8 * g["paroi"]["D"]
    if cercles != len(g["pieux"]) + n_paroi:
        fautes.append((cas, "le plan : autant de cercles que de pieux", cercles, len(g["pieux"]) + n_paroi))
    for k in g["cotes"]:
        if k["texte"] not in textes:
            fautes.append((cas, "le plan : une cote manque", k["texte"]))
    if sorted(t for t in textes if t in EN.PIEDS) != list(EN.PIEDS):
        fautes.append((cas, "le plan : les quatre pieds A, B, C, D"))
    for el, v in zip(r["elements"], vues[1:]):
        _, textes = textes_svg(v["svg"])
        env, dim = el["enveloppe"], el["dimensionnant"]
        i = max(range(len(env["M_max"])), key=lambda k: env["M_max"][k])
        attendus = [f"{fr(env['M_max'][i])} kN·m à x = {fr(env['x'][i], 2)} m"]
        j = max(range(len(env["V_max"])), key=lambda k: env["V_max"][k])
        attendus.append(f"{fr(env['V_max'][j])} kN à x = {fr(env['x'][j], 2)} m")
        attendus.append(f"classeur : M max = {fr(dim['M_max'])}")
        if dim.get("M_min") is not None:
            attendus.append(f"classeur : M min = {fr(dim['M_min'])}")
        if not any(t.startswith("classeur : V max = ") and proche(abs(nombres(t)[-1]), dim["V_max"], 1) for t in textes):
            fautes.append((cas, v["cle"], "le tranchant du classeur manque"))
        for a in attendus:
            if a not in textes:
                fautes.append((cas, v["cle"], "texte attendu absent", a))
        if sum(1 for t in textes if t in [p["titre"].replace("CAS ", "") for p in el["positions"]]) != len(el["positions"]):
            fautes.append((cas, v["cle"], "une rangée par position de tolérance"))
    return fautes


fv = [x for cas in ("neuve", "classeur Greenwood", "classeur 4 pieux", "classeur 8 pieux primaires", "paroi de pieux sécants") for x in fautes_vues(FG, ETATS[cas], cas)]
chk("le plan (autant de cercles que de pieux, la paroi figurée, les cotes, les pieds A B C D) ; les diagrammes (une rangée par position, les extremums "
    "de l'enveloppe et les valeurs du classeur : ceux du moteur) ; des SVG bien formés", not fv, fv[:4])


def fautes_entetes(mod, etat, cas):
    etat = mod.normaliser(copy.deepcopy(etat))
    rep = mod.resultat(copy.deepcopy(etat))
    r = MO.verifier(etat["fondations"][0])
    fautes = []
    for v in rep["vues"]:
        ET.fromstring(v["svg"])
        bande = re.search(r'<div class="et-eff">(.*?)</div>', v["entete"], re.S)
        el = next((e for e in r["elements"] if "mv_" + e["cle"] == v["cle"]), None)
        if (el is None) != (bande is None):
            fautes.append((cas, v["cle"], "la bande des efforts"))
        elif el is not None:
            lus = nombres(texte(re.sub(r"</b>|</span>", " ", bande.group(1))))
            for k in ("M_max", "M_min", "V_max"):
                x = el["dimensionnant"].get(k)
                if x is not None and not any(proche(n, x, 1) for n in lus):
                    fautes.append((cas, v["cle"], k, lus, x))
    return fautes


fh = [x for cas in ("classeur Greenwood", "classeur 4 pieux", "classeur 8 pieux primaires") for x in fautes_entetes(FG, ETATS[cas], cas)]
chk("l'en-tête de chaque vue de poutre ou de semelle porte SES sollicitations (celles que le classeur retient) ; le plan n'en porte pas", not fh, fh[:4])

# ====================================================================== 8. la géométrie du plan de principe
print("\n=== 8. La géométrie du plan de principe")


def fautes_geometrie(g_mod=G):
    fautes = []
    f = EN.exemple("GWW")
    g = g_mod.plan(f)
    h, a, b, s = f["chassis"] / 2, f["dist_a"], f["dist_b"], f["pieux"]["entraxe"]
    attendus = sorted((x, y + k * s / 2) for x in (-h - a, h + b) for y in (h, -h) for k in (1, -1))
    if sorted((round(p["x"], 6), round(p["y"], 6)) for p in g["pieux"]) != attendus or g["paroi"] is not None:
        fautes.append("8 pieux sur semelles : deux pieux par semelle, à ± s/2 de l'axe de la poutre, aux distances a et b des pieds")
    f2 = dict(f, cote_b=EN.COTE_PAROI)
    g2 = g_mod.plan(f2)
    if len(g2["pieux"]) != 4 or any(p["x"] > 0 for p in g2["pieux"]) or not g2["paroi"] or abs(g2["paroi"]["x"] - (h + b)) > 1e-9:
        fautes.append("paroi de pieux sécants : 4 pieux du côté a, la paroi à la distance b")
    f4 = EN.exemple("VAR4")
    g4 = g_mod.plan(f4)
    h4 = f4["chassis"] / 2
    if sorted((p["x"], p["y"]) for p in g4["pieux"]) != sorted((x, y) for x in (-h4, h4) for y in (-h4, h4)) or len(g4["pieces"]) != 4:
        fautes.append("4 pieux en cadre : un pieu sous chaque pied, quatre poutres")
    f8 = EN.exemple("VAR8")
    g8 = g_mod.plan(f8)
    h8, s8 = f8["chassis"] / 2, f8["pieux"]["entraxe"]
    bon = len(g8["pieux"]) == 8
    for k in range(0, 8, 2):
        p, q = g8["pieux"][k], g8["pieux"][k + 1]
        mx, my = (p["x"] + q["x"]) / 2, (p["y"] + q["y"]) / 2
        bon = bon and abs(abs(mx) - h8) < 1e-9 and abs(abs(my) - h8) < 1e-9 and abs(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) - s8) < 1e-9 \
            and abs((p["x"] - q["x"]) * mx + (p["y"] - q["y"]) * my) < 1e-6
    if not bon:
        fautes.append("8 pieux sous primaires : deux pieux à s/2 de chaque angle, sur la perpendiculaire à la diagonale")
    for nom, gg in (("GWW", g), ("paroi", g2), ("VAR4", g4), ("VAR8", g8)):
        x0, y0, x1, y1 = gg["etendue"]
        pts = [q for p in gg["pieces"] for q in p["contour"]] + [c[k] for c in gg["cotes"] for k in ("a", "b")]
        if any(not (x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9) for x, y in pts):
            fautes.append(f"{nom} : une pièce ou une cote sort de l'étendue du plan")
        if sorted((p["nom"], p["x"], p["y"]) for p in gg["pieds"]) != [("A", -gg["chassis"] / 2, gg["chassis"] / 2), ("B", gg["chassis"] / 2, gg["chassis"] / 2),
                                                                      ("C", gg["chassis"] / 2, -gg["chassis"] / 2), ("D", -gg["chassis"] / 2, -gg["chassis"] / 2)]:
            fautes.append(f"{nom} : les pieds A, B, C, D aux angles du châssis")
    return fautes


fg_ = fautes_geometrie()
chk("les pieux là où le schéma les met (semelles, paroi, cadre, primaires à 45°) ; les pieds aux angles du châssis ; rien ne sort de l'étendue", not fg_, fg_)

# ====================================================================== 9. l'export
print("\n=== 9. L'export")


def fautes_export(mod=FG, x_mod=X):
    fautes = []
    deux_groupes = note(("support/inf/1/n", "2"), ("semelle/sup/1/n", "3"), exemple="GWW")["fondations"][0]
    for ex, f in [(ex, EN.exemple(ex)) for ex in ("BASE", "GWW", "VAR4", "VAR8")] + [("GWW à deux groupes de barres", deux_groupes)]:
        r = MO.verifier(f)
        d = x_mod.rbar(r, {"date": "11/11/2026", "indice": "A", "chantier": "Essai"})
        if d["format"] != "stab.fondation_grue" or d["note"]["classeur"] != r["classeur"] or d["note"]["date"] != "11/11/2026" or d["materiaux"] != {"beton": f["beton"], "acier": f["acier"]}:
            fautes.append((ex, "l'en-tête du fichier RBAR"))
        par = {}
        for a in r["armatures"]:
            par.setdefault(a["element"], {})[a["poste"]] = a
        attendus = []
        for piece in r["pieces"]:
            p = par.get(piece["cle"], {})
            for z in ("etriers_extremites", "etriers_appui", "etriers", "etriers_centre", "etriers_travee"):
                if z in p:
                    attendus.append((piece, p, p[z]))
        if len(d["modelesPoutre"]) != len(attendus):
            fautes.append((ex, "un modèle RBAR par pièce et par zone d'étriers", len(d["modelesPoutre"]), len(attendus)))
            continue
        for m, (piece, p, et) in zip(d["modelesPoutre"], attendus):
            gi, gs = p["inf"]["groupes"], p["sup"]["groupes"]
            if m["bas"] != {"n": gi[0]["n"], "d": gi[0]["d"]} or m["haut"] != {"n": gs[0]["n"], "d": gs[0]["d"]}:
                fautes.append((ex, m["nom"], "les barres basses et hautes"))
            if m.get("basSuivants", []) != [{"n": g["n"], "d": g["d"]} for g in gi[1:]] or m.get("hautSuivants", []) != [{"n": g["n"], "d": g["d"]} for g in gs[1:]]:
                fautes.append((ex, m["nom"], "le second groupe de barres"))
            if f is deux_groupes and piece["cle"] == "support" and m.get("basSuivants") != [{"n": 2, "d": 16}]:
                fautes.append((ex, m["nom"], "le second groupe de barres basses (2Ø16) part en 2e lit"))
            if f is deux_groupes and piece["cle"] == "semelle" and m.get("hautSuivants") != [{"n": 3, "d": 16}]:
                fautes.append((ex, m["nom"], "le second groupe de barres hautes (3Ø16) part en 2e lit"))
            if m["etrier"] != {"d": et["d"], "espacement": et["pas"], "brins": et["brins"]}:
                fautes.append((ex, m["nom"], "les étriers", m["etrier"]))
            if ("enrobage" in m) != ("enrobage" in f) or m.get("enrobage") != f.get("enrobage"):
                fautes.append((ex, m["nom"], "l'enrobage"))
            if m["domaine"] != {"largeurMin_mm": piece["b"] * 10, "largeurMax_mm": piece["b"] * 10, "hauteurMin_mm": piece["h"] * 10, "hauteurMax_mm": piece["h"] * 10}:
                fautes.append((ex, m["nom"], "le domaine (largeur et hauteur de la pièce, en mm)"))
            if f["nom"] not in m["nom"] or piece["nom"] not in m["nom"]:
                fautes.append((ex, m["nom"], "le nom du modèle"))
        px = d["pieux"][0]
        if px["diametre_mm"] != f["pieux"]["D"] * 10 or ("capacite" in f["pieux"] and abs(px["charge_t"] - (f["pieux"]["capacite"] / (1 if f["type"] == EN.SEMELLES8 else 10))) > 1e-9):
            fautes.append((ex, "le type de pieu", px))
        g = G.plan(f)
        if [(p["nom"], p["x"], p["y"], p["diametre"]) for p in d["implantation"]["pieux"]] != [(p["nom"], round(p["x"] * 10, 3), round(p["y"] * 10, 3), p["D"] * 10) for p in g["pieux"]]:
            fautes.append((ex, "l'implantation des pieux en mm"))
        if d["armatures"] != r["armatures"] or d["controle"]["pret"] is not True:
            fautes.append((ex, "le tableau des armatures du fichier"))
        # le DXF : des paires code / valeur, un cercle par pieu au rayon réel, de l'ASCII
        lignes = x_mod.dxf(r, {}).split("\n")
        if lignes[-1] != "" or len(lignes[:-1]) % 2 or lignes[:4] != ["0", "SECTION", "2", "HEADER"] or lignes[-3:-1] != ["0", "EOF"] or not "\n".join(lignes).isascii():
            fautes.append((ex, "la forme du DXF"))
        ents = [(lignes[i], lignes[i + 1]) for i in range(0, len(lignes) - 1, 2)]
        rayons = [float(next(v for c, v in ents[k + 1:k + 8] if c == "40")) for k, e in enumerate(ents) if e == ("0", "CIRCLE")]
        if len(rayons) != len(g["pieux"]) or any(abs(x - f["pieux"]["D"] / 2) > 1e-6 for x in rayons):
            fautes.append((ex, "les cercles des pieux du DXF", rayons[:3]))
        # le tableau des armatures (CSV)
        csv = x_mod.tableau(r).strip().split("\n")
        if not csv[0].startswith("Élément;Poste;Armature") or len(csv) != 1 + sum(len(a["groupes"]) if a["genre"] == "barres" else 1 for a in r["armatures"]):
            fautes.append((ex, "le tableau des armatures (CSV)", len(csv)))
    # les cas de la grue (CSV) : une ligne par colonne des quatre grands tableaux, les valeurs du moteur
    r = MO.verifier(EN.exemple("GWW"))
    csv = [li.split(";") for li in x_mod.tableau_cas(r).strip().split("\n")]
    cols = [(t, c) for t in r["tableau_cas"] for c in t["colonnes"]]
    if len(csv) != 1 + len(cols) or len(cols) != 4 * 48:
        fautes.append(("GWW", "les cas de la grue (CSV) : 4 × 48 colonnes", len(csv)))
    else:
        for li, (t, c) in zip(csv[1:], cols):
            lus = [float(x.replace(",", ".")) for x in li[5:11] + li[12:]]
            if li[0] != t["sous_cas"] or li[2] != c["libelle"] or any(abs(a - b) > 6e-4 for a, b in zip(lus, [c[k] for k in ("P1", "P2", "H4", "RA", "RB", "RB_resp", "V_max", "V_central", "M_hor_eq", "M_max", "M_max_eq")])):
                fautes.append(("GWW", "une ligne des cas de la grue", li[:3]))
                break
    # le refus : une vérification PAS OK → pas de fichier RBAR, le bouton retiré, le message à l'écran
    for cas in ("pieu PAS OK (capacité 40 T)", "au-delà de la zone 3 (poutre de 60 de large)"):
        e = mod.normaliser(copy.deepcopy(ETATS[cas]))
        r = MO.verifier(e["fondations"][0])
        try:
            x_mod.rbar(r, {})
            fautes.append((cas, "le fichier RBAR part malgré une vérification PAS OK"))
        except ValueError as err:
            if not str(err).startswith("Export refusé"):
                fautes.append((cas, "le message du refus", str(err)))
    return fautes


fx_ = fautes_export()
chk("le fichier RBAR des quatre classeurs : un modèle de poutre par pièce et par zone d'étriers, ses barres, son second groupe, ses étriers, l'enrobage du "
    "seul classeur qui en a un, le type de pieu, l'implantation en mm ; le DXF ; les deux CSV ; le refus quand une vérification est PAS OK", not fx_, fx_[:4])
rep = req(action="telecharger", etat=copy.deepcopy(ETATS["pieu PAS OK (capacité 40 T)"]), quoi="rbar")
chk("par le routeur : l'export refusé rend le message du module (« Export refusé : … PAS OK »), jamais une exception",
    rep.get("ok") is False and rep.get("erreur", "").startswith("Export refusé") and "PAS OK" in rep["erreur"], rep)
noms = {q: req(action="telecharger", etat=note(("nom", "Grue G1 / Lot 2"), exemple="GWW"), quoi=q)["fichier"] for q in ("rbar", "dxf", "tableau", "cas")}
chk("les quatre fichiers : un nom sûr tiré du nom de la fondation, le type permis",
    [(f["nom"], f["type"]) for f in noms.values()] == [("Grue G1 - Lot 2 - RBAR.json", "application/json"), ("Grue G1 - Lot 2 - plan de principe.dxf", "text/plain"),
                                                       ("Grue G1 - Lot 2 - armatures.csv", "text/csv"), ("Grue G1 - Lot 2 - cas de la grue.csv", "text/csv")],
    [(f["nom"], f["type"]) for f in noms.values()])
chk("« cas » est refusé pour un schéma sans tableau des cas ; un fichier inconnu est refusé",
    req(action="telecharger", etat=note(exemple="VAR4"), quoi="cas").get("ok") is False and req(action="telecharger", etat=note(exemple="GWW"), quoi="autre").get("ok") is False)

# ====================================================================== 10. les éléments de la note
print("\n=== 10. Les éléments de la note")
e = note(exemple="VAR4")
e1 = req(action="ajouter", etat=copy.deepcopy(e))
chk("ajouter : une fondation du schéma de celle qui est ouverte, « F2 », ouverte",
    [(f["nom"], f["type"]) for f in e1["etat"]["fondations"]] == [("F1", EN.CADRE4), ("F2", EN.CADRE4)] and e1["etat"]["actif"] == 1)
e2 = req(action="dupliquer", etat=copy.deepcopy(e1["etat"]), index=0)
chk("copier : la copie s'insère juste après et s'ouvre", [f["nom"] for f in e2["etat"]["fondations"]] == ["F1", "F1 (copie)", "F2"]
    and e2["etat"]["actif"] == 1)
e3 = req(action="modifier", etat=copy.deepcopy(e2["etat"]), chemin=["pieux", "capacite"], valeur="300")
chk("le résumé de la note est celui de sa fondation la plus chargée ; une fondation non conforme rend la note non conforme (pastille)",
    e3["resume"]["niveau"] == "r" and e3["resume"]["gouvernant"].startswith("F1 (copie)") and 'class="status r"' in e3["statut_html"], e3["resume"])
e4 = req(action="supprimer", etat=copy.deepcopy(e3["etat"]), index=1)
chk("supprimer : deux fondations restent ; la dernière ne se supprime pas",
    [f["nom"] for f in e4["etat"]["fondations"]] == ["F1", "F2"]
    and len(req(action="supprimer", etat=copy.deepcopy(e), index=0)["etat"]["fondations"]) == 1)


# ====================================================================== 11. les mutations
print("\n=== 11. Les mutations : chacune vue rouge")
def mutant(module, ancien, nouveau, **remplace):
    """Le module ``module`` dont la source a ``ancien`` remplacé par ``nouveau`` (une seule occurrence) ; ``remplace`` :
    des noms du mutant posés après coup (X=<export muté>…)."""
    src = open(module.__file__, encoding="utf-8").read()
    assert src.count(ancien) == 1, (module.__name__, ancien, src.count(ancien))
    m = types.ModuleType(module.__name__ + "_mutant")
    m.__package__ = module.__package__
    m.__file__ = module.__file__
    exec(compile(src.replace(ancien, nouveau, 1), module.__file__, "exec"), m.__dict__)  # noqa: S102 — le module lui-même, muté
    m.__dict__.update(remplace)
    return m


def rouge(fn):
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 — un mutant qui plante est vu rouge aussi
        return True


def tous_resultats(m):
    return [x for cas, e in ETATS.items() for x in fautes_resultat(m, e, cas)]


def fautes_elements(m):
    e = m.normaliser(note(exemple="VAR4"))
    m.gerer_elements(e, "ajouter")
    out = [] if e["fondations"][1]["type"] == EN.CADRE4 else ["ajouter : le schéma de la fondation ouverte"]
    e = m.normaliser(note(exemple="GWW"))
    e["fondations"][0]["nom"] = "Grue G1"
    m.modifier(e, ["type"], EN.CADRE4)
    if e["fondations"][0]["grue"] != EN.exemple("GWW")["grue"]:
        out.append("changer de schéma : la grue est gardée")
    m.modifier(e, ["exemple"], "VAR8")
    if e["fondations"][0]["nom"] != "Grue G1":
        out.append("reprendre un classeur : le nom est gardé")
    return out


def fautes_fichiers(m):
    e = m.normaliser(note(exemple="GWW"))
    out = []
    if not m.telecharger(e, "tableau", {})["texte"].startswith("Élément;Poste"):
        out.append("« tableau » : le tableau des armatures")
    if not m.telecharger(e, "cas", {})["texte"].startswith("Position;Feuille"):
        out.append("« cas » : les cas de la grue")
    return out


G_ = ETATS["classeur Greenwood"]
MUTATIONS_ECRAN = [
    ("le verdict montre la première vérification, pas celle qui gouverne", 'v = next(x for x in r["verifications"] if x["cle"] == g["cle"])', 'v = r["verifications"][0]', tous_resultats),
    ("la comparaison écrit la limite à la place de la valeur", "s = f\"{a} {_n(v['valeur'], v['unite'])} {signe}", "s = f\"{a} {_n(v['limite'], v['unite'])} {signe}", tous_resultats),
    ("le signe de la comparaison est inversé", 'signe = "≤" if v["valeur"] <= v["limite"] else "&gt;"', 'signe = "≤" if v["valeur"] >= v["limite"] else "&gt;"', tous_resultats),
    ("l'état « LIMITE » du classeur n'est plus écrit sur la ligne", 'if etat and v["etat_classeur"] != K.OK:', "if False:", tous_resultats),
    ("une vérification « LIMITE » ou « PAS OK » garde la couleur de son taux", 'if v["etat"] != MO.OK:\n        return NIVEAUX_ETAT[v["etat"]]', "if False:\n        return 'p'", tous_resultats),
    ("le verdict garde la couleur du taux quand le classeur dit « PAS OK »", 'if r["statut"] != MO.OK:\n        return NIVEAUX_ETAT[r["statut"]]', "if False:\n        return 'p'", tous_resultats),
    ("tous les blocs sont verts", 'out += E.bloc(i, _nom_bloc(b), CLASSES_BLOC[b["etat"]], etat_txt,', 'out += E.bloc(i, _nom_bloc(b), "ok", etat_txt,', tous_resultats),
    ("le taux de l'en-tête d'un bloc est celui de sa première vérification", 'etat_txt = MOTS_BLOC[b["etat"]] + (f" · {_pct(max(taux))}" if taux else "")',
     'etat_txt = MOTS_BLOC[b["etat"]] + (f" · {_pct(taux[0])}" if taux else "")', tous_resultats),
    ("le cisaillement τ n'est plus montré", '            if "formule" in info:\n                rows += C.ligne_verif(info["nom"]', '            if False:\n                rows += C.ligne_verif(info["nom"]', tous_resultats),
    ("la zone de cisaillement est toujours verte", 'None, NIVEAUX_ETAT[info["etat"]],\n', 'None, "p",\n', tous_resultats),
    ("les deux valeurs d'un écart sont interverties", "cmp_ = f\"classeur {_val_ecart(e['classeur'])}{u} · redémontré {_val_ecart(e['redemontre'])}{u}\"",
     "cmp_ = f\"classeur {_val_ecart(e['redemontre'])}{u} · redémontré {_val_ecart(e['classeur'])}{u}\"", tous_resultats),
    ("le dernier écart n'est pas montré", '        rows = ""\n        for e in r["ecarts"]:\n            niv = "r" if e["change_verdict"]',
     '        rows = ""\n        for e in r["ecarts"][:-1]:\n            niv = "r" if e["change_verdict"]', tous_resultats),
    ("un contrôle de géométrie non respecté est dit « OK »", 'None, "p" if c["ok"] else "r", etat_txt="OK" if c["ok"] else "Non respecté")', 'None, "p", etat_txt="OK")', tous_resultats),
    ("l'export RBAR reste proposé quand il est bloqué", '    if ctl["pret"]:\n        boutons += C.bouton_telecharger("rbar"', '    if True:\n        boutons += C.bouton_telecharger("rbar"', tous_resultats),
    ("le résumé de la note oublie l'état « PAS OK »", "pire = next((k for k in (MO.NOK, MO.WARN) if any(y.get(\"statut\") == k for y in resultats)), MO.OK)", "pire = MO.OK", tous_resultats),
    ("la ligne d'efforts d'un bloc oublie le tranchant", '("V<sub>max</sub>", "V_max", "kN")):', "):", tous_resultats),
    ("une saisie refusée reste dans l'état", '    except (ValueError, TypeError, IndexError) as e:\n        etat["fondations"][etat["actif"]] = avant', '    except (ValueError, TypeError, IndexError) as e:\n        pass', fautes_refus),
    ("un nombre de barres décimal est accepté", '                if not float(x).is_integer():\n                    return f"Nombre entier attendu : « {valeur} »"', "                if False:\n                    return ''", fautes_refus),
    ("changer de schéma perd le chantier et la grue", 'for cle in ("projet", "grue", "beton", "acier"):', 'for cle in ("beton", "acier"):', fautes_elements),
    ("reprendre un classeur écrase le nom", '            f = MO.exemple(valeur)\n            f["nom"] = nom', "            f = MO.exemple(valeur)", fautes_elements),
    ("ajouter une fondation oublie le schéma de celle qui est ouverte", 'fs.append(MO.nouvelle_fondation(_fondation(etat)["type"], _nom_libre(fs)))', "fs.append(MO.nouvelle_fondation(S8, _nom_libre(fs)))", fautes_elements),
    ("la cellule du classeur n'est plus écrite après une formule", '''    cel = f'<span class="fg-cel" title="Cellule du classeur">{C.esc(fx["cellule"])}</span>' if fx.get("cellule") else ""''', '    cel = ""',
     lambda m: fautes_fenetres(m, G_, "GWW")),
    ("une fenêtre ne montre que la première formule de sa vérification", '            g = _groupe(v["nom"], v["formules"])', '            g = _groupe(v["nom"], v["formules"][:1])', lambda m: fautes_fenetres(m, G_, "GWW")),
    ("le tableau d'une fenêtre écrit la limite deux fois", "\"valeur\": f\"{a} = {_nu(v['valeur'], v['unite'])}\"", "\"valeur\": f\"{a} = {_nu(v['limite'], v['unite'])}\"", lambda m: fautes_fenetres(m, G_, "GWW")),
    ("la fenêtre des cas de la grue écrit le maximum de la première position partout", "g, mx = t[\"geometrie\"], t[\"max\"]", "g, mx = t[\"geometrie\"], r[\"tableau_cas\"][0][\"max\"]",
     lambda m: fautes_fenetres(m, G_, "GWW")),
    ("le message d'un écart n'est plus dans sa fenêtre", '<div class="fg-msg">{C.symb_math(e["message"])}</div>', '<div class="fg-msg"></div>', lambda m: fautes_fenetres(m, G_, "GWW")),
    ("l'en-tête d'une vue porte les sollicitations de la première pièce", 'el = next((e for e in r["elements"] if "mv_" + e["cle"] == cle), None)',
     'el = r["elements"][0] if cle != "plan" else None', lambda m: fautes_entetes(m, G_, "GWW")),
    ("le nom de la fondation n'est plus échappé dans le résultat", 'nom = C.esc(f.get("nom") or "Fondation")', 'nom = f.get("nom") or "Fondation"', fautes_echappement),
    ("« tableau » rend le fichier des cas de la grue", 'if quoi == "tableau":', 'if quoi == "tablo":', fautes_fichiers),
]
MUTATIONS_EXPORT = [
    ("le fichier RBAR part malgré une vérification PAS OK", '    if not c["pret"]:\n        raise ValueError("Export refusé', '    if False:\n        raise ValueError("Export refusé'),
    ("le second groupe de barres n'est pas exporté", "                if len(p[face][\"groupes\"]) > 1:", "                if False:"),
    ("les barres hautes exportées sont les barres basses", '"haut": _lit(p["sup"]["groupes"]),', '"haut": _lit(p["inf"]["groupes"]),'),
    ("le pas des étriers exporté est celui de la première zone", '"etrier": {"d": et["d"], "espacement": et["pas"], "brins": et["brins"]}}', '"etrier": {"d": et["d"], "espacement": p[zones[0]]["pas"], "brins": et["brins"]}}'),
    ("un enrobage est inventé pour les schémas qui n'en ont pas", '            if "enrobage" in f:\n                m["enrobage"] = f["enrobage"]', '            m["enrobage"] = f.get("enrobage", 30)'),
    ("le domaine du modèle est en cm", '"largeurMin_mm": piece["b"] * 10, "largeurMax_mm": piece["b"] * 10,', '"largeurMin_mm": piece["b"], "largeurMax_mm": piece["b"],'),
    ("le diamètre du pieu exporté est en cm", 't = {"code": f"PX{D:g}".replace(".", "_"), "diametre_mm": D * 10, "etat": "ELS"}', 't = {"code": f"PX{D:g}".replace(".", "_"), "diametre_mm": D, "etat": "ELS"}'),
    ("les pieux du DXF sont dessinés au diamètre pour rayon", '        e += _cercle(p["x"], p["y"], p["D"] / 2, C["pieux"])', '        e += _cercle(p["x"], p["y"], p["D"], C["pieux"])'),
    ("les cas de la grue (CSV) intervertissent RA et RB", '_fr(c["H4"], 3), _fr(c["RA"], 3), _fr(c["RB"], 3),', '_fr(c["H4"], 3), _fr(c["RB"], 3), _fr(c["RA"], 3),'),
    ("le tableau des armatures oublie les étriers", '        elif a["genre"] == "etriers":\n            lignes.append(', '        elif a["genre"] == "etriers":\n            (lambda *_: None)('),
]
MUTATIONS_DESSINS = [
    ("le moment coté est celui du classeur, pas l'extremum de l'enveloppe", '            texte = f"{_fr(val)} {unite} à x = {_fr(x, 2)} m"', '            texte = f"{_fr(reperes[0][1] or val)} {unite} à x = {_fr(x, 2)} m"'),
    ("les valeurs du classeur ne sont plus repérées sur les diagrammes", '            texte = f"classeur : {nom} = {_fr(val)}"', '            continue'),
    ("une position de tolérance n'est pas dessinée", "    for i, p in enumerate(pos):\n        coul, y = COULEURS.get(p[\"cle\"], ENCRE), y1 + i * RANGEE", "    for i, p in enumerate(pos[1:]):\n        coul, y = COULEURS.get(p[\"cle\"], ENCRE), y1 + i * RANGEE"),
    ("le plan oublie un pieu", '    for p in g["pieux"]:\n        cx, cy = P(p["x"], p["y"])', '    for p in g["pieux"][1:]:\n        cx, cy = P(p["x"], p["y"])'),
    ("le plan oublie les cotes", '    for k in g["cotes"]:\n        _cote(', '    for k in g["cotes"][:0]:\n        _cote('),
]
MUTATIONS_GEOMETRIE = [
    ("8 pieux sur semelles : les pieux à l'entraxe entier de l'axe", '                pieux.append({"nom": f"P{n}", "x": x, "y": y + sg * s / 2, "D": D})', '                pieux.append({"nom": f"P{n}", "x": x, "y": y + sg * s, "D": D})'),
    ("la paroi de pieux sécants garde des semelles du côté b", "    cotes_x = [(xa, True)] + ([] if paroi else [(xb, False)])", "    cotes_x = [(xa, True)] + [(xb, False)]"),
    ("8 pieux sous primaires : la primaire le long de la diagonale", "        ux, uy = sx * r2, -sy * r2 ", "        ux, uy = sx * r2, sy * r2 "),
    ("la cote n'est plus comptée dans l'étendue du plan", '    pts += [c[k] for c in g["cotes"] for k in ("a", "b")]\n', ""),
]
LISTES = ((FG, MUTATIONS_ECRAN), (X, MUTATIONS_EXPORT), (SC, MUTATIONS_DESSINS), (G, MUTATIONS_GEOMETRIE))
absents = [(mod.__name__.rsplit(".", 1)[-1], mu[0]) for mod, liste in LISTES for mu in liste if open(mod.__file__, encoding="utf-8").read().count(mu[1]) != 1]
chk("chaque mutation vise un texte présent UNE fois dans sa source (sinon elle ne mute rien)", not absents, absents)
for mod, liste in LISTES:
    liste[:] = [mu for mu in liste if open(mod.__file__, encoding="utf-8").read().count(mu[1]) == 1]
survivants = []
for titre, ancien, nouveau, controle in MUTATIONS_ECRAN:
    m = mutant(FG, ancien, nouveau)
    if not rouge(lambda controle=controle, m=m: controle(m)):
        survivants.append(titre)
for titre, ancien, nouveau in MUTATIONS_EXPORT:
    mx = mutant(X, ancien, nouveau)
    if not rouge(lambda mx=mx: fautes_export(FG, mx)):
        survivants.append(titre)
for titre, ancien, nouveau in MUTATIONS_DESSINS:
    ms = mutant(SC, ancien, nouveau)
    if not rouge(lambda ms=ms: [x for cas in ("classeur Greenwood", "classeur 4 pieux") for x in fautes_vues(FG, ETATS[cas], cas, ms)]):
        survivants.append(titre)
for titre, ancien, nouveau in MUTATIONS_GEOMETRIE:
    mg = mutant(G, ancien, nouveau)
    if not rouge(lambda mg=mg: fautes_geometrie(mg)):
        survivants.append(titre)
total = len(MUTATIONS_ECRAN) + len(MUTATIONS_EXPORT) + len(MUTATIONS_DESSINS) + len(MUTATIONS_GEOMETRIE)
chk(f"{total} mutations (l'écran : {len(MUTATIONS_ECRAN)} ; l'export : {len(MUTATIONS_EXPORT)} ; les dessins : {len(MUTATIONS_DESSINS)} ; la géométrie : "
    f"{len(MUTATIONS_GEOMETRIE)}) : TOUTES vues rouges", not survivants, survivants)
TEMOIN = ("# (retirer : une icône sur la ligne du nom — le chemin finit par « - », comme etabli_compact.retirer)", "# (témoin)")
mt = mutant(FG, *TEMOIN)
temoin_rouge = [t for t, _, _, c in MUTATIONS_ECRAN if rouge(lambda c=c: c(mt))]
chk("le témoin (un commentaire changé) n'est vu rouge par AUCUN de ces contrôles : les rouges viennent bien des mutations", not temoin_rouge, temoin_rouge)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:500])
sys.exit(1 if KO else 0)
