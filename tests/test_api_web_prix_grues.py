# -*- coding: utf-8 -*-
"""Prix des grues — l'ÉCRAN (api_web/prix_grues*.py), joué par le routeur comme le navigateur le joue, sur la base de test
(tests/donnees/prix_grues_base_test.json : prix fictifs, loueurs A/B/C).

Ce que la suite prouve :
  1. SANS BASE — l'écran propose une seule action (« Importer la base de départ ») ; aucun prix nulle part ; un fichier
     qui n'est pas une base est refusé par un message, sans rien casser.
  2. LA BASE VOYAGE DANS LE CARNET — importée, elle revient dans chaque réponse (jamais dans la note) ; l'export JSON la
     rend telle quelle (le carnet est passé au téléchargement).
  3. T1 À T4 PAR L'ÉCRAN (§ 13.1) — en tapant ce qu'un utilisateur tape (catalogue, grue, période, prix forcés et leur
     justification, choix Oui/Non dans le tableau), le total affiché est celui du test ; T2 et T3 disent « INCOMPLET ».
  4. LE PRIX FORCÉ — la case passe en orange, sa justification est demandée sous la ligne, et tant qu'elle est vide un
     point d'attention le dit ; « Prix de la base » le retire.
  5. LA BASE DES PRIX — corriger une case (historique, « Corrigé »), revenir au prix du loueur, marquer à vérifier, retirer
     une grue (elle quitte la liste du calcul), un réglage (le prix du kWh d'une nouvelle année est pris par le calcul).
  6. LES FONDATIONS — valider l'ancien retour de la case de T1 : la grille affiche son prix, « 1 chantier », et le calcul T1
     prend le nouveau prix (§ 13.5.4) ; l'écarter le retire.
  7. LES ÉCUSSONS se lisent sans la couleur (un mot ou une étiquette pour chaque bouclier).
  8. LES REFUS — un chemin inconnu, une date illisible, un texte dans une case de prix : un message, jamais une exception.
  9. MUTATIONS (copie de l'écran, contrôles rapides).

    /usr/bin/python3.12 tests/test_api_web_prix_grues.py [--rapide] [--racine dossier]
"""
import base64
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
sys.path.insert(0, RACINE_MOTEUR)

from api_web import prix_grues as PG  # noqa: E402
from api_web import routeur  # noqa: E402

PG.AUJOURDHUI = dt.date(2026, 11, 14)
BASE_TEST = os.path.join(ICI, "donnees", "prix_grues_base_test.json")
TEXTE_BASE = open(BASE_TEST, encoding="utf-8").read()
BASE = json.loads(TEXTE_BASE)
OK, KO = [], []
FINE, NBSP = " ", " "


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def euros(n: int) -> str:
    """« 181 193 € » tel que l'écran l'écrit (espace fine entre les milliers, insécable avant l'euro)."""
    s = f"{n:,}".replace(",", FINE)
    return f"{s}{NBSP}€"


class Ecran:
    """Une note du module et son carnet, comme le navigateur les tient."""

    def __init__(self):
        self.etat, self.carnet, self.r = None, [], None
        self.envoyer("nouveau")

    def envoyer(self, action, **k):
        q = {"module": "prix_grues", "action": action, "etat": self.etat, "carnet": self.carnet, **k}
        r = json.loads(routeur.traiter(json.dumps(q)))
        self.r = r
        if r.get("ok") and "etat" in r:
            self.etat = r["etat"]
            self.carnet = r.get("carnet", self.carnet)
        return r

    def m(self, chemin, valeur):
        return self.envoyer("modifier", chemin=chemin.split("/") if isinstance(chemin, str) else chemin, valeur=valeur)

    def importer(self, texte=TEXTE_BASE):
        return self.m("base/importer", {"nom": "base.json", "texte": texte})

    @property
    def tout(self):
        r = self.r
        return (r.get("donnees") or "") + (r.get("resultat") or "") + "".join(v["svg"] for v in r.get("vues") or []) + (r.get("volets") or "")


def total_affiche(e: Ecran) -> str:
    m = re.search(r'<div class="pg-total[^"]*">([^<]*) <small>HTVA', e.r["resultat"])
    return m.group(1) if m else ""


def jouer_test(e: Ecran, t: dict) -> None:
    """Un test d'acceptation joué comme un utilisateur : la grue, la période, les options, les prix forcés et leur
    justification, puis Oui sur les postes retenus et Non sur les autres, dans le tableau."""
    g = next(g for g in BASE["grues"] if g["cas"] == t["cas"] and g["modele"] == t["modele"])
    e.m("g/cas", t["cas"])
    e.m("g/grue", g["id"])
    if t.get("debut"):
        e.m("g/periode/mode", "dates")
        e.m("g/periode/debut", t["debut"])
        e.m("g/periode/fin", t["fin"])
    elif t.get("duree_mois"):
        e.m("g/periode/mode", "mois")
        e.m("g/periode/mois", str(t["duree_mois"]).replace(".", ","))
    else:
        e.m("g/periode/mode", "jours")
        e.m("g/periode/jours", str(t["duree_jours"]))
    e.m("g/heures_jour", str(t["reglages"]["heures_grutier_par_jour"]))
    if t.get("hsc_m"):
        e.m("g/hsc_m", str(t["hsc_m"]))
    if t.get("conduite"):
        e.m("g/conduite", t["conduite"])
    if t.get("fondation"):
        e.m("g/fondation/type", t["fondation"]["type"])
        e.m("g/fondation/demolition", t["fondation"]["demolition"])
    for k, v in (t.get("quantites") or {}).items():
        e.m(f"g/quantites/{k}", str(v))
    for k, v in (t.get("prix_forces") or {}).items():
        e.m(f"g/forces/{k}/prix", str(v).replace(".", ","))
        e.m(f"g/forces/{k}/justification", "test d'acceptation")
    codes = sorted(set(re.findall(r'data-chemin="g/choix/([A-Z0-9_]+)"', e.tout)))
    for c in codes:
        e.m(f"g/choix/{c}", "oui" if c in t["postes_retenus"] else "non")


# ------------------------------------------------------------------ 1-2. sans base, import, carnet, export
def sans_base():
    print("== 1. sans base · 2. la base dans le carnet")
    e = Ecran()
    chk("sans base : « Importer la base de départ » (le champ du fichier est dans les données, le centre propose l'action)",
        'data-chemin="base/importer"' in e.r["donnees"] and "Importer la base de départ" in e.tout and e.carnet == [])
    chk("sans base : aucun montant à l'écran", "€</b>" not in e.tout and not re.search(r"\d\s?€", re.sub(r"<[^>]+>", "", e.r["resultat"])))
    e.importer("pas du json")
    # (un REFUS revient en « message » — rouge à l'écran —, la note telle qu'avant ; une action réussie en « info »)
    chk("un fichier qui n'est pas du JSON : refusé par un message, le carnet reste vide", e.r["ok"] and "JSON" in (e.r["message"] or "")
        and not e.r.get("info") and e.carnet == [], e.r.get("message") or e.r.get("erreur"))
    e.importer(json.dumps({"grues": []}))
    chk("un JSON qui n'est pas une base : refusé (« il manque … »)", e.r["ok"] and "manque" in (e.r["message"] or ""), e.r.get("message"))
    e.importer()
    chk("la base importée : une INFORMATION (pas un refus), le carnet la porte, la NOTE ne la porte pas", e.r["ok"] and not e.r["message"]
        and "Base des prix importée" in (e.r.get("info") or "") and len(e.carnet) == 1
        and len(e.carnet[0]["grues"]) == len(BASE["grues"]) and "carnet" not in e.etat and "grues" not in json.dumps(e.etat)[:0] + "x",
        e.r.get("message"))
    chk("l'état de la note est petit (la base n'y est pas)", len(json.dumps(e.etat)) < 3000, len(json.dumps(e.etat)))
    r = e.envoyer("telecharger", quoi="base_json")
    chk("l'export JSON rend la base du carnet, telle quelle", r["ok"] and json.loads(r["fichier"]["texte"]) == e.carnet[0]
        and r["fichier"]["nom"].endswith(".json"), r.get("erreur"))
    e2 = Ecran()
    r = e2.envoyer("telecharger", quoi="base_json")
    chk("l'export sans base : refusé par un message", not r["ok"] and "aucune base" in r["erreur"], r)
    return e


# ------------------------------------------------------------------ 3. T1 à T4 par l'écran
def acceptation():
    print("== 3. T1 à T4 joués par l'écran (§ 13.1)")
    for t in BASE["tests_acceptation"]:
        e = Ecran()
        e.importer()
        jouer_test(e, t)
        a = t["attendu"]
        tot = total_affiche(e)
        chk(f"{t['id']} : l'écran affiche {euros(a['total_arrondi'])} HTVA", tot == euros(a["total_arrondi"]), (tot, euros(a["total_arrondi"])))
        inc = "INCOMPLET" in e.r["resultat"]
        chk(f"{t['id']} : {'INCOMPLET – ' + str(a['prix_manquants']) + ' prix manquant' if a['prix_manquants'] else 'complet'}",
            inc == bool(a["prix_manquants"]) and (not a["prix_manquants"] or f"{a['prix_manquants']} prix manquant" in e.r["resultat"]))
        if t["id"].startswith("T1"):
            chk("T1 : la note dit le total du projet dans son résumé (l'accueil le montre)", euros(a["total_arrondi"]) in e.r["resume"]["statut"], e.r["resume"])
            chk("T1 : l'alerte de variante et le bouton « Prendre la variante »", "variante max. 39 m suffit" in e.r["resultat"]
                and 'data-chemin="g/grue" data-valeur="G003"' in e.r["resultat"])
            e.m("g/grue", "G003")
            chk("« Prendre la variante » : l'autre grue est prise, l'alerte disparaît", "variante max. 39 m" not in e.r["resultat"])


# ------------------------------------------------------------------ 4. le prix forcé
def prix_force():
    print("== 4. le prix forcé")
    e = Ecran()
    e.importer()
    e.m("g/cas", "Grue tour - Loueur A")
    e.m("g/grue", "G004")
    e.m("g/periode/mode", "mois")
    e.m("g/periode/mois", "8")
    centre = e.r["vues"][0]["svg"]
    chk("MONT sans tarif : la ligne est manquante (rouge), « INCOMPLET – 2 prix manquants »", 'class="pg-l manq" id="pg-l-MONT"' in centre
        and "2 prix manquants" in e.r["resultat"])
    e.m("g/forces/MONT/prix", "18 000")
    centre = e.r["vues"][0]["svg"]
    chk("le prix tapé dans la case FORCE la ligne : case orange, justification demandée dessous (vide : orange)",
        'class="pg-l force" id="pg-l-MONT"' in centre and 'class="pg-just manque"' in centre and 'data-chemin="g/forces/MONT/justification"' in centre)
    chk("tant que la justification est vide, un point d'attention le dit", "Prix forcé sans justification" in e.r["resultat"])
    e.m("g/forces/MONT/justification", "devis du loueur")
    chk("justifié : plus d'alerte, « 1 prix forcé »", "Prix forcé sans justification" not in e.r["resultat"] and "1 prix forcé" in e.r["resultat"])
    e2 = Ecran()
    e2.importer()
    e2.m("g/cas", "Grue tour - Loueur A")
    e2.m("g/grue", "G004")
    e2.m("g/forces/MONT/prix", "1000")
    e2.m("g/cas", "Grue mobile - Loueur B")
    chk("changer de catalogue oublie la grue et les prix forcés de l'autre catalogue (« Choisir une grue »)",
        "Choisir une grue" in e2.r["resultat"] and not e2.etat["grues"][0]["forces"] and e2.etat["grues"][0]["grue"] is None)
    e.m("g/forces/MONT/-", "1")
    chk("« Prix de la base » retire le prix forcé : la ligne redevient manquante", 'class="pg-l manq" id="pg-l-MONT"' in e.r["vues"][0]["svg"])
    e.m("g/libres/+", "1")
    e.m("g/libres/0/libelle", "Clôture provisoire")
    e.m("g/libres/0/prix", "1 250")
    chk("une ligne libre ajoutée et chiffrée compte dans le total (groupe « Lignes libres »)", "Clôture provisoire" in e.r["vues"][0]["svg"]
        and "Lignes libres" in e.r["vues"][0]["svg"])
    e.m("g/choix/ANTICOL_LOC", "non")
    chk("Non sur l'anti-collision : la ligne et son montage passent dans « Postes non retenus »",
        re.search(r'<details class="pg-non".*id="pg-l-ANTICOL_LOC".*id="pg-l-ANTICOL_MONT"', e.r["vues"][0]["svg"], re.S) is not None)


# ------------------------------------------------------------------ 5. la base des prix
def base_des_prix():
    print("== 5. la base des prix")
    e = Ecran()
    e.importer()
    e.m("page", "base")
    chk("la page Base des prix : les trois loueurs, le premier ouvert, son catalogue au centre, le dossier à droite",
        all(f'data-valeur="Loueur {x}"' in e.r["donnees"] for x in "ABC") and "Dossier Loueur A" in e.r["resultat"]
        and 'data-chemin="b/grue/G004/location_jour_chassis_standard"' in e.r["vues"][0]["svg"])
    e.m("b/grue/G004/location_jour_chassis_standard", "200")
    chk("corriger une case : « Corrigé », l'historique a une ligne, la valeur du loueur reste dessous",
        len(e.carnet[0]["historique"]) == 1 and "Revenir au prix du loueur" in e.r["resultat"] and "Valeur du loueur" in e.r["resultat"])
    e.m("page", "calcul")
    e.m("g/cas", "Grue tour - Loueur A")
    e.m("g/grue", "G004")
    e.m("g/periode/mode", "mois")
    e.m("g/periode/mois", "1")
    chk("le calcul prend la correction (location 200 €/J × 20 j = 4 000 €), écusson « Corrigé »",
        "4 000\xa0€" in e.r["vues"][0]["svg"] and "pg-ec-corrige" in e.r["vues"][0]["svg"])
    e.m("b/revenir/grue/G004/location_jour_chassis_standard", "1")
    chk("« Revenir au prix du loueur » : la location reprend son prix", "4 000\xa0€" not in e.r["vues"][0]["svg"])
    e.m("b/verifier/grue/G004/location_jour_chassis_standard", "1")
    chk("« Marquer à vérifier » : l'écusson de la location devient « À vérifier »",
        re.search(r'id="pg-l-LOC".*?pg-ec-a_verifier', e.r["vues"][0]["svg"], re.S) is not None)
    e.m("b/retirer/G001", "1")
    chk("retirer une grue : elle quitte la liste du calcul (pas la base)", 'value="G001"' not in e.r["donnees"]
        and any(g["id"] == "G001" for g in e.carnet[0]["grues"]))
    e.m("b/reglage/prix_kwh/2026", "0,60")
    chk("un prix du kWh 2026 dans les réglages : le calcul le prend (1 300 kWh × 0,60 × 1 mois = 780 €)",
        "780\xa0€" in e.r["vues"][0]["svg"], re.findall(r"id=\"pg-l-ELEC\".{0,900}", e.r["vues"][0]["svg"], re.S)[:1])
    e.m("page", "base")
    e.m("vb/tableau", "historique")
    chk("l'historique liste les changements (correction, retour, à vérifier, retrait, réglage)", e.r["vues"][0]["svg"].count("<tr><td>") >= 5)
    e.m("vb/tableau", "postes")
    chk("les accessoires : une ligne par poste, une colonne par catalogue", 'data-chemin="b/poste/GRUTIER/0"' in e.r["vues"][0]["svg"])


# ------------------------------------------------------------------ 6. fondations
def fondations():
    print("== 6. les fondations (§ 13.5.4)")
    e = Ecran()
    e.importer()
    t1 = next(t for t in BASE["tests_acceptation"] if t["id"].startswith("T1"))
    jouer_test(e, t1)
    e.m("page", "fondations")
    e.m("vf/demolition", "avec")
    m4 = "Liebherr 125 EC-B 6 Litronic (> 39m HSC)"
    e.m("vf/case", f"{m4}|Dalle sur sol")
    prix = BASE["rex_fondations_historique"][1]["prix_reel"]
    chk("la case de T1 (dalle, avec démolition) : l'ancien tableau, un retour « À vérifier » avec Valider et Écarter",
        "ancien tableau" in e.r["resultat"] and 'data-chemin="b/fond/retour/H02" data-valeur="valide"' in e.r["resultat"])
    e.m("b/fond/retour/H02", "valide")
    chk("validé : la case prend le prix du chantier, « 1 chantier · année inconnue »",
        euros(int(prix)) in e.r["resultat"] and "1 chantier · année inconnue" in e.r["resultat"], re.sub(r"<[^>]+>", " ", e.r["resultat"])[:300])
    e.m("page", "calcul")
    attendu = round(t1["attendu"]["materiel"] + t1["attendu"]["main_oeuvre"] - 13200 + prix)
    chk(f"T1 prend la fondation validée : {euros(attendu)}", total_affiche(e) == euros(attendu), total_affiche(e))
    e.m("page", "fondations")
    e.m("b/fond/retour/H02", "ecarte")
    e.m("page", "calcul")
    chk("écarté : T1 revient au tableau", total_affiche(e) == euros(t1["attendu"]["total_arrondi"]), total_affiche(e))
    e.m("page", "fondations")
    e.m("b/fond/corriger", "20 000")
    chk("notre correction de la case : « Corrigé », 20 000 €", "pg-ec-corrige" in e.r["resultat"] and euros(20000) in e.r["resultat"])
    e.m("b/fond/corriger", "")
    chk("« Revenir au calcul automatique »", "pg-ec-corrige" not in e.r["resultat"])


# ------------------------------------------------------------------ 7-8. écussons, refus
def ecussons_et_refus():
    print("== 7. écussons · 8. refus")
    e = Ecran()
    e.importer()
    t1 = next(t for t in BASE["tests_acceptation"] if t["id"].startswith("T1"))
    jouer_test(e, t1)
    html = e.tout
    boucliers = re.findall(r'<span class="pg-(ec|pt) pg-ec-[a-z_]+[^"]*"[^>]*>(.*?)</span>(?:</span>)?', html)
    sans_mot = [b for b in re.findall(r'<span class="pg-ec pg-ec-[a-z_]+[^"]*"[^>]*>.*?</svg><span>([^<]*)</span>', html) if not b.strip()]
    points = re.findall(r'<span class="pg-pt pg-ec-[a-z_]+"[^>]*>', html)
    chk(f"chaque écusson porte son mot ({len(boucliers)} vus) et chaque point son étiquette (aria-label)",
        boucliers and not sans_mot and all('aria-label="' in p for p in points), (len(sans_mot), [p for p in points if "aria-label" not in p][:2]))
    pictos = {k: v for k, v in re.findall(r'<span class="pg-(?:ec|pt) pg-ec-([a-z_]+)[^"]*"[^>]*><svg[^>]*><path class="b"[^>]*/>'
                                         r'<g class="p">(.*?)</g></svg>', html)}
    chk("chaque bouclier porte un PICTOGRAMME, différent d'un écusson à l'autre (lisible imprimé en noir et blanc)",
        len(pictos) >= 2 and all(v.strip() for v in pictos.values()) and len(set(pictos.values())) == len(pictos), pictos)
    avant = json.dumps(e.etat)
    for chemin, valeur, mot in (("g/inconnu", "1", "champ inconnu"), ("g/periode/debut", "31/02/2026", "n'est pas une date"),
                                ("g/heures_jour", "beaucoup", "n'est pas un nombre"), ("g/choix/LOC", "peut-être", "Auto, Oui ou Non"),
                                ("page", "ailleurs", "page inconnue"), ("g/grue", "G999", "pas dans le catalogue")):
        e.m(chemin, valeur)
        chk(f"refus « {chemin} = {valeur} » : un message ({mot}), la note intacte",
            e.r["ok"] and mot in (e.r["message"] or "") and json.dumps(e.etat) == avant, e.r.get("message") or e.r.get("erreur"))
    e.m("page", "base")
    e.m("b/grue/G004/location_jour_chassis_standard", "environ 150")
    chk("un texte qui n'est pas un prix dans une case de la base : refusé par un message", e.r["ok"] and "n'est pas un prix" in (e.r["message"] or ""))
    e.m("b/grue/G004/location_jour_chassis_standard", "sur demande")
    chk("« sur demande » est accepté (une correction de notre part : « Corrigé », la case dit « sur demande »)",
        e.r["ok"] and "pg-ec-corrige" in e.r["resultat"] and 'value="sur demande"' in e.r["vues"][0]["svg"])
    n = PG.normaliser({"grues": [{"repere": "X" * 50, "inconnu": 1, "periode": {"mode": "mois", "mois": 3}}], "page": "nulle", "actif": 9})
    chk("normaliser : un état d'ailleurs relu (repère coupé, clé inconnue écartée, page et grue ouverte valides)",
        n["page"] == "calcul" and n["actif"] == 0 and len(n["grues"][0]["repere"]) == PG.REPERE_MAX and "inconnu" not in n["grues"][0]
        and n["grues"][0]["periode"]["mois"] == 3)


def texte_pdf(octets: bytes) -> str:
    p = subprocess.run(["pdftotext", "-layout", "-", "-"], input=octets, capture_output=True)
    return p.stdout.decode("utf-8", "replace")


def rapport():
    print("== 10. le rapport PDF (§ 11.1, § 11.2, § 13.4) · « À faire » · « ? »")
    e = Ecran()
    r = e.envoyer("telecharger", quoi="note_pdf")
    chk("sans base : le rapport est refusé par un message", not r["ok"] and "importer la base" in r["erreur"], r.get("erreur"))
    e.importer()
    r = e.envoyer("telecharger", quoi="note_pdf")
    chk("sans grue choisie : refusé par un message", not r["ok"] and "choisir" in r["erreur"], r.get("erreur"))
    t1 = next(t for t in BASE["tests_acceptation"] if t["id"].startswith("T1"))
    jouer_test(e, t1)
    e.m("g/forces/DEMONT/justification", "")
    detail = e.r["vues"][0]["svg"].split('<details class="pg-non"')[0]      # (l'écran AVANT le téléchargement)
    r = e.envoyer("telecharger", quoi="note_pdf", infos={"nom": "Projet essai", "chantier": "Chantier essai"})
    pdf = base64.b64decode(r["fichier"]["base64"]) if r.get("ok") else b""
    txt = re.sub(r"[ \u00a0\u202f]+", " ", texte_pdf(pdf))
    retenus = [ln for ln in re.findall(r'id="pg-l-([A-Z0-9_]+)"', detail)]
    libelles = re.findall(r'<span class="nm">([^<]*)</span>', detail)
    chk(f"T1 : un PDF ({len(pdf)} octets), « Ce qui est calculé · Grue tour (T) » et son texte, les hypothèses",
        pdf[:5] == b"%PDF-" and "Ce qui est calculé · Grue tour (T)" in txt and "Mois = mois complets + fraction du mois entamé" in txt
        and "HYPOTHÈSES" in txt and "Jours facturés" in txt and "Prix du kWh" in txt, r.get("erreur") or txt[:300])
    manquent = [l for l in libelles if re.sub(r"\s+", " ", l.replace("&#x27;", "'")) not in txt]
    chk(f"§ 13.4 : chaque ligne retenue ({len(libelles)}) est dans le rapport, avec sa source (année, « ancien tableau », « forcé »)",
        len(libelles) == len(retenus) and not manquent and "ancien tableau" in txt and "forcé" in txt and "2025" in txt, manquent)
    chk("le total (181 193 €), matériel et main-d'œuvre ; les prix forcés et leur justification (l'absente dite)",
        "181 193 €" in txt and "Matériel" in txt and "Main-d'œuvre" in txt and "PRIX FORCÉS (2)" in txt and "test d'acceptation" in txt
        and "SANS JUSTIFICATION" in txt)
    chk("la légende des écussons en bas de page (les mots : À jour, À vérifier, Pas à jour, Manquant, Corrigé)",
        all(m in txt for m in ("À jour", "À vérifier", "Pas à jour", "Manquant", "Corrigé")))
    e.envoyer("ajouter")
    t4 = next(t for t in BASE["tests_acceptation"] if t["id"].startswith("T4"))
    jouer_test(e, t4)
    r = e.envoyer("telecharger", quoi="rapport_projet", infos={"nom": "Projet essai", "chantier": "Chantier essai"})
    txt = re.sub(r"[ \u00a0\u202f]+", " ", texte_pdf(base64.b64decode(r["fichier"]["base64"]))) if r.get("ok") else ""
    somme = t4["attendu"]["total_arrondi"] + 181193
    chk(f"le rapport du projet : la synthèse (GT1, GT2), le total du projet ({somme} €), puis le rapport de chaque grue",
        "Total du projet HTVA" in txt and f"{somme:,} €".replace(",", " ") in txt and txt.count("Ce qui est calculé") == 2
        and "Ce qui est calculé · Grue mobile, grutier compris (MA)" in txt, r.get("erreur") or txt[:400])
    e.envoyer("activer", index=0)
    r = e.envoyer("telecharger", quoi="note_pdf", infos={"nom": "Façade « Est »", "chantier": "Résidence & Cie", "indice": "C", "date": "03/04/2026"})
    txt = re.sub(r"[ \u00a0\u202f]+", " ", texte_pdf(base64.b64decode(r["fichier"]["base64"]))) if r.get("ok") else ""
    chk("l'identification de la note du site dans l'en-tête du rapport : « chantier · nom · date · indice »",
        "Résidence & Cie · Façade « Est » · 03/04/2026 · indice C" in txt, r.get("erreur") or txt[:200])
    e.envoyer("calculer")
    chk("« À faire (n) » en tête du résultat (retours à valider, questions, fichiers à préparer) et la présentation « ? »",
        "À faire (" in e.r["resultat"] and 'data-calcul="pg-aide"' in e.r["resultat"] and 'id="ck-pg-aide"' in e.r["volets"]
        and "Ce qu'il fait." in e.r["volets"].replace("&#x27;", "'"))


def echange():
    print("== 11. l'échange avec un loueur (§ 7.1, § 7.2) : préparer, envoyer, importer, contrôler, appliquer, annuler")
    from grues.prix import lecture_loueur as LL
    e = Ecran()
    e.importer()
    e.m("page", "base")
    e.m("vb/loueur", "Loueur A")
    res = e.r["resultat"]
    mail = re.search(r'href="(mailto:[^"]*)"', res)
    chk("le dossier du loueur : « Préparer le fichier 2027 », « Écrire le mail » (objet et texte du cahier), « Importer le fichier reçu » "
        "(un classeur lu en binaire)", "Préparer le fichier 2027" in res and 'data-telecharger="fichier_loueur"' in res and mail is not None
        and "Mise%20%C3%A0%20jour%20de%20vos%20prix%202027" in mail.group(1) and 'data-chemin="imp/fichier"' in res and 'data-binaire="1"' in res,
        mail.group(1)[:120] if mail else res[:200])
    avant = json.dumps(e.etat)
    e.m("vb/prep/retour", "31/02/2027")
    chk("une date de retour illisible est refusée par un message", "date" in (e.r["message"] or ""), e.r.get("message"))
    e.m("vb/prep/annee", "deux mille")
    chk("une année illisible est refusée par un message, la note EXACTEMENT celle d'avant (rien d'écrit à moitié)",
        "année" in (e.r["message"] or "") and json.dumps(e.etat) == avant, e.r.get("message"))
    e.m("vb/prep/retour", "15/12/2026")
    e.m("vb/prep/par", "J. T.")
    r = e.envoyer("telecharger", quoi="fichier_loueur")
    o = base64.b64decode(r["fichier"]["base64"]) if r.get("ok") else b""
    lu = LL.lire(e.carnet[0], o, "x.xlsx") if o else {"infos": {}, "resume": {}}
    chk("« Préparer le fichier » rend DEMANDE_PRIX_2027_LOUEUR_A.xlsx, avec « Demandé par » et la date de retour ; relu : 0 changement, "
        "186 prix confirmés", r.get("ok") and r["fichier"]["nom"] == "DEMANDE_PRIX_2027_LOUEUR_A.xlsx" and lu["infos"].get("demande_par") == "J. T."
        and lu["infos"].get("retour_avant") == "2026-12-15" and lu["resume"].get("changements") == 0 and lu["resume"].get("prix_confirmes") == 186,
        r.get("erreur") or lu["resume"])
    e.m("b/demande/envoyee", "1")
    chk("« Le fichier est parti » : une information (pas un refus), le dossier passe en « Demande envoyée »",
        "demande 2027 envoyée" in (e.r.get("info") or "") and not e.r["message"] and "pg-ec-demande" in e.r["resultat"], e.r.get("info"))
    e.m("imp/fichier", {"nom": "faux.xlsx", "base64": base64.b64encode(b"pas un classeur").decode()})
    chk("un fichier qui n'est pas un classeur : refusé par un message, on reste sur la page", "classeur" in (e.r["message"] or "")
        and e.etat["page"] == "base" and not e.etat.get("imp"), e.r.get("message"))
    rempli = base64.b64encode(open(os.path.join(ICI, "donnees", "prix_grues_loueur_a_rempli.xlsx"), "rb").read()).decode()
    e.m("imp/fichier", {"nom": "prix_grues_loueur_a_rempli.xlsx", "base64": rempli})
    centre = e.r["vues"][0]["svg"]
    choix = re.findall(r'data-chemin="imp/choix/([0-9a-f]{12})"( checked)?', centre)
    chk("le fichier reçu ouvre la page de CONTRÔLE : 9 changements cochés, la case vidée décochée, la valeur non lue à saisir, la nouvelle "
        "grue et la grue retirée cochées, l'autre prestation décochée — rien n'est appliqué",
        e.etat["page"] == "import" and "9 changement(s)" in (e.r.get("info") or "") and len(choix) == 13
        and sum(1 for _, c in choix if c) == 11 and 'data-chemin="imp/saisie/' in centre and "« ± 30 »" in centre
        and e.carnet[0]["grues"][0]["location_jour_chassis_standard"] == 132.0, (len(choix), sum(1 for _, c in choix if c), e.r.get("info")))
    chk("les baisses et les écarts au-delà de 15 % sont surlignés ; à droite les compteurs, les avertissements, « Appliquer »",
        centre.count('<tr class="fort">') == 3 and 'data-chemin="imp/appliquer"' in e.r["resultat"] and "181" in e.r["resultat"]
        and "Cases vidées par le loueur" in e.r["resultat"], centre.count('<tr class="fort">'))
    saisie = re.search(r'data-chemin="imp/saisie/([0-9a-f]{12})"', centre).group(1)
    avant = json.dumps(e.etat)
    e.m(f"imp/saisie/{saisie}", "environ 30")
    chk("une valeur non lue mal saisie est refusée par un message, la note intacte", "n'est pas un prix" in (e.r["message"] or "")
        and json.dumps(e.etat) == avant, e.r.get("message"))
    e.m(f"imp/saisie/{saisie}", "30")
    e.m("imp/filtre", "changements")
    chk("le filtre « Changements » ne montre que les changements", e.r["vues"][0]["svg"].count('data-chemin="imp/choix/') == 9)
    # la première ligne (G001, 132 → 115 €/J) DÉCOCHÉE à l'écran : elle ne s'applique pas
    premiere = re.search(r'data-chemin="imp/choix/([0-9a-f]{12})"', e.r["vues"][0]["svg"]).group(1)
    e.m(f"imp/choix/{premiere}", False)
    chk("une case décochée l'est dans la note (et le reste à l'écran)", e.etat["imp"]["choix"].get(premiere) is False
        and f'data-chemin="imp/choix/{premiere}" checked' not in e.r["vues"][0]["svg"])
    e.m("imp/appliquer", "1")
    g2 = next(g for g in e.carnet[0]["grues"] if g["id"] == "G002")
    g1 = next(g for g in e.carnet[0]["grues"] if g["id"] == "G001")
    g4 = next(g for g in e.carnet[0]["grues"] if g["id"] == "G004")
    chk("« Appliquer » : 9 prix appliqués (8 changements cochés et la valeur saisie), la ligne décochée NON appliquée (G001 garde "
        "132 €/J), l'année et la source du fichier, retour à la Base des prix",
        "9 prix appliqués" in (e.r.get("info") or "") and e.etat["page"] == "base" and g2["cabine_jour"] == 30.0 and g1["location_jour_chassis_standard"] == 132.0
        and g4["transport_montage_ff"] == 9500 and g4["annee_par_champ"]["transport_montage_ff"] == 2027 and "Annuler le dernier import" in e.r["resultat"]
        and "Dernier fichier reçu" in e.r["resultat"], e.r.get("info"))
    e.m("b/import/annuler", "1")
    g4 = next(g for g in e.carnet[0]["grues"] if g["id"] == "G004")
    chk("« Annuler le dernier import » remet la base et le dossier (« Demande envoyée »)", g4.get("transport_montage_ff") is None
        and "pg-ec-demande" in e.r["resultat"] and "Annuler le dernier import" not in e.r["resultat"], e.r.get("info"))
    e.m("page", "import")
    chk("la page de contrôle sans fichier : refusée par un message", "page inconnue" in (e.r["message"] or ""), e.r.get("message"))


def rapides():
    sans_base()
    acceptation()
    prix_force()
    base_des_prix()
    fondations()
    ecussons_et_refus()
    rapport()
    echange()


MUTATIONS = [
    ("api_web/prix_grues_calcul.py", '''        out += (f'<tr class="pg-just{"" if j else " manque"}"><td colspan="7"><label>Justification\'''',
     '''        out += (f'<tr class="pg-just{"" if j else ""}"><td colspan="7"><label>Justification\'''', "la justification vide plus signalée"),
    ("api_web/routeur.py", '''        kw["carnet"] = q.get("carnet") if isinstance(q.get("carnet"), list) else []''', '''        kw["carnet"] = []''',
     "le carnet oublié au téléchargement"),
    ("api_web/prix_grues.py", '''            s.update(cas=valeur or None, grue=None, choix={}, quantites={}, forces={})''',
     '''            s.update(cas=valeur or None)''', "le catalogue changé sans oublier la grue de l'autre catalogue"),
    ("api_web/prix_grues_base.py", '''        return ED.corriger(base, ref, valeur, "", ini, auj)''', '''        return None''', "une case de la base qui ne corrige rien"),
    ("api_web/prix_grues_fondations.py", '''        return ED.statut_retour(base, ch[3], valeur, motif, ini, auj)''', '''        return None''', "Valider un retour qui ne fait rien"),
    ("api_web/prix_grues_rapport.py", '''        for ln in ret:
            _ligne(pg, ln)''', '''        for ln in ret[:-1]:
            _ligne(pg, ln)''', "le rapport qui oublie une ligne"),
    ("api_web/prix_grues_commun.py", '''f'<g class="p">{PICTOS.get(cle, "")}</g></svg>')''', '''f'</svg>')''', "un écusson sans pictogramme"),
    ("api_web/prix_grues_echange.py", '''                if j in imp["choix"]:
                    choix[IL.cle_item(sorte, it)] = imp["choix"][j]''', '''                if False:
                    choix[IL.cle_item(sorte, it)] = imp["choix"][j]''', "« Appliquer » qui oublie les cases décochées à l'écran"),
    ("api_web/prix_grues_echange.py", '''    return j, IL.PAR_DEFAUT[sorte] if v is None else v''', '''    return j, True''',
     "l'écran qui coche tout (cases vidées comprises)"),
    ("api_web/prix_grues.py", '''        etat.clear()
        etat.update(avant)''', '''        pass''', "un refus qui laisse la note à moitié changée"),
    ("api_web/prix_grues_rapport.py", '''(f" · indice {indice}" if indice else "")''', '''""''', "le rapport sans l'indice de la note"),
]
TEMOIN = ("api_web/prix_grues.py", "PAGES = (", "PAGES = (")


def muter(fichier, ancien, nouveau):
    with tempfile.TemporaryDirectory() as tmp:
        for nom in os.listdir(RACINE):
            if nom in (".git", ".temoin_tests", "__pycache__", "node_modules", "api_web"):
                continue
            os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
        shutil.copytree(os.path.join(RACINE, "api_web"), os.path.join(tmp, "api_web"), ignore=shutil.ignore_patterns("__pycache__"))
        cible = os.path.join(tmp, fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    t0 = time.time()
    rapides()
    print(f"   (contrôles rapides : {time.time() - t0:.1f} s)")
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    print("== 9. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : l'écran recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
