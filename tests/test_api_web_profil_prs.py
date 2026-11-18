# -*- coding: utf-8 -*-
"""(lot C, C6) Le PRS À L'ÉCRAN du module « Choix de profilé » (api_web/profil_prs.py, la branche PRS de api_web/profil.py,
le croquis api_web/dessin_profil.section_prs), vérifié en Python ordinaire, à travers le routeur, comme l'écran l'appelle.

  1. LA SAISIE — « Catalogue · PRS » ; les chemins prs/b1 … prs/a, les bords ; h et la norme produit dits ; les refus
     d'une ligne de prs.section_prs (t_w ≥ b₁, c ≤ 0, dimension ≤ 0) ; la barre reste intacte.
  2. LA CASE « section soudée équivalente » — offerte au PRS bisymétrique seulement (« autres tests PRS » n° 2) ; cochée,
     le § 6.3.2.3 décide ; perdue quand le PRS devient monosymétrique.
  3. LA BIBLIOTHÈQUE — le carnet du module : enregistré puis repris dans une AUTRE note → mêmes propriétés, recalculées,
     jamais stockées (« autres tests PRS » n° 4) ; le sélecteur d'un PRS est la bibliothèque, sans ★.
  4. LES PROPRIÉTÉS — lues dans acier/barre/prs.py, jamais dans acier/profiles ; PRS-A et PRS-B du brief (11.8) ; le
     même I_t sur la page Catalogue (5,657·10⁶ mm⁴) ; le même tableau à l'écran et dans la fiche de la note.
  5. LE CROQUIS — « G », « C », « mm » ; aucun texte sur un autre ; G et C à leur cote, l'axe y par G.
  6. LE VERDICT — bords non saisis : la méthode ELS « Bloqué », sa puce jamais verte ; deux PRS de plats différents :
     deux résultats ; la masse ρ·A au bouton du profilé.
  7. MUTATIONS — chaque altération plausible de l'écran du PRS fait rougir au moins un contrôle.

    /usr/bin/python3.12 -B tests/test_api_web_profil_prs.py
"""
import html
import json
import math
import os
import re
import sys
import types
import xml.dom.minidom

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import prs as PRS  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from api_web import catalogue as CATP  # noqa: E402
from api_web import dessin_profil  # noqa: E402,F401
from api_web import profil  # noqa: E402,F401
from api_web import profil_prs  # noqa: E402,F401
from api_web import profil_saisie  # noqa: E402,F401
from api_web import registre as REG  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []
LARGEUR_CAR = 0.56          # (la place d'un caractère, en tailles de police : celle que le croquis compte, dessin_profil.LARGEUR_CAR)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def mod(nom):
    """Le module tel qu'il est CHARGÉ (une mutation le remplace le temps d'un passage)."""
    return sys.modules[f"api_web.{nom}"]


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="profil", **k))))


def texte(h):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", h)).replace(" ", " ").replace(" ", " ").replace("​", "").split())


class Ecran:
    """Une note ouverte, comme l'écran la tient : l'état, et le carnet (la bibliothèque) qui voyage avec chaque requête."""

    def __init__(self, carnet=None):
        self.carnet = list(carnet or [])
        self.r = req(action="nouveau", carnet=self.carnet)

    def __call__(self, chemin, valeur):
        self.r = req(action="modifier", etat=self.r["etat"], chemin=chemin.split("/"), valeur=valeur, carnet=self.carnet)
        if self.r.get("ok") and isinstance(self.r.get("carnet"), list):
            self.carnet = self.r["carnet"]
        return self.r

    @property
    def el(self):
        return self.r["etat"]["elements"][0]


PRS_B = (("b1", "300"), ("t1", "30"), ("hw", "540"), ("tw", "15"), ("b2", "200"), ("t2", "30"), ("a", "6"))
PRS_A = (("b1", "300"), ("t1", "30"), ("hw", "540"), ("tw", "15"), ("b2", "300"), ("t2", "30"), ("a", "6"))


def ecran_prs(plats=PRS_B, bords=None, nuance="S355", carnet=None):
    e = Ecran(carnet)
    e("nuance", nuance)
    e("prs", "prs")
    for k, v in plats:
        e(f"prs/{k}", v)
    if bords:
        e("prs/bords", bords)
    return e


def boites(svg):
    """Les textes d'un SVG, leur boîte estimée (x1, y1, x2, y2), comme le croquis les compte : [(texte, boîte)]."""
    out = []
    for m in re.finditer(r"<text ([^>]*)>(.*?)</text>", svg, re.S):
        a = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
        t = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).replace("​", "")
        x, y, fs = float(a["x"]), float(a["y"]), float(a["font-size"])
        w = len(t) * LARGEUR_CAR * fs * (1.12 if a.get("font-weight") else 1.0)
        if "rotate" in a.get("transform", ""):
            out.append((t, (x - 0.8 * fs, y - w / 2, x + 0.2 * fs, y + w / 2)))
            continue
        x1 = {"middle": x - w / 2, "end": x - w}.get(a.get("text-anchor"), x)
        out.append((t, (x1, y - 0.8 * fs, x1 + w, y + 0.2 * fs)))
    return out


def recouvrements(svg):
    b = boites(svg)
    return [(t1, t2) for i, (t1, a) in enumerate(b) for t2, z in b[i + 1:]
            if min(a[2], z[2]) - max(a[0], z[0]) > 0.01 and min(a[3], z[3]) - max(a[1], z[1]) > 0.01]


def repere(svg, lettre):
    m = re.search(rf'data-repere="{lettre}" data-cx="([-\d.]+)" data-cy="([-\d.]+)"', svg)
    return (float(m.group(1)), float(m.group(2))) if m else None


def axe_y(svg):
    m = re.search(r'<line x1="[-\d.]+" y1="([-\d.]+)" x2="[-\d.]+" y2="([-\d.]+)"[^>]*data-axe="y"', svg)
    return (float(m.group(1)), float(m.group(2))) if m else None


def verdicts(res):
    """[(titre court, classe)] des CARTES du résultat, dans l'ordre. (Mise à jour 1, § 1.4 — le harnais suit : le bloc des grosses
    pastilles est parti ; le panneau commence par le nom de la barre, la ligne « Profilé », puis les cartes.)"""
    return [(texte(n), c) for c, n in re.findall(r'<section class="et-bloc ([a-z]+)" data-theme="\d+">.*?<span class="n">(.*?)</span>', res, re.S)]


def controles():
    """[(nom, vrai ?, info)] : tout ce que ce fichier vérifie — rejoué sous chaque mutation."""
    P, PP, DP = mod("profil"), mod("profil_prs"), mod("dessin_profil")
    P._RESULTATS.clear()
    P._CANDIDATS.clear()
    out = []

    def c(nom, cond, info=""):
        out.append((nom, bool(cond), info))

    # ---------------------------------------------------------------- 1. la saisie
    e0 = Ecran()
    don0 = e0.r["donnees"]
    c("1 · une barre du catalogue : « Catalogue · PRS » sous l'acier, « Catalogue » choisi ; aucun champ de plat",
      'data-chemin="prs" data-sorte="choix" data-valeur="catalogue"' in don0 and re.search(r'class="on" data-chemin="prs"[^>]*data-valeur="catalogue"', don0)
      and 'data-chemin="prs/b1"' not in don0, re.findall(r'data-chemin="prs[^"]*"', don0))
    e = ecran_prs()
    el = e.el
    plats = {k: el["prs"][k] for k in PRS.DIMS + ("a",)}
    c("1 · les chemins prs/b1 … prs/a écrivent les plats de la barre (PRS-B : 300 × 30 / 540 × 15 / 200 × 30, a = 6) ; le nom suit : « PRS … »",
      e.r["ok"] and plats == {k: float(v) for k, v in PRS_B} and el["profil"] == "PRS 600x300x15x30x200x30" and el["prs"]["bords"] is None, (plats, el["profil"]))
    don = e.r["donnees"]
    c("1 · la rubrique d'un PRS : ses sept champs en trois lignes « b × t » (la gorge sur la ligne de l'âme), les bords, h = t₁ + h_w + t₂ = 600 mm, la norme produit EN 10025-2",
      [len(re.findall(r'data-chemin="prs/[a-z0-9]+" data-sorte="(?:nombre|texte)"', x)) for x in re.findall(r'<div class="pf-plat[^"]*">.*?</div></div>(?=<div class="pf-plat|<div class="row)', don, re.S)] == [2, 3, 2]
      and all(f'data-chemin="prs/{k}"' in don for k in ("b1", "t1", "hw", "tw", "b2", "t2", "a"))
      and 'data-chemin="prs/bords"' in don and "h = t₁ + h w + t₂ = 600 mm" in texte(don) and "EN 10025-2" in texte(don), texte(don)[:300])
    e("prs/a", "")
    c("1 · la gorge vidée : non saisie (s = 0, côté sûr) — le moteur le dit (prs.MSG_GORGE) ; le champ montre « s = 0 »",
      e.r["ok"] and e.el["prs"]["a"] is None and PRS.MSG_GORGE in texte(e.r["resultat"]) and 'placeholder="s = 0"' in e.r["donnees"], texte(e.r["resultat"])[:200])
    e("prs/a", "6")
    e("prs/bords", "larges_plats")
    b_ = e.el["prs"]["bords"]
    e("prs/bords", "")
    c("1 · les bords : « larges_plats » écrit, puis vidés (non saisis)", b_ == "larges_plats" and e.el["prs"]["bords"] is None, b_)
    avant = json.dumps(e.el, sort_keys=True)
    refus = {}
    for nom, chemin, v in (("t_w ≥ b₁", "prs/tw", "300"), ("t_w > b₂", "prs/tw", "250"), ("c ≤ 0 (gorge trop grande)", "prs/a", "200"),
                           ("dimension nulle", "prs/hw", "0"), ("dimension négative", "prs/t2", "-5"), ("texte", "prs/b1", "abc")):
        m = e(chemin, v).get("message") or ""
        refus[nom] = m
    c("1 · les refus du moteur (t_w ≥ b₁ et b₂, c ≤ 0, dimension ≤ 0, un texte) : « Saisie refusée : PRS : … » en UNE ligne de 160 caractères au plus, la barre intacte",
      all(m.startswith("Saisie refusée : ") and "\n" not in m and len(m) <= 160 for m in refus.values())
      and all("PRS : " in refus[k] for k in ("t_w ≥ b₁", "t_w > b₂", "c ≤ 0 (gorge trop grande)", "dimension nulle", "dimension négative"))
      and json.dumps(e.el, sort_keys=True) == avant, refus)
    c("1 · un sigle garde ses capitales dans un refus (« PRS : … », jamais « pRS : … »)", not any("pRS" in m for m in refus.values()), refus)
    hors = e("prs/b9", "1")
    c("1 · un champ inconnu du PRS : « Champ inconnu »", "Champ inconnu" in (hors.get("message") or ""), hors.get("message"))
    c("1 · les bords de l'écran sont ceux du moteur (prs.BORDS) : une seule liste", {k for k, _ in PP.BORDS} == set(PRS.BORDS), PP.BORDS)

    # ---------------------------------------------------------------- 2. la case « section soudée équivalente »
    a = ecran_prs(PRS_A, bords="oxycoupes")
    case_bi = 'data-chemin="prs/soudee"' in a.r["donnees"]
    case_mono = 'data-chemin="prs/soudee"' in e.r["donnees"]
    refuse = e("prs/soudee", "true").get("message") or ""
    a("travees/0", "10")
    a("maintiens/0/-", "1")
    meth0 = P.calculer(a.el)["methodes"]["ec3"]["grandeurs"]["deversement"].get("methode")
    a("prs/soudee", "true")
    coche = a.el["parametres"].get(PP.SOUDEE)
    meth1 = P.calculer(a.el)["methodes"]["ec3"]["grandeurs"]["deversement"]
    c("2 · la case « section soudée équivalente » : offerte au PRS bisymétrique, absente du monosymétrique (et refusée s'il la demande)",
      case_bi and not case_mono and "bisymétrique" in refuse, (case_bi, case_mono, refuse))
    c("2 · cochée : la table des paramètres la porte, le § 6.3.2.3 décide et le § 6.3.2.2 est rendu pour information ; décochée : le § 6.3.2.2 décide",
      coche is True and meth0 == "§ 6.3.2.2" and meth1.get("methode") == "§ 6.3.2.3" and (meth1.get("information") or {}).get("methode") == "§ 6.3.2.2",
      (coche, meth0, meth1.get("methode")))
    a("prs/b2", "200")
    c("2 · le PRS devient monosymétrique : la déclaration est retirée (jamais pour un monosymétrique)", PP.SOUDEE not in a.el["parametres"], a.el["parametres"])

    # ---------------------------------------------------------------- 3. la bibliothèque (le carnet)
    b = ecran_prs(bords="oxycoupes")
    tab0 = PP.proprietes(PP.section(b.el))
    rg = b("carnet/garder", "1")
    lib = rg.get("carnet") or []
    c("3 · « Enregistrer » : le PRS est dans la bibliothèque rendue par la réponse (jamais dans la note), un message le dit",
      rg.get("ok") and len(lib) == 1 and lib[0]["nom"] == "PRS 600x300x15x30x200x30" and "carnet" not in rg["etat"] and "enregistré" in (rg.get("info") or ""),
      (rg.get("info"), lib))
    c("3 · la bibliothèque garde les dimensions, la nuance et les bords — AUCUNE propriété (prs.vers_bibliotheque)",
      bool(lib) and set(lib[0]) <= {"famille", "dims", "nom", "nuance", "bords"} and set(lib[0]["dims"]) <= {"h", "b", "tw", "tf", "b2", "tf2", "a"}, lib)
    c("3 · la barre garde une COPIE de ses plats (clé « prs ») : la note se suffit", set(b.el["prs"]) == set(PRS.DIMS) | {"a", "bords", "nom"}, b.el["prs"])
    c("3 · enregistré, le bouton dit « Retirer de la bibliothèque »", 'data-chemin="carnet/retirer"' in rg["donnees"], "")
    autre = Ecran(lib)
    sel_cat = req(action="selecteur", etat=autre.r["etat"], role="profil", carnet=lib)
    autre("prs", "prs")
    sel = req(action="selecteur", etat=autre.r["etat"], role="profil", carnet=lib)
    h = sel.get("html") or ""
    c("3 · dans une AUTRE note, le sélecteur d'un PRS est la bibliothèque : le PRS enregistré, sa masse, sa pastille — pas de ★ (aucun « plus léger qui tient »)",
      sel.get("ok") and 'data-choix="PRS 600x300x15x30x200x30"' in h and "181,3" in h and "pk-row opt" not in h and "plus léger" not in h
      and "PRS 600x300x15x30x200x30" not in (sel_cat.get("html") or ""), texte(h)[:300])
    ch = req(action="choisir", etat=autre.r["etat"], role="profil", nom="PRS 600x300x15x30x200x30", carnet=lib)
    el2 = ch["etat"]["elements"][0] if ch.get("ok") else {}
    tab2 = PP.proprietes(PP.section(el2)) if el2 else []
    c("3 · repris de la bibliothèque : mêmes plats, même nuance, MÊMES propriétés (recalculées, jamais stockées) — autres tests PRS n° 4",
      ch.get("ok") and el2["profil"] == "PRS 600x300x15x30x200x30" and el2["prs"]["bords"] == "oxycoupes" and tab2 == tab0 and len(tab0) >= 15,
      (ch.get("erreur"), el2.get("profil")))
    c("3 · le résultat repris est celui de la barre d'origine (même calcul)", ch.get("ok") and texte(ch["resultat"]) .replace("B1", "") == texte(rg["resultat"]).replace("B1", ""),
      "")
    autre.r = ch
    autre.carnet = lib
    autre("prs/t1", "32")
    c("3 · un plat repris puis changé : le nom suit les plats (« PRS 602x… »), jamais celui de la bibliothèque", autre.el["profil"] == "PRS 602x300x15x32x200x30", autre.el["profil"])
    # (intégration du lot C, reste « important » de C6) un PRS est reconnu par son ENREGISTREMENT, pas par son nom seul : mêmes plats,
    # autre nuance, autre gorge, autres bords — « Enregistrer » proposé, aucune ligne cochée (le taux coché était celui de l'autre)
    sosie = ecran_prs(PRS_B[:-1] + (("a", "4"),), bords="larges_plats", nuance="S235", carnet=lib)
    sel_s = req(action="selecteur", etat=sosie.r["etat"], role="profil", carnet=lib)
    meme = ecran_prs(bords="oxycoupes", carnet=lib)
    sel_m = req(action="selecteur", etat=meme.r["etat"], role="profil", carnet=lib)
    c("3 · mêmes plats, en S235, a = 4, larges plats : ce n'est pas le PRS de la bibliothèque (S355, a = 6, oxycoupés) — « Enregistrer » proposé, "
      "aucune ligne du sélecteur cochée ; le même PRS : « Retirer », sa ligne cochée",
      sosie.el["profil"] == "PRS 600x300x15x30x200x30" and 'data-chemin="carnet/garder"' in sosie.r["donnees"]
      and 'data-chemin="carnet/retirer"' not in sosie.r["donnees"] and "pk-row cur" not in (sel_s.get("html") or "")
      and 'data-chemin="carnet/retirer"' in meme.r["donnees"] and "pk-row cur" in (sel_m.get("html") or ""),
      (sosie.el["profil"], "pk-row cur" in (sel_s.get("html") or ""), "pk-row cur" in (sel_m.get("html") or "")))
    # (intégration du lot C, R1) l'aide des plats et des efforts : dans l'icône .aide[title] de leur label (au doigt, etabli.js n'ouvre
    # qu'elle), jamais un title posé sur le label (lisible au seul survol, au PC)
    eff = Ecran()
    eff("mode", "efforts")
    plats_ = re.findall(r'<div class="pf-plat[^"]*"><label>(.*?)</label>', b.r["donnees"])
    bloc_ = eff.r["donnees"].partition('class="pf-eff')[2].partition("Forme des diagrammes")[0]
    effs_ = re.findall(r'<label>(<span class="sym">.*?)</label>', bloc_)
    c("3 · R1 : l'aide des trois plats et des efforts imposés est dans l'icône ⓘ (.aide[title]) de son label, lisible au doigt",
      len(plats_) == 3 and all('class="aide" title="' in x for x in plats_) and len(effs_) >= 4 and all('class="aide" title="' in x for x in effs_)
      and not re.search(r'<div class="pf-plat[^"]*"><label title=', b.r["donnees"]) and '<label title=' not in bloc_,
      (len(plats_), len(effs_)))
    rr = b("carnet/retirer", "1")
    c("3 · « Retirer de la bibliothèque » : la bibliothèque est vide, la barre garde son PRS", rr.get("carnet") == [] and b.el["profil"] == "PRS 600x300x15x30x200x30", rr.get("carnet"))
    lu = PP.lire_carnet([{"famille": "PRS", "dims": {"h": 600, "b": 300, "tw": 15, "tf": 30}}, {"famille": "PRS", "dims": {"h": 600, "b": 300, "tw": 15, "tf": 30}},
                         {"famille": "PRS", "dims": {"h": 10, "b": 300, "tw": 15, "tf": 30}}, "abc", None, {"famille": "IPE"}])
    c("3 · une bibliothèque abîmée se relit : l'illisible est écarté, un nom en double aussi", [x["nom"] for x in lu] == ["PRS 600x300x15x30"], lu)
    faux = dict(PRS.vers_bibliotheque(PRS.section_prs(300, 30, 540, 15, 200, 30, a=6, nuance="S355", bords="oxycoupes")), It=1.0, Iy=1.0, A=1.0)
    relu = PP.lire_carnet([faux])
    ch3 = req(action="choisir", etat=autre.r["etat"], role="profil", nom=faux["nom"], carnet=[faux])
    c("3 · un enregistrement qui porterait des propriétés : relu SANS elles (rendu tel que prs.vers_bibliotheque l'écrit), et le PRS repris les RECALCULE",
      relu and not {"It", "Iy", "A"} & set(relu[0]) and ch3.get("ok") and ch3.get("carnet") == relu
      and dict((x[0], x[1]) for x in PP.proprietes(PP.section(ch3["etat"]["elements"][0])))["I_t"] == "475,67", (relu, ch3.get("erreur")))
    retour = req(action="calculer", etat=b.r["etat"], carnet="abc")
    c("3 · un carnet qui n'est pas une liste : rendu vide, sans erreur", retour.get("ok") and retour.get("carnet") == [], retour.get("erreur"))

    # ---------------------------------------------------------------- 4. les propriétés
    sA = PRS.section_prs(300, 30, 540, 15, 300, 30, a=6, nuance="S355", bords="oxycoupes")
    sB = PRS.section_prs(300, 30, 540, 15, 200, 30, a=6, nuance="S355", bords="oxycoupes")
    tA, tB = (dict((x[0], x[1]) for x in PP.proprietes(s_)) for s_ in (sA, sB))
    c("4 · PRS-A (brief 11.8) : A 261 cm², I_y 166 023 cm⁴, I_t 565,67 cm⁴ (5,657·10⁶ mm⁴), I_w 10 965 000 cm⁶ (à 0,01 %), f_y des plats 345 / 355 / 345, de calcul 345",
      tA["A"] == "261" and tA["I_y"].replace(" ", " ") == "166 023" and tA["I_t"] == "565,67" and abs(float(tA["I_w"].replace(" ", "")) - 1.0965e7) / 1.0965e7 < 1e-4
      and (tA["f_y,f1"], tA["f_y,w"], tA["f_y,f2"], tA["f_y"]) == ("345", "355", "345", "345") and "z_j" not in tA and "W_el,y" in tA, tA)
    c("4 · PRS-B (brief 11.8) : z_g 337 mm, W_el,y 5 265,2 / 4 108,7 cm³, W_pl,y 5 218,5 cm³, z_s 117,7 mm, z_j 144,78 mm, ψ_f 0,5429, I_t 475,67 cm⁴",
      tB["z_g"] == "337" and tB["W_el,y,haut"].replace(" ", " ") == "5 265,2" and tB["W_el,y,bas"].replace(" ", " ") == "4 108,7"
      and tB["W_pl,y"].replace(" ", " ") == "5 218,5" and tB["z_s"] == "117,7" and tB["z_j"] == "144,78" and tB["ψ_f"] == "0,5429" and tB["I_t"] == "475,67", tB)
    cat = CATP._perso_calcule("PRS", (("h", 600.0), ("b", 300.0), ("tw", 15.0), ("tf", 30.0)))
    c("4 · une seule source de I_t : PRS-A, la page Catalogue et les Barres donnent le même I_t (565,67 cm⁴ = 5,657·10⁶ mm⁴)",
      abs(cat["It"] - sA["It"] / 1e4) < 1e-9 and abs(sA["It"] - 5.657e6) / 5.657e6 < 1e-4, (cat["It"], sA["It"]))
    origines = {x[0]: x[3] for x in PP.proprietes(sB)}
    c("4 · chaque propriété dit son origine (I_t : Σ (b·t³/3)·(1 − 0,63·t/b) ; z_j : annexe E ANB ; f_y : l'épaisseur et le tableau)",
      all(origines.values()) and "0,63" in origines["I_t"] and "annexe E" in origines["z_j"] and "tableau 3.3 ANB" in origines["f_y,f1"], origines)
    import acier.profiles.familles as FA_
    import acier.profiles.consultation as CO_
    gardes = {}
    for m_ in (FA_, CO_):
        for k_ in [k for k in dir(m_) if callable(getattr(m_, k)) and not k.startswith("_") and getattr(getattr(m_, k), "__module__", "") == m_.__name__]:
            gardes[(m_, k_)] = getattr(m_, k_)
            setattr(m_, k_, lambda *a_, **k__: (_ for _ in ()).throw(AssertionError("acier/profiles lu")))
    try:
        sans = PP.proprietes(PP.section(b.el))
        vue = P.vue_section(b.el)
        lu_ok = sans == tab0 and "565" not in vue[:0]
    except Exception as ex:  # noqa: BLE001
        lu_ok = f"exception {ex}"
    finally:
        for (m_, k_), f_ in gardes.items():
            setattr(m_, k_, f_)
    c("4 · les propriétés et le croquis d'un PRS sont lus dans acier/barre/prs.py, JAMAIS dans acier/profiles (familles, consultation interdits : même tableau)",
      lu_ok is True, lu_ok)
    fi = P.fiche_profil(b.el["profil"], b.el)
    vue = P.vue_section(b.el)
    c("4 · le MÊME tableau à l'écran (vue « Section ») et pour la note (fiche_profil : « tableau », « proprietes ») ; chaque valeur écrite dans la vue",
      fi["tableau"] == tab0 and fi["proprietes"] == [x[:3] for x in tab0] and all(texte(v_) in texte(vue) for _, v_, _, _ in tab0) and fi["famille"] == "PRS", fi.get("sous"))
    note = req(action="telecharger", etat=b.r["etat"], quoi="note_pdf", carnet=b.carnet)
    # (lot C, C5 — le harnais suit : la note d'un PRS SORT, son croquis et ses propriétés au panneau ; tests/test_note_profil.py le détaille)
    c("4 · la NOTE d'un PRS sort (bloc C5) : un PDF, jamais un KeyError", note.get("ok") is True and (note.get("fichier") or {}).get("type") == "application/pdf"
      and "KeyError" not in str(note.get("erreur")), str(note.get("erreur"))[:200])

    # ---------------------------------------------------------------- 5. le croquis
    for nom_s, s_ in (("PRS-A", sA), ("PRS-B", sB), ("PRS-C", PRS.section_prs(300, 30, 1420, 25, 500, 50, nuance="S235")),
                      ("larges semelles", PRS.section_prs(600, 20, 160, 10, 600, 20, a=5, nuance="S235", bords="larges_plats"))):
        svg = DP.section_prs(s_, None, "x", PP.proprietes(s_))
        txts = [t for t, _ in boites(svg)]
        g_, c_, y_ = repere(svg, "G"), repere(svg, "C"), axe_y(svg)
        try:
            xml.dom.minidom.parseString(svg)
            bien = True
        except Exception:  # noqa: BLE001
            bien = False
        c(f"5 · croquis {nom_s} : SVG bien formé ; « G », « C » et « mm » écrits ; G et C repérés aussi par un symbole (cercle barré, losange)",
          bien and "G" in txts and "C" in txts and any(t.startswith("Cotes en mm") for t in txts) and "<circle" in svg and g_ and c_, txts[:8])
        c(f"5 · croquis {nom_s} : aucun texte sur un autre (boîtes estimées)", not recouvrements(svg), recouvrements(svg)[:4])
        c(f"5 · croquis {nom_s} : G à sa cote (z_g = {s_['z_g']:.1f} mm), C à la sienne (z_g + z_s), l'axe y par G — pas par l'origine",
          g_ and c_ and y_ and abs(g_[1] + s_["z_g"]) < 0.01 and abs(c_[1] + s_["z_g"] + s_["z_s"]) < 0.01 and abs(y_[0] + s_["z_g"]) < 0.01 and abs(y_[1] + s_["z_g"]) < 0.01
          and g_[0] == 0 and c_[0] == 0, (g_, c_, y_))
    seul = DP.section_prs(sB, 300)
    c("5 · le croquis seul (la note écrit son tableau) : sans tableau des propriétés, à la largeur demandée", "PROPRIÉTÉS" not in seul and 'width="300"' in seul, "")
    vues = {v_["cle"]: v_ for v_ in b.r["vues"]}
    c("5 · la vue « Section » de l'écran est le croquis du PRS (G, C, son tableau) ; ses textes ne se recouvrent pas",
      "section" in vues and repere(vues["section"]["svg"], "C") and "PROPRIÉTÉS" in vues["section"]["svg"] and not recouvrements(vues["section"]["svg"])
      and vues["section"]["titre"] == "Section PRS 600x300x15x30x200x30", list(vues))

    # ---------------------------------------------------------------- 6. le verdict, la masse, deux PRS
    n = ecran_prs()
    vs = verdicts(n.r["resultat"])
    c("6 · bords non saisis : la carte ELS « Bloqué », jamais verte ; le moteur dit pourquoi en une ligne",
      bool(vs) and vs[0][0] == "ELS" and vs[0][1] != "ok" and "Bloqué" in texte(n.r["resultat"])
      and P.calculer(n.el)["methodes"]["historique"]["statut"] == RE.BLOQUE and "Bords des semelles à saisir" in texte(n.r["resultat"]), vs)
    m_ = P.calculer(n.el)["masse"]
    c("6 · la masse au bouton du profilé : ρ·A (prs.MASSE_VOLUMIQUE 7 850 kg/m³ × 231 cm²) = 181,3 kg/m",
      abs(m_ - PRS.MASSE_VOLUMIQUE * 23100e-6) < 1e-9 and re.search(r'class="pf-prof"[^>]*>.*?181,3', n.r["donnees"], re.S)
      and "bibliothèque" in re.search(r'class="pf-prof"[^>]*title="([^"]*)"', n.r["donnees"]).group(1), m_)
    d1 = ecran_prs(bords="oxycoupes")
    d2 = ecran_prs((("t1", "32"),) + PRS_B[2:], bords="oxycoupes")
    c("6 · deux PRS de plats différents : deux signatures, deux résultats (la clé « prs » est gardée, C0e)",
      P._signature(d1.el) != P._signature(d2.el) and P.calculer(d1.el)["methodes"]["ec3"]["taux_max"] != P.calculer(d2.el)["methodes"]["ec3"]["taux_max"], "")
    q = ecran_prs(PRS_A)
    q("prs/soudee", "true")
    q("prs", "catalogue")
    c("6 · retour au catalogue : le profilé en I le plus proche (HEB 600), plus de PRS ni de déclaration",
      q.el["profil"] == "HEB 600" and q.el["prs"] is None and PP.SOUDEE not in q.el["parametres"], (q.el["profil"], q.el["parametres"]))
    ch2 = req(action="choisir", etat=d1.r["etat"], role="profil", nom="IPE 330", carnet=[])
    c("6 · un profilé du catalogue choisi pour une barre PRS : le PRS est effacé", ch2.get("ok") and ch2["etat"]["elements"][0]["prs"] is None
      and ch2["etat"]["elements"][0]["profil"] == "IPE 330", ch2.get("erreur"))
    c("6 · le module déclare son carnet (la bibliothèque voyage avec la requête)", "profil" in REG.carnets(), REG.carnets())
    return out


def muter(nom_mod, ancien, nouveau):
    """Les contrôles qui rougissent quand le texte ``ancien`` du module ``nom_mod`` (« api_web.profil_prs ») devient
    ``nouveau`` ; None si le texte n'y est pas une seule fois. Le module muté remplace l'original partout où un module du
    site le tient (et dans le registre du routeur), puis l'original est remis, quoi qu'il arrive."""
    orig = sys.modules[nom_mod]
    src = open(orig.__file__, encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    m = types.ModuleType(nom_mod)
    m.__file__, m.__package__ = orig.__file__, orig.__package__
    exec(compile(src.replace(ancien, nouveau), orig.__file__, "exec"), m.__dict__)
    poses = [(x, k) for x in list(sys.modules.values()) if getattr(x, "__name__", "").startswith("api_web") and x is not orig
             for k, v in list(vars(x).items()) if v is orig]
    cle = nom_mod.rpartition(".")[2]
    dans_reg = REG._MODULES.get(cle) is orig
    try:
        for x, k in poses:
            setattr(x, k, m)
        sys.modules[nom_mod] = m
        if dans_reg:
            REG._MODULES[cle] = m
        rouges = [n for n, ok, _ in controles() if not ok]
    except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir
        rouges = [f"exception {type(e).__name__}: {e}"]
    finally:
        for x, k in poses:
            setattr(x, k, orig)
        sys.modules[nom_mod] = orig
        if dans_reg:
            REG._MODULES[cle] = orig
    return rouges


MUTATIONS = [
    ("api_web.profil_prs", "le nom ne suit plus les plats (celui de la bibliothèque reste)", '        p["nom"] = None                               # (le nom suit', '        pass  # (le nom suit'),
    ("api_web.profil_prs", "la case « soudée équivalente » offerte au monosymétrique", '    if not sec["monosymetrique"]:\n        s += C.case(', '    if True:\n        s += C.case('),
    ("api_web.profil_prs", "la déclaration gardée quand le PRS devient monosymétrique", '            el["parametres"].pop(SOUDEE, None)        # (jamais pour un monosymétrique)', '            pass'),
    ("api_web.profil_prs", "la case cochée pour un monosymétrique acceptée", '        if oui and (p["b1"], p["t1"]) != (p["b2"], p["t2"]):', '        if False:'),
    ("api_web.profil_prs", "des propriétés gardées dans la bibliothèque relue", '            out.append(PR.vers_bibliotheque(s))', '            out.append(e)'),
    ("api_web.profil_prs", "I_t écrit par Σ b·t³/3 (l'écran Streamlit, +4,2 %)", '("I_t", n(s["It"] / 1e4, 2)',
     '("I_t", n((p["b1"] * p["t1"] ** 3 + p["hw"] * p["tw"] ** 3 + p["b2"] * p["t2"] ** 3) / 3e4, 2)'),
    ("api_web.profil_prs", "z_j de (E.4) au lieu de l'intégrale exacte", '("z_j", n(zj["z_j"], 2)', '("z_j", n(zj["z_j_approche"], 2)'),
    ("api_web.profil_prs", "la bibliothèque relue sans recalcul (un enregistrement abîmé passe)", '            s = PR.de_bibliotheque(e)\n', '            s = PR.de_bibliotheque(e) if e.get("dims", {}).get("h", 0) > 100 else PR.de_bibliotheque(e)\n            out.append(e); continue\n'),
    ("api_web.profil_prs", "la nuance de la bibliothèque ignorée à la reprise", '    el["nuance"] = s["nuance"]\n    if s["monosymetrique"]:', '    pass\n    if s["monosymetrique"]:'),
    ("api_web.profil_prs", "la norme produit plus dite", '· tôles {norme(sec)} · cotes en mm', '· cotes en mm'),
    ("api_web.dessin_profil", "les axes par l'origine (mi-hauteur), pas par G", '    g.append(_ligne(-bm / 2 - 5 * u, -zg, bm / 2 + 5 * u, -zg, AXE,', '    g.append(_ligne(-bm / 2 - 5 * u, -h / 2, bm / 2 + 5 * u, -h / 2, AXE,'),
    ("api_web.dessin_profil", "C dessiné en G", '    g.append(_marque_c(0.0, -zc, u))', '    g.append(_marque_c(0.0, -zg, u))'),
    ("api_web.dessin_profil", "G sans sa lettre", '    texte(-2.6 * u, -zg - 2.0 * u, "G", 4.4 * u, ancre="end", gras=True)\n', ''),
    ("api_web.dessin_profil", "la lettre C posée sur celle de G", '    texte(2.6 * u, -zc - 2.0 * u, "C", 4.4 * u, ancre="start"', '    texte(-2.6 * u, -zg - 2.0 * u, "C", 4.4 * u, ancre="end"'),
    ("api_web.dessin_profil", "« mm » plus dit sous le croquis", '"Cotes en mm · trois plats', '"Cotes · trois plats'),
    ("api_web.profil", "le sélecteur d'un PRS ouvre le catalogue", '        return _selecteur_prs(etat, el)\n', '        pass\n'),
    ("api_web.profil", "la vue Section d'un PRS lit le catalogue", '    if el.get("prs"):\n        s = PP.section(el)', '    if False:\n        s = PP.section(el)'),
    ("api_web.profil", "le carnet plus déclaré (la bibliothèque ne voyage plus)", "CARNET = True\n", "CARNET = False\n"),
    ("api_web.profil", "un refus écrit « pRS : … » (première lettre forcée en minuscule)", '        return f"Saisie refusée : {M.minuscule(str(e))}"\n    return mot',
     '        return f"Saisie refusée : {str(e)[:1].lower() + str(e)[1:]}"\n    return mot'),
    ("api_web.profil", "un PRS de la bibliothèque choisi sans ses plats", '    if entree is not None:\n        PP.prendre(el, entree)', '    if entree is not None:\n        el["profil"] = nom'),
    ("api_web.profil", "le sélecteur d'un PRS marque une ★", '"opt": False,\n               "prop": PP.ligne_selecteur', '"opt": True,\n               "prop": PP.ligne_selecteur'),
    ("api_web.profil_saisie", "la rubrique sans le PRS (le seul bouton du catalogue)", "    return s + PP.rubrique(el, r, carnet)", "    return s + PP.bouton(el, r)"),
    # (intégration du lot C) un PRS reconnu par son nom seul dans la bibliothèque
    ("api_web.profil_prs", "la bibliothèque reconnaît un PRS à son nom seul (bouton « Retirer »)",
     "    return e is not None and any(x == e for x in lire_carnet(list(carnet or [])))",
     '    return e is not None and any(x["nom"] == e["nom"] for x in lire_carnet(list(carnet or [])))'),
    ("api_web.profil", "le sélecteur coche le PRS de même nom (son taux à la place de celui de la barre)", '"cur": e == sien,', '"cur": s["nom"] == el["profil"],'),
    ("api_web.profil_prs", "l'aide d'un plat au survol du label (illisible au doigt)",
     """<div class="pf-plat{" ame" if gorge else ""}"><label><span>{libelle}</span>""",
     """<div class="pf-plat{" ame" if gorge else ""}"><label title="{C.esc(aide)}"><span>{libelle}</span>"""),
    ("api_web.profil_saisie", "l'aide d'un effort au survol du label (illisible au doigt)",
     """        lignes += f'<label><span class="sym">{M.sub(sym)}</span><span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span></label>{champs}'""",
     """        lignes += f'<label title="{C.esc(aide)}"><span class="sym">{M.sub(sym)}</span></label>{champs}'"""),
]


def main():
    print("== 1 à 6. contrôles")
    for nom, ok, info in controles():
        chk(nom, ok, info)
    print("== 7. mutations")
    for nom_mod, nom, ancien, nouveau in MUTATIONS:
        rouges = muter(nom_mod, ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans {nom_mod}, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » ({nom_mod}) : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, les modules sont rendus : les contrôles repassent tous", all(ok for _, ok, _ in controles()))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
