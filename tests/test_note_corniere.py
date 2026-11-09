# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Cornières d'appui » sur le site (acier/corniere/note.py, api_web/note_corniere.py — le lot
écran y arrive par api_web/corniere.telecharger), vérifiée en Python ordinaire, SANS l'écran du module : l'état de la
note du site est construit par le moteur (nouvelle cornière + modifications), {"version": 2, "cornieres": [...]}.

- TOUTE la note en un PDF valide : la page de garde du site (projet, partie, date, indice, sommaire) puis deux pages par
  cornière (trois avec une page de suite), dans l'ordre de la liste, sur ≥ 8 états : A historique, A équilibre, B avec
  Y = 85 et Y = 45, ailes inégales (150 puis 100 horizontal), petite (L 40×40×4), grosse (L 250×250×35, équilibre, quatre
  ancrages), deux cornières dans une note (A avec largeur d'appui, B retournée), la garde (ancrage bas), une note v1 relue ;
- PARITÉ avec le moteur de l'écran (``moteur.verifier``), cornière par cornière, par pdftotext : le nom, la configuration
  et le mode, le statut et le taux max, chaque taux des vérifications, M1 à M4, H, N et V par ancrage (ELS et ELU), et
  chaque RÉSULTAT des formules écrites (la chaîne des sections, la coupe gouvernante, les ancrages) — aux arrondis
  d'affichage près ; les textes des dessins (R, H, L) y sont ;
- chaque FORMULE de la note (acier/corniere/formules.py) ré-évaluée deux fois, sur tous les blocs de tous les états :
  avec les nombres imprimés (l'arrondi d'affichage) et avec les termes exacts (1e-9) ;
- l'IDENTIFICATION : la note du site (chantier et nom PIÉGÉS — accents, guillemets, « & », tiret long —, indice « C »,
  date) dans la garde (PROJET, PARTIE, DATE, INDICE) et dans l'en-tête de chaque page (« 03/04/2026 · indice C »),
  jamais « indice 0 » ni « indice A » ; le nom du fichier « Note de calcul - <nom>.pdf », le contrat du routeur ;
- la PAGE DE SUITE : un bloc qui ne tient pas passe en page 3, la colonne d'origine le dit, le sommaire compte la page ;
- les REFUS : une cornière hors domaine (le message du moteur, le nom de la cornière), une note sans cornière ;
- COMME DANS LE NAVIGATEUR (sonde calquée sur celle du paquet de la note, site/web/construire._SONDE_NOTE) : sans
  fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet, la note se fait, son texte est
  celui de CPython, ses polices sont celles du paquet, et les fichiers du dépôt qu'elle charge en plus de l'écran sont
  ceux que ``note_corniere.FICHIERS`` déclare (le paquet de la note les recevra) ;
- MUTATIONS (chacune doit faire rougir un contrôle) : une cornière oubliée, une donnée non transmise au moteur (la
  tolérance dy), le mode mal nommé, un gabarit de formule faux, l'indice de la note perdu.

    python3 tests/test_note_corniere.py
"""
import base64
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from acier.corniere import entree as E  # noqa: E402
from acier.corniere import formules as FO  # noqa: E402
from acier.corniere import methode as MT  # noqa: E402
from acier.corniere import moteur as MO  # noqa: E402
from acier.corniere import note as NT  # noqa: E402
from acier.formats import F  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import note_corniere as NC  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def corniere(**kw):
    c = MO.nouvelle_corniere()
    c.update(kw)
    return c


def etat(*cs):
    return {"version": 2, "cornieres": [copy.deepcopy(c) for c in cs], "actif": 0}


ETATS = {
    "A historique (cornière neuve)": etat(corniere()),
    "A équilibre": etat(corniere(nom="Équilibre", mode="equilibre")),
    "B, Y = 85": etat(corniere(nom="Retournée", configuration="B")),
    "B, Y = 45 (cas de contrôle retourné, sans jeu)": etat(corniere(nom="Retournée basse", configuration="B", Y=45, dy=0, dx=0, R=42.76)),
    "inégale L 150×100×12, 150 horizontal": etat(corniere(nom="Inégale 150", profil="L 150x100x12", Y=65)),
    "inégale L 150×100×12, 100 horizontal": etat(corniere(nom="Inégale 100", profil="L 150x100x12", aile_horizontale="B", Y=115, X=60, dx=15)),
    "petite L 40×40×4": etat(corniere(nom="Petite", profil="L 40x40x4", X=25, dx=5, Y=28, dy=2, d=8, R=4, L=200, n=2)),
    "grosse L 250×250×35, équilibre, quatre ancrages": etat(corniere(nom="Grosse", profil="L 250x250x35", X=150, dx=30, Y=180, dy=15, d=24, R=200, L=600,
                                                                 n=4, mode="equilibre")),
    "deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)": etat(corniere(nom="Appui A", a=100),
                                                                     corniere(nom="Appui B", configuration="B", uV=0, yc=10, dx=0, dy=0)),
    "garde : ancrage bas, cas de contrôle, R_ELU saisie": etat(corniere(nom="Garde", Y=40, dy=0, dx=0, R=42.76, R_ELU=68.9)),
    "note v1 relue (modèle lame), saisie libre": etat({"nom": "Ancienne", "standard": False, "A": 100, "B": 100, "t": 10, "V": 20, "e": 50}),
}
NOM = "Façade « Est » & l'atelier \"Nord\""
CHANTIER = "Résidence d'Été & Cie — « Lot 2 »"
INFOS = {"nom": NOM, "chantier": CHANTIER, "indice": "C", "date": "03/04/2026"}
INTERDITS = ("indice 0", "indice A", "INDICE 0", "INDICE A")
TETE = "Note de calcul · 03/04/2026 · indice C"


def norm(t):
    return O.norm(t)


def plat(t):
    return re.sub(r"\s+", "", t)


def plat_sym(t):
    """Sans blanc ni « _ » : un symbole indicé (R_ELU) sort de pdftotext en « RELU » ou « R ELU » ; la virgule française
    de la note (``note.francais``) appliquée au texte attendu."""
    return plat(NT.francais(t).replace("_", ""))


def pages_brutes(octets):
    """Le texte de chaque page dans l'ordre du dessin (pdftotext -raw : une formule reste d'un tenant)."""
    return O.pages(octets, brut=True)


def debut_pages(tx, nom):
    """L'indice de la page 1 d'une cornière (son bandeau : « Cornière d'appui — <nom> » et « η max »), puis le nombre
    de pages qu'elle occupe (2, ou 3 avec une page de suite)."""
    for i, t in enumerate(tx):
        n = norm(t)
        if f"{NT.TITRE} — {nom}" in n and "η max" in n:
            k = 2
            if i + 2 < len(tx) and "SUITE DE" in norm(tx[i + 2]) and nom in norm(tx[i + 2]):
                k = 3
            return i, k
    return None, 0


def nombre(x, nd):
    return F(x, nd)


def ecarts_parite(e, octets):
    """Les écarts entre la note et le moteur de l'écran, cornière par cornière (voir l'en-tête)."""
    tx = pages_brutes(octets)
    out = []
    attendu_pages = 1
    debuts = []
    for c in e["cornieres"]:
        res = MO.verifier(c)
        nom = res["corniere"]["nom"]
        i, k = debut_pages(tx, nom)
        if i is None:
            out.append(f"« {nom} » : aucune page de calcul à son nom")
            continue
        debuts.append(i)
        attendu_pages += k
        p1, p2 = norm(tx[i]), norm(tx[i + 1])
        p12 = p1 + " " + p2 + (" " + norm(tx[i + 2]) if k == 3 else "")
        # le bandeau : statut, η max, gouvernante, combinaison, configuration et mode
        for x in (NT.STATUTS[res["statut"]][0], f"η max {NT._pct(res['taux_max'])}", res["gouvernant"]["nom"],
                  res["gouvernant"]["combinaison"]["libelle"], f"Configuration {res['configuration']} · {E.LIBELLES_MODE[res['mode']]}"):
            if x not in p1:
                out.append(f"« {nom} » p. 1 : « {x} » absent")
        # chaque vérification : son taux ; les moments M1 à M4
        for v in res["verifications"]:
            if not O.present(NT._pct(v["taux"]), p1):
                out.append(f"« {nom} » p. 1 : taux {v['cle']} {NT._pct(v['taux'])} absent")
            if v["cle"] in ("S1", "S2", "S3", "S4") and not O.present(nombre(abs(v["M"]), 3), p1):
                out.append(f"« {nom} » p. 1 : M de {v['cle']} = {nombre(abs(v['M']), 3)} absent")
        # H, N et V par ancrage
        anc = res["ancrages"]
        for lab, val in (("H", res["efforts"]["H"]), ("N_ELS", anc["N_ELS"]), ("V_ELS", anc["V_ELS"]), ("N_ELU", anc["N_ELU"]), ("V_ELU", anc["V_ELU"])):
            if not O.present(nombre(val, 2), p1):
                out.append(f"« {nom} » p. 1 : {lab} = {nombre(val, 2)} absent")
        # les résultats des formules écrites : la chaîne de chaque section, la coupe gouvernante, les ancrages
        subs = []
        for cle in ("S1", "S2", "S3", "S4", "nette"):
            b = FO.bloc(res, cle)
            if b is not None:
                subs += FO.chaine(b)
        subs += FO.coupe(FO.bloc(res, res["gouvernant"]["cle"])) + FO.subs(FO.bloc(res, "ancrages"))
        for s in subs:
            r = FO.fmt(s.res, s.nd) if s.nd != -1 else NT._pct(s.res)
            if not O.present(r, p1):
                out.append(f"« {nom} » p. 1 : résultat « {FO.rendre(s)} » absent")
        # la page 2 : le titre, les sections de méthode, les avertissements, les références, les dessins
        if f"DÉTAILS ET MÉTHODE — {nom}" not in p2:
            out.append(f"« {nom} » p. 2 : le titre manque")
        g = res["geometrie"]
        for sec in MT.sections(res["mode"], res["configuration"], g["dx"] > 0 or g["dy"] > 0, g["a"] is not None):
            if plat(sec["titre"].upper()) not in plat(p12):
                out.append(f"« {nom} » p. 2 : section « {sec['titre']} » absente")
        for a in res["avertissements"]:
            if plat_sym(a["message"][:60]) not in plat_sym(p12):
                out.append(f"« {nom} » p. 2 : avertissement « {a['message'][:40]} » absent")
        if norm(MT.REFERENCES[0][:40]) not in p12:
            out.append(f"« {nom} » p. 2 : les références manquent")
        for x in (f"R = {NT._kn(res['efforts']['R']).replace(',00', '')}", f"L = {NT._mm(g['L'])}"):
            if x not in p12:
                out.append(f"« {nom} » : le texte du dessin « {x} » manque")
    if len(tx) != attendu_pages:
        out.append(f"{len(tx)} pages pour {attendu_pages} attendues (garde + pages de chaque cornière)")
    # le sommaire de la garde : les pages annoncées sont celles où chaque cornière commence
    annonces = [int(x) for x in re.findall(r"page (\d+)", tx[0])]
    if annonces != [i + 1 for i in debuts]:
        out.append(f"sommaire : pages annoncées {annonces}, pages réelles {[i + 1 for i in debuts]}")
    return out


def ecarts_identification(octets, infos=INFOS):
    tx = pages_brutes(octets)
    out = []
    tout = norm(" ".join(tx))
    out += [f"« {x} » imprimé" for x in INTERDITS if x in tout]
    g = plat(tx[0])
    for lab, val in (("PROJET", infos["chantier"]), ("PARTIE", infos["nom"]), ("DATE", infos["date"]), ("INDICE", infos["indice"])):
        if plat(lab + val) not in g:
            out.append(f"garde : {lab} {val} absent")
    for i, t in enumerate(tx[1:]):
        if f"{infos['date']} · indice {infos['indice']}" not in norm(t):
            out.append(f"page {i + 2} : « {infos['date']} · indice {infos['indice']} » absent de l'en-tête")
    return out


# ---------------------------------------------------------------- 1. la note de chaque état, la parité
NOTES, DUREES = {}, {}
for nom_e, e in ETATS.items():
    t0 = time.time()
    f = NC.fichier(copy.deepcopy(e), INFOS)
    DUREES[nom_e] = time.time() - t0
    octets = base64.b64decode(f["base64"])
    NOTES[nom_e] = octets
    n = len(e["cornieres"])
    tx = pages_brutes(octets)
    chk(f"{nom_e} : un PDF ({len(tx)} pages, {len(octets) // 1000} ko, {DUREES[nom_e]:.2f} s), nommé d'après la note, rien qui déborde",
        octets[:5] == b"%PDF-" and len(tx) >= 1 + 2 * n and f["nom"] == f"Note de calcul - {C.nom_fichier(NOM, '')}.pdf" and f["type"] == "application/pdf"
        and not NT.derniers_avertissements and C.refus_fichier(f) is None, (octets[:8], len(tx), f["nom"], NT.derniers_avertissements))
    ec = ecarts_parite(e, octets)
    chk(f"{nom_e} : parité avec le moteur — nom, statut, η max, gouvernante, taux, M1 à M4, H, N et V, résultats des formules, méthode, dessins",
        not ec, ec[:5])
    ec = ecarts_identification(octets)
    chk(f"{nom_e} : l'identification de la note du site dans la garde et l'en-tête de chaque page", not ec, ec[:4])
chk("au moins huit états, dont une note à deux cornières", len(ETATS) >= 8 and any(len(e["cornieres"]) == 2 for e in ETATS.values()))
print(f"--   durée de génération : {min(DUREES.values()):.2f} à {max(DUREES.values()):.2f} s par note (CPython, {len(ETATS)} notes)")
tailles = {nom_e: (NC.pdf(copy.deepcopy(e), INFOS), list(NT.dernieres_variantes))[1] for nom_e, e in ETATS.items()}
print(f"--   tailles retenues : {sorted(set(v for vs in tailles.values() for v in vs))} — {sum(1 for vs in tailles.values() if 'serrée' in vs)} note(s) serrée(s)")

# le contenu de la note v1 relue : l'avis de relecture est imprimé (attention), le nom et la section gardés
tv1 = norm(" ".join(pages_brutes(NOTES["note v1 relue (modèle lame), saisie libre"])))
chk("une note v1 relue : l'avis « note d'avant le 10/11/2026, modèle lame » est imprimé dans les limites, le nom et la saisie libre 100 × 100 × 10 gardés",
    "modèle lame" in tv1 and "Ancienne" in tv1 and "100 × 100 × 10" in tv1 and "saisie libre" in tv1, tv1[:300])
# la garde (ancrage bas) : l'avertissement rouge avec ses deux pourcentages
res_g = MO.verifier(ETATS["garde : ancrage bas, cas de contrôle, R_ELU saisie"]["cornieres"][0])
tg = norm(" ".join(pages_brutes(NOTES["garde : ancrage bas, cas de contrôle, R_ELU saisie"])))
chk("la garde : l'avertissement rouge « équilibre P1 plus sévère » avec ses deux pourcentages est imprimé ; R_ELU saisie (68,9 kN), pas « auto »",
    res_g["garde"]["depasse"] and f"{100 * res_g['garde']['taux_equilibre_P1']:.1f} %".replace(".", ",") in tg
    and f"{100 * res_g['garde']['taux_historique']:.1f} %".replace(".", ",") in tg and "plus sévère" in tg and "68,9" in tg and "(auto" not in tg,
    [x for x in tg.split(". ") if "sévère" in x or "68,9" in x][:3])
# la configuration B : la section nette, la pression sur le support, le mode imposé
tb = norm(" ".join(pages_brutes(NOTES["B, Y = 85"])))
res_b = MO.verifier(ETATS["B, Y = 85"]["cornieres"][0])
chk("la configuration B : la section nette (b_net, la face du trou), la pression sur le support, l'avertissement du mode imposé",
    "nette au trou (dessus)" in tb and "SECTIONNETTE" in plat(tb) and F(res_b["ancrages"]["pression_support"], 1) in tb
    and "Pression du bout libre sur le support" in tb and "formules du classeur ne s'appliquent pas" in tb and "(info)" not in tb,
    [x for x in tb.split(" · ") if "nette" in x][:3])

# ---------------------------------------------------------------- 2. chaque formule ré-évaluée
n_sub, ecarts = 0, []
for nom_e, e in ETATS.items():
    for c in e["cornieres"]:
        res = MO.verifier(c)
        for b, subs in FO.toutes(res):
            for s in subs:
                n_sub += 1
                ok1, v1 = FO.verifier(s)
                ok2, v2 = FO.exact(s)
                if not (ok1 and ok2):
                    ecarts.append((nom_e, b["cle"], FO.rendre(s), v1, v2))
chk(f"{n_sub} formules (tous les blocs de {sum(len(e['cornieres']) for e in ETATS.values())} cornières) ré-évaluées : les nombres imprimés redonnent le "
    "résultat du moteur (arrondi d'affichage) et les termes exacts le redonnent à 1e-9", not ecarts and n_sub >= 800, ecarts[:4])
cles_moteur = {f["cle"] for e in ETATS.values() for c in e["cornieres"] for b in MO.verifier(c)["calcul"] for f in b["formules"]}
chk("chaque clé de formule que le moteur nomme a son gabarit, et chaque gabarit sert au moins une fois",
    cles_moteur <= set(FO.GABARITS) and set(FO.GABARITS) <= cles_moteur, (sorted(cles_moteur - set(FO.GABARITS)), sorted(set(FO.GABARITS) - cles_moteur)))
res0 = MO.verifier(ETATS["A historique (cornière neuve)"]["cornieres"][0])
chk("en mode historique, les cellules du classeur sont imprimées avec les étiquettes (C19 C35, C55 C25 C23 C56 C57, C82 C83…)",
    all(x in norm(pages_brutes(NOTES["A historique (cornière neuve)"])[1]) for x in ("C19 C35", "C55 C25 C23 C56 C57", "C78 C82 C83")),
    norm(pages_brutes(NOTES["A historique (cornière neuve)"])[1])[:300])
chk("la section gouvernante est écrite « M3 », pas « M(s) » ; le taux « η »", any(s.nom == "M3" for s in FO.chaine(FO.bloc(res0, "S3")))
    and any(s.nom == "η" for s in FO.coupe(FO.bloc(res0, "S3"))) and not any(s.nom in FO.TAUX_DETAIL for s in FO.coupe(FO.bloc(res0, "S3"))))

# ---------------------------------------------------------------- 3. l'identification, le nom du fichier, la correspondance
e2 = ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"]
chk("infos_doc : la correspondance de la poutre (projet = chantier, partie = nom, indice, date)",
    NC.infos_doc(e2, INFOS) == {"nom_projet": CHANTIER, "partie": NOM, "indice": "C", "date": "03/04/2026"})
chk("sans chantier : projet = le nom de la note, partie = « Cornières d'appui » ; sans infos : le nom de la cornière seule, indice A",
    NC.infos_doc(e2, {"nom": "Terrasse"}) == {"nom_projet": "Terrasse", "partie": NC.TITRE, "indice": "A"}
    and NC.infos_doc(ETATS["A équilibre"], {}) == {"nom_projet": "Équilibre", "partie": NC.TITRE, "indice": "A"}
    and NC.infos_doc(e2, {}) == {"nom_projet": NC.TITRE, "partie": NC.TITRE, "indice": "A"})
f_sans = NC.fichier(copy.deepcopy(ETATS["A équilibre"]), {})
t_sans = norm(" ".join(pages_brutes(base64.b64decode(f_sans["base64"]))))
chk("sans infos : le fichier porte le nom de la cornière, la garde dit « indice A » et la date du jour",
    f_sans["nom"] == "Note de calcul - Équilibre.pdf" and "indice A" in t_sans and time.strftime("%d/%m/%Y") in t_sans, (f_sans["nom"], t_sans[:200]))
f_dang = NC.fichier(copy.deepcopy(ETATS["A équilibre"]), {"nom": "Note <b>/x"})
chk("un nom de note dangereux donne un nom de fichier sûr (celui de commun.nom_fichier)",
    f_dang["nom"] == C.nom_fichier("Note de calcul - Note <b>/x", "pdf") and "<" not in f_dang["nom"] and "/" not in f_dang["nom"]
    and C.refus_fichier(f_dang) is None, f_dang["nom"])

# ---------------------------------------------------------------- 4. la page de suite, les refus
vrai_limites = NT.Methode.limites


def limites_bavardes(self, d, fr):
    vrai_limites(self, d, fr)
    for i in range(14):
        self.paragraphe(d, fr, f"Ligne de remplissage {i + 1} pour faire déborder la colonne des limites.")


NT.Methode.limites = limites_bavardes
try:
    octets = NC.pdf(copy.deepcopy(ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"]), INFOS)
    reports = NT.derniers_reports
finally:
    NT.Methode.limites = vrai_limites
tx = pages_brutes(octets)
chk("un bloc qui ne tient pas (limites allongées) passe en PAGE DE SUITE : 1 + 3 + 3 pages, « Suite p. 3 : … » en page 2, « SUITE DE LA PAGE 2 » en "
    "page 3 avec les références, le sommaire compte les pages (2 et 5), rien ne déborde, la note n'est pas refusée",
    len(tx) == 7 and "Suite p. 3" in norm(tx[2]) and "SUITE DE LA PAGE 2" in norm(tx[3]) and "RÉFÉRENCES" in plat(tx[3])
    and [int(x) for x in re.findall(r"page (\d+)", tx[0])] == [2, 5] and not NT.derniers_avertissements and reports
    and ecarts_parite(ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"], octets) == [],
    (len(tx), reports, NT.derniers_avertissements, [t[:80] for t in tx[2:4]]))
try:
    NC.fichier(etat(corniere(nom="Trop loin", X=130)), INFOS)
    r = "aucune erreur"
except ValueError as err:
    r = str(err)
chk("une cornière hors domaine est refusée avec son nom et le message du moteur, jamais une note fausse",
    "« Trop loin »" in r and "au-delà de l'aile" in r, r)
try:
    NC.fichier({"version": 2, "cornieres": [], "actif": 0}, INFOS)
    r = "aucune erreur"
except ValueError as err:
    r = str(err)
chk("une note sans cornière est refusée", "aucune cornière" in r, r)
chk("generer_note accepte aussi une simple liste de cornières", NT.generer_note([corniere()], {"indice": "B"})[:5] == b"%PDF-")

# ---------------------------------------------------------------- 5. les mutations
e_mut = ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"]
vrai_poser = NT.poser
compte = []


def poser_oublieux(d, calc, doc_meta, k, reports=()):
    compte.append(1)
    return [] if len(compte) == 2 else vrai_poser(d, calc, doc_meta, k, reports)


NT.poser = poser_oublieux
try:
    mut = NC.pdf(copy.deepcopy(e_mut), INFOS)
finally:
    NT.poser = vrai_poser
chk("mutation « une cornière oubliée » : la parité rougit", ecarts_parite(e_mut, mut) != [])

vrai_init = NT.Calcul.__init__


def init_sans_dy(self, c):
    c = dict(c)
    c["dy"] = 0.0
    vrai_init(self, c)


NT.Calcul.__init__ = init_sans_dy
try:
    mut = NC.pdf(copy.deepcopy(ETATS["A historique (cornière neuve)"]), INFOS)
finally:
    NT.Calcul.__init__ = vrai_init
chk("mutation « une donnée non transmise au moteur » (la tolérance dy oubliée) : la parité rougit", ecarts_parite(ETATS["A historique (cornière neuve)"], mut) != [])

vrai_E = NT.E
NT.E = types.SimpleNamespace(**{k: v for k, v in vars(E).items() if not k.startswith("__")})
NT.E.LIBELLES_MODE = {E.HISTORIQUE: "méthode du classeur", E.EQUILIBRE: "équilibre"}
try:
    mut = NC.pdf(copy.deepcopy(ETATS["A historique (cornière neuve)"]), INFOS)
finally:
    NT.E = vrai_E
chk("mutation « le mode mal nommé » : la parité rougit", ecarts_parite(ETATS["A historique (cornière neuve)"], mut) != [])

vrai_gab = FO.GABARITS["Xe"]
FO.GABARITS["Xe"] = (("X", "r", "e"), "X - r", 1)
try:
    s_mut = next(s for s in FO.chaine(FO.bloc(res0, "S1")) if s.nom == "Xe")
    rouge = not FO.exact(s_mut)[0] and not FO.verifier(s_mut)[0]
finally:
    FO.GABARITS["Xe"] = vrai_gab
chk("mutation « un gabarit faux » (Xe = X − r) : la ré-évaluation rougit", rouge)

vrai_infos = NC.infos_doc
NC.infos_doc = lambda etat_, infos: {k: v for k, v in vrai_infos(etat_, infos).items() if k != "indice"}
try:
    mut = NC.pdf(copy.deepcopy(ETATS["A historique (cornière neuve)"]), INFOS)
finally:
    NC.infos_doc = vrai_infos
chk("mutation « l'indice de la note perdu » : l'identification rougit (« indice 0 » imprimé)", ecarts_identification(mut) != [])

# ---------------------------------------------------------------- 6. comme dans le navigateur
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
try:
    import construire as K  # noqa: E402
finally:
    sys.path.pop(0)
SONDE = r"""
import builtins, json, os, sys
RACINE, STUB, REFUS = {racine!r}, {stub!r}, {refus!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REFUS or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
def depot():
    mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
    return {{os.path.relpath(f, RACINE) for f in mods | lus if f.startswith(RACINE + os.sep) and not f.endswith(".pyc")}}
from acier.corniere import moteur as MO
from api_web import dessin_corniere as DC
with _open({etats!r}, encoding="utf-8") as f:
    etats = json.load(f)
for e in etats:
    for c in e["cornieres"]:
        DC.vues(MO.verifier(c))
ecran = depot()
numpy_ecran = "numpy" in sys.modules
from api_web import note_corniere as NC
sorties = [NC.fichier(e, {infos!r})["base64"] for e in etats]
note = depot()
import PIL
print(json.dumps({{"note": sorted(f for f in note - ecran if not f.endswith(".ttf")), "numpy_ecran": numpy_ecran,
                   "numpy": "numpy" in sys.modules, "polices": sorted(os.path.basename(f) for f in lus if f.endswith(".ttf")),
                   "pil": PIL.__file__, "fonttools": "fontTools" in sys.modules,
                   "rl_accel": any(m.split(".")[-1] == "_rl_accel" for m in sys.modules), "sorties": sorties}}))
"""


def sonde(etats, infos):
    dossier = tempfile.mkdtemp()
    try:
        for f in K.NOTE_POLICES:
            shutil.copy(os.path.join(RACINE, "ndc_pdf", "fonts", f), dossier)
        ch = os.path.join(dossier, "etats.json")
        with open(ch, "w", encoding="utf-8") as fh:
            json.dump(etats, fh, ensure_ascii=False)
        code = SONDE.format(racine=RACINE, stub=O.STUB, refus=K.NOTE_REFUS, etats=ch, infos=infos)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE, env={**os.environ, "NDC_FONT_DIR": dossier})
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"erreur": p.stderr[-1500:]}
    out["sorties"] = [base64.b64decode(s) for s in out["sorties"]]
    return out


e_nav = [ETATS["A historique (cornière neuve)"], ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"], ETATS["petite L 40×40×4"],
         ETATS["grosse L 250×250×35, équilibre, quatre ancrages"]]
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
s = sonde(e_nav, infos_nav)
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet, la note se fait",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB), s.get("erreur") or {k: v for k, v in s.items() if k != "sorties"})
if "erreur" not in s:
    cpy = [O.pages(NC.pdf(copy.deepcopy(x), infos_nav)) for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [O.pages(o) for o in s["sorties"]] == cpy)
    pol = sorted({p for o in s["sorties"] for p in O.polices(o)})
    noms_paquet = {f[:-4] for f in K.NOTE_POLICES}
    chk(f"navigateur : les polices de la note sont celles du paquet (et Helvetica, police standard non embarquée) : {pol}",
        set(pol) <= noms_paquet | {"Helvetica"} and set(s["polices"]) <= set(K.NOTE_POLICES), (pol, s["polices"]))
    chk("navigateur : numpy est déjà chargé par l'écran du module (le catalogue des cornières) — la note ne l'ajoute pas", s["numpy_ecran"] is True)
    with open(os.path.join(RACINE, "site", "web", "moteur", "manifeste.json"), encoding="utf-8") as fh:
        moteur = set(json.load(fh)["fichiers"])
    # déjà dans le paquet de la note de la poutre (ndc_pdf, le Pillow factice) ou dans un paquet du site empaqueté entier
    # (api_web : construire.PAQUETS_ENTIERS — le manifeste du moteur est celui de la dernière construction)
    deja = {f for f in s["note"] if f.startswith(("ndc_pdf/", "site/note_pdf/")) or f.split("/")[0] in K.PAQUETS_ENTIERS}
    hors = sorted(set(s["note"]) - moteur - deja)
    print(f"--   fichiers du dépôt chargés par la note, hors écran, paquet du moteur et paquet de la poutre : {hors}")
    chk("navigateur : ces fichiers sont ceux que note_corniere.FICHIERS déclare, et tous les déclarés sont chargés (le paquet de la note les reçoit)",
        set(hors) <= set(NC.FICHIERS) and all(f in s["note"] for f in NC.FICHIERS), (hors, [f for f in NC.FICHIERS if f not in s["note"]]))
if "corniere" in K.NOTE_PDF:
    octets_zip, _ = K.paquet_note()
    noms = set(zipfile.ZipFile(io.BytesIO(octets_zip)).namelist())
    chk("paquet de la note (note.zip) : il porte les fichiers que la note des cornières déclare", all(f in noms for f in NC.FICHIERS),
        [f for f in NC.FICHIERS if f not in noms])
else:
    print("--   paquet de la note : « corniere » n'est pas encore dans construire.NOTE_PDF — contrôle en attente de l'intégration")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
