# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Charges sur plancher » sur le site (api_web/plancher.telecharger → api_web/note_cas.py),
vérifiée en Python ordinaire.

- TOUTE la note en un PDF : deux pages par cas, dans l'ordre de la liste ; le texte de chaque page (pdftotext) est
  celui de la note de ce cas faite SEULE par le générateur validé (plancher/charges/note.py), et la taille retenue est
  la sienne — ce qui lie les deux copies nommées de note_cas ;
- PARITÉ avec l'écran du site, cas par cas : le calcul que la note refait (``Calcul`` : ``etat.construire`` puis
  ``verification.verifier``) rend le statut, le taux, la vérification gouvernante et chaque ligne de vérification
  (effet E, résistance R, taux, statut) du moteur de l'écran (``plancher._calculer``) ; le verdict, le taux de chaque
  famille, M_Ed / V_Ed / M_Rd / V_Rd de chaque élément, les charges saisies (engins et leurs appuis, étançons, files
  d'étais, stocks) et les appuis de l'écran s'écrivent dans les pages du cas. Seul écart de VERDICT, mesuré et
  rapporté : entre 85 et 95 %, l'écran dit « Conforme » (le code couleur du bureau), la note « CONFORME · ATTENTION »
  (le seuil du moteur, ``parametres.seuil_attention`` = 0,85, celui de l'écran Streamlit) ;
- un cas que le moteur ne sait pas construire arrête la note avec le message de l'écran (« Donnée manquante : … ») ;
  un cas « données manquantes » (refus du moteur, P-7) se fait comme dans Streamlit ;
- l'identification : TOUJOURS celle de la note du site, avec la correspondance de la note de la poutre ; celle que
  porte chaque cas (un fichier Streamlit) ne s'imprime pas et l'écran ne la montre plus (09/11/2026) ;
- le routeur la rend en base 64, nommée d'après la note ; un autre fichier est refusé ; une note qui déborde est
  refusée, jamais tronquée ;
- MUTATIONS (chacune doit faire rougir un contrôle) : un cas oublié, des infos mal transmises, la page 2 sans
  l'échelle du plan de la page 1, la taille imposée ;
- comme dans le navigateur : la note se fait sans fontTools ni accélérateurs C, avec le Pillow factice et les polices
  du paquet ; son texte est celui de CPython ; les fichiers du dépôt qu'elle charge en plus de l'écran sont ceux que
  ``plancher.NOTE.fichiers`` déclare.

    python3 tests/test_note_plancher.py
"""
import base64
import copy
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import note_cas as NC  # noqa: E402
from api_web import plancher as PL  # noqa: E402
from api_web import routeur as R  # noqa: E402
from plancher.charges import note as NT  # noqa: E402
from plancher.charges.parametres import DEFAUT as PAS  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def nombre(t: str) -> float:
    return float(t.replace("−", "-").replace(" ", "").replace(" ", "").replace(" ", "").replace(",", "."))


def fr(x, nd) -> str:
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def note_de(*cas_):
    """Une note faite des cas donnés (chacun pris d'une note d'exemple, construite par les saisies de l'écran)."""
    e = copy.deepcopy(cas_[0])
    for x in cas_[1:]:
        e["cas"] += copy.deepcopy(x["cas"])
        for k in ("engins", "repartiteurs", "hourdis"):
            e["fiches"][k] += [f for f in x["fiches"][k] if f not in e["fiches"][k]]
    return PL.normaliser(e)


H, D, P = PL.exemple("hourdis"), PL.exemple("dalle"), PL.exemple("predalle")
NOK = copy.deepcopy(H)
PL._faire(NOK, (["nom"], "C4 sans répartiteur"), (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"))
ETATS = {
    "hourdis (engin sur plateaux, étançon sur filière, stock)": PL.normaliser(H),
    "trois cas : hourdis, dalle coulée, hourdis NON CONFORME": note_de(H, D, NOK),
    "prédalle sans chape durcie (données manquantes, P-7), puis dalle": note_de(P, D),
    "neuve (le SP 200-4X du catalogue Ergon par défaut)": PL.normaliser(PL.nouvel_etat()),
    "fiche générique à renseigner (données manquantes)": PL.normaliser(PL._faire(
        PL.nouvel_etat(), (["plancher", "hourdis", "type_cat"], "Hourdis générique 200 · à renseigner"))),
    "hourdis, V_Rd = 95 kN (tranchant à 90,5 %)": PL.normaliser(PL._faire(copy.deepcopy(H), (["plancher", "hourdis", "V_Rd"], "95"))),
}
# l'identification saisie à l'écran, propre à chaque cas
ID = copy.deepcopy(ETATS["trois cas : hourdis, dalle coulée, hourdis NON CONFORME"])
for i, (projet, partie, indice) in enumerate((("Bâtiment A", "Plancher R+1", "D"), ("", "", ""), ("Bâtiment C", "", "E"))):
    for k, v in (("projet", projet), ("partie", partie), ("indice", indice)):
        ID["actif"] = i
        assert PL.modifier(ID, ["identification", k], v) is None
ETATS["trois cas, une identification saisie par cas"] = PL.normaliser(ID)
INFOS = {"nom": "Levage", "chantier": "Quai 4", "indice": "B", "date": "01/02/2026"}
STATUT_NOTE_ECRAN = {"ok": "p", "attention": "f", "nok": "r"}

# 1. la note de toute la note : un PDF, deux pages par cas, chaque page = celle du cas seul
NOTES = {}
for nom, e in ETATS.items():
    f, octets, ks = O.note_et_tailles(PL, e, INFOS)
    NOTES[nom] = octets
    n = len(e["cas"])
    chk(f"{nom} : un PDF de {2 * n} pages ({n} cas), nommé d'après la note",
        octets[:5] == b"%PDF-" and len(O.pages(octets)) == 2 * n and f["nom"] == "Note de calcul - Levage.pdf", (octets[:8], len(O.pages(octets)), f["nom"]))
    ecarts = O.controle_identite(PL, e, INFOS, octets, ks)
    chk(f"{nom} : chaque page est celle du cas fait seul par le générateur, dans sa taille", not ecarts, ecarts[:3])
statuts_essai = sorted({(PL._calculer(e, c).get("res") or {}).get("statut") for e in ETATS.values() for c in e["cas"]})
chk(f"les états d'essai couvrent les statuts du moteur : {statuts_essai}", {"attention", "nok", "refus"} <= set(statuts_essai), statuts_essai)

# 2. la parité avec l'écran du site
ecarts, verdicts, n_cas, n_v, n_ch = [], [], 0, 0, 0
for nom, e in ETATS.items():
    tx, brut = O.pages(NOTES[nom]), O.pages(NOTES[nom], brut=True)
    for i, c in enumerate(e["cas"]):
        n_cas += 1
        ei = copy.deepcopy(e)
        ei["actif"] = i
        r = PL.resultat(ei)
        site = PL._calculer(ei, ei["cas"][i])["res"]
        note = NT.Calcul(ei, ei["cas"][i]).res
        cle = lambda x: (x["statut"], x.get("taux_max"), x.get("gouverne"),  # noqa: E731
                         [(v["famille"], v["objet"], v.get("E"), v.get("R"), v["taux"], v["statut"]) for v in x.get("verifications", [])])
        if cle(site) != cle(note):
            ecarts.append((nom, i, "calcul", cle(site)[:3], cle(note)[:3]))
        p1, b1 = O.norm(tx[2 * i]), O.norm(brut[2 * i])
        # le verdict : le statut de la note, la pastille et le statut de l'écran
        if NT.STATUTS[site["statut"]][0] not in p1:
            ecarts.append((nom, i, "statut", NT.STATUTS[site["statut"]][0]))
        statut_ecran = re.search(r'class="status (\w+)"', r["statut_html"]).group(1)
        v = O.verdict(r["resultat"])
        if site["statut"] == "refus":
            # l'écran : « Vérification impossible », pastille « — » ; la note : « DONNÉES MANQUANTES »
            if v != ("c", "—") or "Vérification impossible" not in r["resultat"] or "DONNÉES MANQUANTES" not in p1 or statut_ecran != "c":
                ecarts.append((nom, i, "refus", v, statut_ecran))
            continue
        t = site["taux_max"]
        if v is None or v[1] != O.pct_note(t) or f"η max {O.pct_note(t)}" not in p1:
            ecarts.append((nom, i, "η", v, O.pct_note(t)))
        if statut_ecran != STATUT_NOTE_ECRAN[site["statut"]]:
            verdicts.append((nom, c["nom"], t, statut_ecran, NT.STATUTS[site["statut"]][0]))
        g = site.get("gouverne")
        if g and f"gouverne : {PL.LIB_FAM[g['famille']].lower()} · {g['objet']}" not in p1:
            ecarts.append((nom, i, "gouverne", g))
        # le taux de chaque famille (colonne Résultat) = le bandeau de la note
        for nom_l, piece, niv, txt in O.lignes_resultat(r["resultat"]):
            if nom_l in dict(NT.FAMILLES).values() and re.fullmatch(r"\d+ %", txt):
                n_v += 1
                if f"{nom_l} {txt}" not in p1:
                    ecarts.append((nom, i, "famille", nom_l, txt))
            m = re.fullmatch(r"(\w+) · (\d+(?:,\d+)?) kN", nom_l)
            if m:   # un appui de l'écran (« P1 · 45 kN », établi compact : l'élément qui le porte est sur la ligne du dessous)
                n_ch += 1
                if not re.search(rf"\b{m.group(1)} .*? {m.group(2)} ", b1):
                    ecarts.append((nom, i, "appui", nom_l))
        # M_Ed, M_Rd, V_Ed, V_Rd et les taux de chaque élément (tableau ÉLÉMENTS) = les lignes du moteur
        par = {}
        for vv in site["verifications"]:
            if vv["famille"] in ("flexion", "tranchant") and vv["taux"] is not None:
                par.setdefault(vv["objet"], {})[vv["famille"]] = vv
        for objet, ff in par.items():
            if len(ff) < 2:
                continue
            n_v += 2
            fl, tr = ff["flexion"], ff["tranchant"]
            ligne = (f"{objet} {fr(fl['E'], 1)} {fr(fl['R'], 1)} {O.pct_note(fl['taux'])} "
                     f"{fr(tr['E'], 1)} {fr(tr['R'], 1)} {O.pct_note(tr['taux'])}")
            if ligne not in b1:
                ecarts.append((nom, i, "élément", ligne))
        # la vérification gouvernante de l'écran (« V_Ed = 86,02 kN ≤ V_Rd = 90,00 kN ») = sa ligne dans la note
        cmp_ = re.search(r'<div class="gov"><b>[^<]*</b>(.*?)</div>', r["resultat"])
        nums = re.findall(r"-?\d[\d   ]*(?:,\d+)?", O.ecran(cmp_.group(1))) if cmp_ else []
        if g and g["famille"] in ("flexion", "tranchant") and len(nums) >= 2:
            vv = par[g["objet"]][g["famille"]]
            if abs(nombre(nums[-2]) - vv["E"]) > 0.006 or abs(nombre(nums[-1]) - vv["R"]) > 0.006:
                ecarts.append((nom, i, "gouvernante", nums, vv["E"], vv["R"]))
        # les charges saisies (cadre Données) = la ligne « Charges : » de la note
        cas = ei["cas"][i]
        ch_txt = re.search(r"Charges : (.*?)\| Répartiteurs", b1)
        ch_txt = ch_txt.group(1) if ch_txt else ""
        for ch in cas["charges"]:
            n_ch += 1
            if ch["type"] == "engin":
                attendu = [f"{k} {fr(v_, 0)}" for k, v_ in (ch.get("P") or {}).items()]
            elif ch["type"] == "ponctuel":
                attendu = [f"{ch['ref']} étançon {fr(ch['P'], 0)} kN"]
            elif ch["type"] == "lineaire":
                attendu = [f"{ch['ref']} file d'étais {fr(ch['p'], 1)} kN/m"]
            else:
                attendu = [f"{ch['ref']} stock {fr(ch['q'], 1)} kN/m²"]
            if not all(a in ch_txt for a in attendu):
                ecarts.append((nom, i, "charge", ch["ref"], attendu, ch_txt[:120]))
chk(f"parité : {n_cas} cas, {n_v} taux et lignes d'éléments, {n_ch} charges et appuis — la note écrit le statut, le taux, la vérification "
    "gouvernante, les taux des familles, M_Ed / V_Ed / résistances de chaque élément et les charges de l'écran",
    not ecarts and n_cas >= 10 and n_v >= 60 and n_ch >= 20, ecarts[:4])
chk("le seul écart de verdict mesuré : entre 85 et 95 %, l'écran dit « Conforme » (code du bureau), la note « CONFORME · ATTENTION » "
    "(seuil du moteur, 0,85)", len(verdicts) >= 1 and all(PAS.seuil_attention < x[2] < 0.95 and x[3] == "p" and x[4] == "CONFORME · ATTENTION" for x in verdicts),
    verdicts)
for x in verdicts:
    print(f"--   écart de verdict rapporté : {x[1]} à {x[2] * 100:.2f} % — écran « Conforme », note « {x[4]} »")

# 3. un cas que le moteur ne sait pas construire : la note s'arrête avec le message de l'écran
vrai_c = PL.ET.construire


def construire(etat, cas):
    if cas["nom"].startswith("C2"):
        raise KeyError("fiche de plateau « Tôle 9 » absente")
    return vrai_c(etat, cas)


PL.ET.construire = construire
try:
    e = ETATS["trois cas : hourdis, dalle coulée, hourdis NON CONFORME"]
    r = json.loads(R.traiter(json.dumps({"module": "plancher", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
    e2 = copy.deepcopy(e)
    e2["actif"] = 1
    ecran = O.ecran(PL.resultat(e2)["resultat"])
finally:
    PL.ET.construire = vrai_c
chk("un cas que le moteur ne construit pas arrête la note, avec le message que l'écran montre (« Donnée manquante : … »)",
    not r["ok"] and "Cas « C2 dalle »" in r["erreur"] and "Donnée manquante" in r["erreur"] and "Tôle 9" in r["erreur"]
    and "Tôle 9" in ecran, (r.get("erreur"), ecran[:200]))

# 4. l'identification : TOUJOURS celle de la note du site (09/11/2026 ; avant, celle de chaque cas, « 0 » par défaut,
# passait devant — le PDF disait « indice 0 » sous une note B)
attendu = lambda i, c: ["Note de calcul · 01/02/2026 · indice B", f"Quai 4 · Levage · cas {c['nom']}"]  # noqa: E731
e = ETATS["trois cas, une identification saisie par cas"]
t_id = O.norm(" ".join(O.pages(NOTES["trois cas, une identification saisie par cas"])))
ecarts = O.controle_infos(NOTES["trois cas, une identification saisie par cas"], e, attendu)
chk("l'identification que porte chaque cas (un fichier Streamlit : « Bâtiment A », « Plancher R+1 », indices D et E) ne "
    "s'imprime pas : celle de la note du site, dans chaque cas", not ecarts and not any(x in t_id for x in (
        "Bâtiment A", "Bâtiment C", "Plancher R+1", "indice D", "indice E", "indice 0")), ecarts)
e = ETATS["trois cas : hourdis, dalle coulée, hourdis NON CONFORME"]
ecarts = O.controle_infos(NOTES["trois cas : hourdis, dalle coulée, hourdis NON CONFORME"], e, attendu)
chk("le chantier, le nom, l'indice et la date de la note du site (l'identification de l'état, « 0 » par défaut, n'y est pas)",
    not ecarts, ecarts)
donnees = PL.resultat(copy.deepcopy(e))["donnees"]
chk("l'écran ne montre plus de champ d'identification (un seul composant, les trois formats) : il dit d'où vient celle "
    "de la note PDF", 'data-chemin="identification/' not in donnees and f'title="{C.esc(NC.AIDE_IDENTIFICATION)}"' in donnees,      # (au survol de ⓘ)
    donnees.count("identification/"))

# 5. le routeur, les refus
r = json.loads(R.traiter(json.dumps({"module": "plancher", "action": "telecharger", "etat": e, "quoi": "note_pdf",
                                     "infos": {"nom": "Quai : levage/1"}})))
chk("le routeur rend la note en base 64 sous un nom de fichier sûr",
    r.get("ok") and r["fichier"]["nom"] == "Note de calcul - Quai - levage-1.pdf" and base64.b64decode(r["fichier"]["base64"])[:5] == b"%PDF-",
    r.get("erreur") or r["fichier"]["nom"])
r = json.loads(R.traiter(json.dumps({"module": "plancher", "action": "telecharger", "etat": e, "quoi": "autre"})))
chk("un autre fichier que la note est refusé avec un message", not r["ok"] and "inconnu" in r["erreur"], r)
vrai = NT._essai
NT._essai = lambda calc, doc_meta, k, chemin: vrai(calc, doc_meta, k, chemin) + ["page 1, colonne 318 : 3 pt"]
try:
    r1 = json.loads(R.traiter(json.dumps({"module": "plancher", "action": "telecharger", "etat": ETATS["hourdis (engin sur plateaux, étançon sur filière, stock)"],
                                          "quoi": "note_pdf"})))
    r3 = json.loads(R.traiter(json.dumps({"module": "plancher", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
finally:
    NT._essai = vrai
chk("une note qui déborde encore en taille serrée est refusée, jamais tronquée (un cas, puis trois)",
    not r1["ok"] and not r3["ok"] and all("ne tient pas" in x["erreur"] and "3 pt" in x["erreur"] for x in (r1, r3))
    and "« C1 levage en rive »" in r3["erreur"], (r1.get("erreur"), r3.get("erreur")))

vrai_pages = NC.pages
NC.pages = lambda *a: vrai_pages(*a) + ["page 2, colonne 300 : 2 pt"]
try:
    r3b = json.loads(R.traiter(json.dumps({"module": "plancher", "action": "telecharger", "etat": e, "quoi": "note_pdf"})))
finally:
    NC.pages = vrai_pages
chk("… et le débordement relevé en posant les pages d'un cas dans la note entière (second garde-fou) l'est aussi",
    not r3b["ok"] and "ne tient pas" in r3b["erreur"] and "2 pt" in r3b["erreur"], r3b.get("erreur"))

# 6. les mutations
vrai_pages = NC.pages
compte = []


def sans_dernier(NT_, gen, d, calc, doc_meta, k, reports=()):
    compte.append(1)
    return [] if len(compte) == 3 else vrai_pages(NT_, gen, d, calc, doc_meta, k, reports)


NC.pages = sans_dernier
try:
    mut = NC.pdf(PL.NOTE, copy.deepcopy(e), INFOS)
finally:
    NC.pages = vrai_pages
chk("mutation « un cas oublié » : le contrôle d'identité rougit", O.controle_identite(PL, e, INFOS, mut) != [])
vrai_infos = NC.infos_du_cas
NC.infos_du_cas = lambda gen, etat, cas, infos: {"nom_projet": (infos or {}).get("nom"), "indice": "0", "date": (infos or {}).get("date")}
try:
    mut = NC.pdf(PL.NOTE, copy.deepcopy(e), INFOS)
finally:
    NC.infos_du_cas = vrai_infos
chk("mutation « infos mal transmises » (le nom à la place du chantier) : le contrôle de l'identification rougit",
    O.controle_infos(mut, e, attendu) != [])
vrai_note = PL.NOTE
PL.NOTE = PL.NOTE._replace(suite=lambda n1: (None,))
try:
    mut = NC.pdf(PL.NOTE, copy.deepcopy(e), INFOS)
finally:
    PL.NOTE = vrai_note
chk("mutation « page 2 sans l'échelle du plan de la page 1 » : le contrôle d'identité rougit", O.controle_identite(PL, e, INFOS, mut) != [])
vrai_var = NC.variante
for k, nom_k in NT.VARIANTES:
    NC.variante = lambda NT_, calc, doc_meta, brouillon, k=k: (k, [])
    try:
        _, mut, ks = O.note_et_tailles(PL, e, INFOS)
        rouge = O.controle_identite(PL, e, INFOS, mut, ks) != []
    except AssertionError as err:
        rouge, ks = "ne tient pas" in str(err), str(err)[:80]
    finally:
        NC.variante = vrai_var
    if k == NT.VARIANTES[0][0]:
        chk(f"taille imposée « {nom_k} » : les cas d'essai la prennent déjà (rien ne rougit, attendu)", not rouge, ks)
    else:
        chk(f"mutation « toujours la taille {nom_k} » : le contrôle des tailles rougit", rouge, ks)

# 7. comme dans le navigateur
e_nav = list(ETATS.values())
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
s = O.sonde("plancher", e_nav, infos_nav)
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les polices du paquet, la note se fait",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB), s.get("erreur") or s)
if "erreur" not in s:
    cpy = [O.pages(base64.b64decode(PL.telecharger(copy.deepcopy(x), "note_pdf", infos_nav)["base64"])) for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [O.pages(o) for o in s["sorties"]] == cpy)
    pol = sorted({p for o in s["sorties"] for p in O.polices(o)})
    noms_paquet = {f[:-4] for f in s["polices_paquet"]}
    chk(f"navigateur : les polices de la note sont celles du paquet (et Helvetica, police standard non embarquée) : {pol}",
        set(pol) <= noms_paquet | {"Helvetica"} and set(s["polices"]) <= set(s["polices_paquet"]), (pol, s["polices"]))
    chk("navigateur : numpy est déjà chargé par l'écran du module (avec_numpy) — la note ne l'ajoute pas", s["numpy_ecran"] is True)
    rel = O.releve_paquet(PL, s)
    print(f"--   fichiers du dépôt chargés par la note, hors paquet du moteur et paquet de la poutre : {rel['hors']}")
    chk("navigateur : ces fichiers sont ceux que plancher.NOTE.fichiers déclare (le paquet de la note les reçoit)",
        set(rel["hors"]) <= set(rel["declares"]) and not rel["non_charges"], rel)
manque = O.paquet_integre(PL)
if manque is None:
    print("--   paquet de la note : « plancher » n'est pas encore dans construire.NOTE_PDF — contrôle en attente de l'intégration")
else:
    chk("paquet de la note (note.zip) : il porte les fichiers que la note des charges sur plancher déclare", not manque, manque)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
