# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Choix de profilé » (les barres en acier) sur le site — api_web/note_profil.py et ce qu'elle a
ajouté au kit des notes (ndc_pdf : la barre de λ̄, les substitutions repliées et coupées, le petit tableau, le verdict
riche, le peintre des dessins en mode riche) —, vérifiée en Python ordinaire, à travers le routeur comme le navigateur.

  1. TOUTE la note en un PDF valide : la garde du site (projet, partie, date, indice, une ligne par barre, sa page), puis
     la planche de chaque barre, dans l'ordre — sur douze notes : la barre neuve, la méthode historique seule, l'Eurocode
     seul, un poteau comprimé et fléchi en efforts imposés (l'annexe A), un tube rectangulaire, un tube rond, une poutre
     continue qui ne tient pas, une barre sans effort, trois barres dans une note, une note de la version 1 relue, un nom
     de 40 caractères sur une poutre de cinq travées au plus long nom de profilé (le résumé de l'en-tête se raccourcit),
     un nom de 60 caractères, la borne du site (il cède dans l'en-tête et sur la garde, par « … »).
  2. PARITÉ avec le moteur de l'écran (``profil.calculer``), barre par barre, par pdftotext : son nom, son profilé, son
     acier, son état ; les seules méthodes cochées ; chaque vérification — son titre, les deux membres de son critère,
     son taux — ; chaque RÉSULTAT et chaque TERME des formules que l'écran écrit ; le bilan des combinaisons ; la
     comparaison des deux méthodes ; les paramètres à confirmer ; la flèche ; ce que la note ne vérifie pas ; et les
     valeurs du DESSIN (les charges, la portée, les extrêmes des enveloppes).
  3. LA MISE EN PAGE : deux colonnes de calcul, jamais trois ; une planche qui déborde se POURSUIT (« suite ») ; aucun
     mot ne passe d'une colonne à l'autre ni ne sort de la page (une ligne trop longue se coupe devant un opérateur) ;
     les facteurs communs à (6.61) et (6.62) ne sont écrits qu'une fois.
  4. L'IDENTIFICATION : celle de la note du site (chantier et nom PIÉGÉS), dans la garde et dans l'en-tête de chaque
     page ; le nom du fichier.
  5. LE KIT : ce que la note lui a ajouté ne change RIEN aux notes qui ne s'en servent pas (une planche sans les genres
     neufs est dessinée par le code d'avant) ; chaque ajout fait ce qu'il dit.
  6. LES REFUS : une barre que le moteur refuse, une note sans barre.
  7. COMME DANS LE NAVIGATEUR : sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du
     paquet, la note se fait, son texte est celui de CPython, et elle ne charge du dépôt, en plus de l'écran et du
     kit, que son propre fichier.
  8. MUTATIONS : chacune doit faire rougir un contrôle.

    /usr/bin/python3.12 tests/test_note_profil.py
"""
import base64
import copy
import io
import json
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from acier.barre import ecriture as W  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_barre as DB  # noqa: E402
from api_web import note_profil as NP  # noqa: E402
from api_web import profil as P  # noqa: E402
from api_web import profil_mots as M  # noqa: E402
from api_web import profil_resultat as PR  # noqa: E402
from api_web import routeur as R  # noqa: E402
from ndc_pdf import mathx as MX  # noqa: E402
from ndc_pdf import riche as RI  # noqa: E402
from ndc_pdf import styles as STY  # noqa: E402
from ndc_pdf.kit import Doc, Frame  # noqa: E402

OK, KO = [], []
INFOS = {"nom": "Halle « nord » & auvent — lot 2", "chantier": "Rue de l'Écluse, 12", "indice": "C", "date": "14/11/2026"}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:500]}"))


# ------------------------------------------------------------------ les notes
def note(*saisies, barres=()):
    """Une note du site : la barre neuve, puis des saisies ; ``barres`` : d'autres barres, chacune ses saisies (une
    saisie « profil » passe par le sélecteur, comme à l'écran)."""
    etat = P.nouvel_etat()

    def saisir(liste):
        for chemin, valeur in liste:
            if chemin == "profil":
                P.choisir_section(etat, "profil", valeur)
            else:
                err = P.modifier(etat, chemin.split("/"), valeur)
                assert err is None, (chemin, valeur, err)

    saisir(saisies)
    for b in barres:
        P.gerer_elements(etat, "ajouter")
        saisir(b)
    return etat


NOM_60 = "POUTRE PRINCIPALE DU PLANCHER HAUT DU BÂTIMENT B, FILE 12-18"       # la borne du site, en capitales (les plus larges)
assert len(NOM_60) == P.NOM_MAX == 60, len(NOM_60)
EFFORTS = (("mode", "efforts"), ("efforts/elu/N", "400"), ("efforts/elu/My", "45"), ("efforts/elu/Vz", "30"), ("efforts/service/N", "290"),
           ("efforts/service/My", "32"), ("efforts/service/Vz", "21"))
ANCIENNE = {"version": 1, "actif": 0, "elements": [{"nom": "P1", "acier": "S355", "familles": ["HEB"], "My": 60.0, "Vz": 40.0, "avec_N": True, "N": 300.0,
                                                    "L": 4.0, "profil": "HEB 200"}]}
NOTES = {
    "la barre neuve": note(),
    "la méthode historique seule": note(("methodes", "historique"), ("nom", "H1")),
    "l'Eurocode seul": note(("methodes", "ec3"), ("nom", "E1"), ("cas/1/charges/+", "P")),
    "un poteau comprimé et fléchi (efforts imposés, annexe A)": note(("nom", "PT1"), ("profil", "HEB 200"), *EFFORTS),
    "un tube rectangulaire": note(("nom", "T1"), ("profil", "RHS 300x100x8"), *EFFORTS),
    "un tube rond": note(("nom", "R1"), ("profil", "CHS 219.1x6.3"), *EFFORTS),
    "une poutre continue qui ne tient pas": note(("nom", "C1"), ("modele", "continue"), ("travees/+", 1)),
    "une barre sans effort": note(("nom", "Z1"), ("mode", "efforts")),
    "trois barres dans une note": note(("cas/1/charges/+", "P"),
                                       barres=((("profil", "HEB 200"),) + EFFORTS, (("modele", "continue"), ("travees/+", 1), ("methodes", "ec3")))),
    "une note de la version 1": ANCIENNE,
    "un nom de 40 caractères sur cinq travées": note(("nom", "Poutre principale du plancher haut R+1 a"), ("profil", "SHS 400x400x12.5 (froid)"),
                                                    ("modele", "continue"), ("travees/+", 1), ("travees/+", 1), ("travees/+", 1)),
    "un nom de 60 caractères": note(("nom", NOM_60), ("profil", "SHS 400x400x12.5 (froid)"), ("modele", "continue"), ("travees/+", 1)),
}


def demander(etat, infos=INFOS):
    return json.loads(R.traiter(json.dumps({"module": "profil", "action": "telecharger", "etat": etat, "quoi": "note_pdf", "infos": infos})))


def faire(etat, infos=INFOS):
    """(octets, pages en texte brut, nom du fichier) ; lève si la note est refusée."""
    r = demander(etat, infos)
    assert r.get("ok"), r
    octets = base64.b64decode(r["fichier"]["base64"])
    return octets, [O.norm(t) for t in O.pages(octets, brut=True)], r["fichier"]["nom"]


def barres(etat):
    """[(barre relue, résultat du MOTEUR)] de la note — le moteur lui-même (``profil.calculer`` n'en est que la mémoire) :
    une note qui calculerait autre chose que l'écran ne trouve pas ici son propre écho."""
    e = P.normaliser(copy.deepcopy(etat))
    return [(el, MO.verifier(el)) for el in e["elements"]]


NOM_LISIBLE = 20            # ce qu'un nom qui cède garde au moins de lisible, en caractères


def titre(el, r):
    """Le titre d'une barre, en entier : « nom — profilé »."""
    return f"{el['nom']} — {r['profil']}"


def porte_titre(texte, el, r):
    """``texte`` porte-t-il le titre de la barre : « nom — profilé » en entier ou, pour un nom long qui a cédé, « début
    du nom… — profilé » — ce qui reste est le DÉBUT du nom, et il en reste au moins NOM_LISIBLE caractères ?
    (Écrit sans ``note_profil.entete`` : une note qui céderait trop, ou autre chose que le nom, ne trouve pas son écho.)"""
    nom, profil = O.norm(el["nom"]), O.norm(r["profil"])
    if f"{nom} — {profil}" in texte:
        return True
    m = re.search(re.escape(nom[:NOM_LISIBLE]) + r"([^—…]*)… — " + re.escape(profil), texte)
    return bool(m) and len(nom) > NOM_LISIBLE and nom.startswith((nom[:NOM_LISIBLE] + m.group(1)).rstrip())


def resume_attendu(el, r, court=False):
    """Le résumé d'une barre, sous son nom : « S235 · 6,00 m · 74 % historique · 52 % Eurocode 3 » — ``court`` : ses seuls
    taux. Réécrit ici, sur le résultat du moteur : la note n'est pas son propre témoin."""
    mots = [] if court else [el["nuance"]] + ([" + ".join(M.m(x) for x in el["travees"]) + " m"] if el["mode"] == "chargee" else [])
    for k, nom in (("historique", "historique"), ("ec3", "Eurocode 3")):
        t = (r["methodes"].get(k) or {}).get("taux_max")
        if t is not None:
            mots.append((f"> {C.PLAFOND_TAUX} %" if C.pct(t) > C.PLAFOND_TAUX else f"{C.pc(C.pct(t))} %") + f" {nom}")
    return " · ".join(mots)


def pages_de(pages, el, r):
    """Les pages de la planche d'une barre : celles dont l'en-tête porte son titre."""
    return [(i + 1, t) for i, t in enumerate(pages) if i and porte_titre(t[:200], el, r)]


def nombre_de(res: str) -> str:
    """Le nombre d'un résultat écrit (« 156,7 N/mm² » → « 156,7 »)."""
    return res.split(" ")[0]


def taux(t) -> str:
    """Un taux comme la fenêtre de calcul de l'écran l'écrit (« η = … = 51,9 % ») — réécrit ici : la note n'est pas son
    propre témoin."""
    return f"> {C.PLAFOND_TAUX} %" if t * 100.0 > C.PLAFOND_TAUX else W.nombre(t * 100.0, 1) + " %"


def comparables(r) -> bool:
    """Les deux méthodes sont rendues, avec au moins un taux à mettre en regard (réécrit ici)."""
    return len(r["methodes"]) == 2 and any(li["historique"] is not None or li["ec3"] is not None for li in r["comparaison"])


def entre(T: str, debut: str, fins) -> str:
    """Le texte d'un bloc : de son titre au titre du bloc suivant."""
    i = T.find(debut)
    if i < 0:
        return ""
    j = min((k for k in (T.find(f, i + len(debut)) for f in fins) if k > 0), default=len(T))
    return T[i:j]


def nu(t: str) -> str:
    """Un texte comparé sans ses blancs, ses soulignés ni la barre de λ̄ : pdftotext rend « χ_LT/f » en « χLT/f »."""
    return re.sub(r"[\s_\u0304]", "", t)


# ------------------------------------------------------------------ 2. la parité d'une note avec le moteur de l'écran
def parite(etat, pages):
    """Les écarts entre la note et le moteur, barre par barre : [(barre, quoi, attendu)]."""
    faux = []
    for el, r in barres(etat):
        mes = pages_de(pages, el, r)
        T = " ".join(t for _n, t in mes)

        def veut(quoi, attendu, present=None):
            ok = present if present is not None else (O.norm(attendu) in T)
            if not ok:
                faux.append((el["nom"], quoi, attendu))

        def nombre(quoi, attendu):
            veut(quoi, attendu, O.present(attendu, T))

        if not mes:
            faux.append((el["nom"], "aucune page", titre(el, r)))
            continue
        niv = P._niveau(r)
        veut("l'état", PR.mot_niveau(niv).upper())
        # (l'en-tête de CHAQUE page de la barre dit ses taux — le résumé entier s'il tient ; un nom de plus de 40 caractères
        # peut les lui prendre : la garde et les verdicts les disent)
        court = resume_attendu(el, r, court=True)
        if len(el["nom"]) <= 40 and court:
            veut("les taux dans l'en-tête de chaque page", court, all(O.norm(court) in t[:320] for _n, t in mes))
        veut("l'acier", f"Acier {el['nuance']}")
        nombre("la masse", f"{M.n(r['masse'], 1)} kg/m")
        for a in ("y", "z"):
            veut("la longueur de flambement", f"{M.n(r['longueurs'][a]['u'], 3)} × {M.m(r['longueurs'][a]['L'])} m")
        if el["mode"] == "chargee":
            for c in el["cas"]:
                veut("un cas de charge", f"{c['nom']} — ")
                for ch in c["charges"]:
                    veut("une charge", NP._charge(ch))
            for L in el["travees"]:
                nombre("une portée", f"{M.m(L)} m")
            env = DB.enveloppes(r)
            if env:                                             # les textes que SEUL le dessin écrit, et les extrêmes de ses enveloppes
                quoi = "ELU" if env[0] == "ec3" else "service"
                e = env[1]
                veut("le dessin : le nom de ses diagrammes", f"M ({quoi}) kN·m")
                veut("le dessin : le nom de ses diagrammes", f"V ({quoi}) kN")
                if len(el["travees"]) == 1:
                    veut("le dessin : la cote de la portée", f"L = {M.m(el['travees'][0])} m")
                nombre("le dessin : M max", C.fr(max(e["M_max"]), 1))
                nombre("le dessin : V max", C.fr(max(e["V_max"]), 1))
                veut("le dessin : le titre du panneau", "L A B A R R E")
        else:
            veut("le dessin : le titre du panneau", "L A S E C T I O N")
            dims = P.fiche_profil(el["profil"])["dims"]
            cote = "h" if "h" in dims else "d"                 # le texte que SEUL le dessin écrit : « h = 200 »
            veut("le dessin : la cote de la section", f"{cote} = {str(dims[cote]).rstrip('0').rstrip('.').replace('.', ',')}")
            for cle, _s, _u, _aide in M.EFFORTS:
                for j in ("service", "elu"):
                    k = {"service": "historique", "elu": "ec3"}[j]
                    if k in r["methodes"] and el["efforts"][j][cle]:
                        nombre("un effort imposé", M.r1(el["efforts"][j][cle]))
        for k in ("historique", "ec3"):
            tete = f"{M.METHODES[k]} — "
            if k not in r["methodes"]:
                veut("une méthode non cochée n'est pas imprimée", tete, tete not in T)
                continue
            x = r["methodes"][k]
            veut("la tête de la méthode", f"{M.METHODES[k]} — {x['norme']}")
            if x.get("taux_max") is not None:
                nombre("le taux de tête de la méthode", taux(x["taux_max"]))
            if len(x.get("combinaisons") or []) >= 2:
                veut("la combinaison qui gouverne", x["combinaison"])
            for v in x["verifications"]:
                veut("le titre d'une vérification", f"{M.METHODES_COURT[k]} · {v['titre'][:1].lower() + v['titre'][1:]}")
                if v["taux"] is not None:
                    nombre(f"le taux de {v['id']}", taux(v["taux"]))
                if v["valeur"] is not None and not PR.interaction(v):
                    signe = "≤" if v["statut"] != RE.NOK else ">"
                    nombre(f"la valeur de {v['id']}, puis le sens de la comparaison", f"{M.r1(v['valeur'])} {v['unite']} {signe}")
                    nombre(f"la limite de {v['id']}", f"{M.r1(v['limite'])} {v['unite']}")
                elif v["valeur"] is not None:
                    nombre(f"la somme de {v['id']}", f"η = {C.fr(v['valeur'], 3)}")
                for f, syms in PR.formules_de(x, v, taux=False):
                    _parts, res = W.morceaux(f, syms)
                    nombre(f"le résultat de {f['cle']}", nombre_de(res))
                    for t in f["termes"]:
                        nombre(f"un terme de {f['cle']}", W.nombre(t["valeur"]))
                for msg in v["messages"]:
                    veut(f"un message de {v['id']}", msg, nu(msg.split(" (")[0]) in nu(T))
        if comparables(r):
            # (le bloc peut passer d'une page à la suivante : l'en-tête de la page de suite, avec son résumé, tombe alors dedans)
            bloc = entre(T, "Comparaison des deux méthodes", ("Paramètres à confirmer", "Ce que la note ne vérifie pas"))
            for res in (resume_attendu(el, r), resume_attendu(el, r, court=True)):
                bloc = bloc.replace(O.norm(res), "")
            veut("la comparaison", "Comparaison des deux méthodes", bool(bloc))
            for li in r["comparaison"]:
                veut("un sujet de la comparaison", li["sujet"], O.norm(li["sujet"].split(",")[0]) in bloc)
                for k in ("historique", "ec3"):
                    if li[k] is not None:
                        veut(f"un taux de la comparaison ({li['sujet']}, {k})", taux(li[k]), O.present(taux(li[k]), bloc))
            ecrits = len(re.findall(r"\d %", bloc))
            veut("la comparaison n'écrit que ses taux", ecrits, ecrits == sum(1 for li in r["comparaison"] for k in ("historique", "ec3") if li[k] is not None))
        else:
            veut("pas de comparaison sans les deux méthodes", "Comparaison", "Comparaison des deux méthodes" not in T)
        ac = (r["methodes"].get("ec3") or {}).get("a_confirmer") or []
        veut("les paramètres à confirmer", "Paramètres à confirmer", ("Paramètres à confirmer" in T) == bool(ac))
        for p in ac:
            veut("la valeur d'un paramètre à confirmer", PR.valeur_param(p["valeur"]))
        fl = (r.get("statique") or {}).get("fleche")
        if fl and fl.get("valeur"):
            nombre("la flèche", f"{C.fr(abs(fl['valeur']), 1)} mm")
        veut("ce que la note ne vérifie pas", "Ce que la note ne vérifie pas")
        veut("… dont les raidisseurs", "raidisseurs")
    return faux


def sommaire(etat, pages):
    """Les écarts de la garde : chaque barre y a sa ligne, son état, et la page où sa planche commence."""
    faux = []
    garde = pages[0]
    annoncees = [int(n) for n in re.findall(r"page (\d+)", garde)]
    vraies = []
    for el, r in barres(etat):
        mes = pages_de(pages, el, r)
        vraies.append(mes[0][0] if mes else None)
        # (la garde nomme la barre en entier — elle en a la place ; un nom de 60 caractères y cède, par « … »)
        if not porte_titre(garde, el, r):
            faux.append((el["nom"], "absent de la garde", titre(el, r)))
        for quoi in (resume_attendu(el, r), PR.mot_niveau(P._niveau(r)).upper()):
            if O.norm(quoi) not in garde:
                faux.append((el["nom"], "absent de la garde", quoi))
    if annoncees != vraies:
        faux.append(("les pages du sommaire", annoncees, vraies))
    return faux


# ------------------------------------------------------------------ 3. la mise en page
def mots(octets):
    """[(page, largeur, hauteur, [(xMin, yMin, xMax, yMax, mot)])] (pdftotext -bbox)."""
    x = subprocess.run(["pdftotext", "-bbox", "-", "-"], input=octets, capture_output=True, check=True).stdout.decode("utf-8")
    out = []
    for i, m in enumerate(re.finditer(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', x, re.S), 1):
        ws = [(float(a), float(b), float(c), float(d), t) for a, b, c, d, t in
              re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', m.group(3))]
        out.append((i, float(m.group(1)), float(m.group(2)), ws))
    return out


def entetes_chevauches(octets):
    """Les pages dont deux mots se recouvrent : dans l'en-tête d'une planche (le titre passé sous le résumé ou sous la
    pastille), ou n'importe où sur la garde (le nom d'une barre passé sous sa pastille) : [(page, mot, mot)]."""
    faux = []
    for page, w, h, ws in mots(octets):
        zone = sorted((x for x in ws if w < h or x[1] < 62), key=lambda x: x[0])
        for i, a in enumerate(zone):
            for b in zone[i + 1:]:
                if b[0] >= a[2] - 0.5:
                    break
                if min(a[3], b[3]) - max(a[1], b[1]) > 2:
                    faux.append((page, a[4], b[4]))
    return faux


def debordements(octets):
    """Les mots d'une planche qui passent d'une colonne à l'autre, du panneau aux colonnes, ou sortent de la page."""
    S = NP.Planche()
    m, zx = S.margin, S.margin + S.coupe_w + S.gap
    gw = S.largeur_colonne()
    frontieres = (m + S.coupe_w + 11.0, zx + gw + S.col_gap / 2.0)       # le bord du panneau ; le filet entre les colonnes
    haut = m + 2 + 38                                                     # sous l'en-tête
    faux = []
    for page, w, h, ws in mots(octets):
        if w < h:
            continue                                                      # la garde
        for x0, y0, x1, y1, t in ws:
            if y0 < haut:
                continue
            if x1 > w - m + 1.0 or y1 > h - m + 6.0 or any(x0 < f - 1.0 and x1 > f + 1.0 for f in frontieres):
                faux.append((page, round(x0), round(x1), round(y1), t))
    return faux


# ------------------------------------------------------------------ les contrôles (rejoués par les mutations)
def controles(noms=None):
    """[(nom, ok, info)] sur les notes ``noms`` (toutes par défaut)."""
    out = []
    for nom in (noms or NOTES):
        etat = NOTES[nom]
        try:
            octets, pages, fichier = faire(etat)
        except AssertionError as err:
            out.append((f"« {nom} » : la note se fait", False, str(err)[:300]))
            continue
        bs = barres(etat)
        ecarts = parite(etat, pages)
        out.append((f"« {nom} » : un PDF de {len(pages)} pages qui dit, barre par barre, les chiffres du moteur de l'écran "
                    f"({sum(len(x['verifications']) for _e, r in bs for x in r['methodes'].values())} vérifications)",
                    octets[:5] == b"%PDF-" and len(pages) >= 1 + len(bs) and not ecarts, ecarts[:4]))
        s = sommaire(etat, pages)
        out.append((f"« {nom} » : la garde a une ligne par barre — son titre, son résumé, son état — et la page où sa planche commence", not s, s[:3]))
        d = debordements(octets) + entetes_chevauches(octets)
        out.append((f"« {nom} » : aucun mot ne passe d'une colonne à l'autre, du panneau aux colonnes, ni ne sort de la page ; dans l'en-tête, le "
                    "titre ne passe ni sous le résumé ni sous la pastille ; sur la garde, aucun mot n'en recouvre un autre", not d, d[:4]))
        ident = [i + 1 for i, t in enumerate(pages) if i and not (INFOS["date"] in t[:120] and f"indice {INFOS['indice']}" in t[:120])]
        garde = pages[0]
        out.append((f"« {nom} » : l'identification de la note du site (chantier et nom piégés, indice C, date) dans la garde et dans l'en-tête de "
                    "chaque page ; le nom du fichier",
                    not ident and all(O.norm(INFOS[k]) in garde for k in ("chantier", "nom", "date")) and "indice A" not in " ".join(pages)
                    and fichier == C.nom_fichier(f"Note de calcul - {INFOS['nom']}", "pdf") and fichier.endswith(".pdf"), (ident, fichier, garde[:200])))
    return out


def main():
    print("== 1, 2, 3, 4. douze notes : un PDF valide, les chiffres du moteur, la mise en page, l'identification")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 2 bis. ce que la note imprime, cas par cas")
    _o, pages, _f = faire(NOTES["la barre neuve"])
    T = " ".join(pages[1:])
    el, r = barres(NOTES["la barre neuve"])[0]
    h, e = r["methodes"]["historique"], r["methodes"]["ec3"]
    chk("la barre neuve : la méthode historique vient la PREMIÈRE, puis l'Eurocode 3, puis leur comparaison, les paramètres à confirmer, ce qui "
        "n'est pas vérifié — et les taux de tête sont ceux de l'écran (73,8 % ; 51,9 %)",
        0 < T.index("Méthode historique — NBN B 51-001") < T.index("Eurocode 3 — NBN EN 1993-1-1") < T.index("Comparaison des deux méthodes")
        < T.index("Paramètres à confirmer") < T.index("Ce que la note ne vérifie pas")
        and taux(h["taux_max"]) == "73,8 %" and taux(e["taux_max"]) == "51,9 %" and "73,8 %" in T and "51,9 %" in T, T[:300])
    chk("… la méthode historique s'écrit comme le classeur du bureau : σ_adm = f_y/s = 235 / 1,5 = 156,7 N/mm², les charges critiques, le "
        "déversement de la documentation (I_d, A_d, r_d, λ_ed, φ_ed), puis σ = σ_P + σ_My + σ_Mz",
        all(x in T for x in ("235 / 1,5", "156,7 N/mm", "4811 kN", "1390 kN", "301 cm", "19,3 cm", "3,94 cm", "76,1", "0,727", "115,6 N/mm")), T[1500:2200])
    chk("… la flèche de service est donnée (10,0 mm, L/599) et dite non vérifiée ; le poids propre est compté",
        "10,0 mm" in T and "L/599" in T and "elle n'est pas vérifiée" in T and "Poids propre du profilé : compté" in T)
    sec = NP.planche(el, r, NP.Planche())
    chk("… sa planche : un bloc de données, une tête par méthode, une vérification par bloc numéroté de 1 à n, chacune finie par son verdict",
        [v["num"] for v in sec["verifs"]] == list(range(1, len(sec["verifs"]) + 1))
        and len(sec["verifs"]) == 1 + sum(1 + len(x["verifications"]) for x in r["methodes"].values()) + 3
        and all(v["items"][-1] == ("k", 0) and len(v["verdicts"]) == 1 for v in sec["verifs"] if " · " in v["titre"] or " — NBN" in v["titre"]),
        [(v["num"], v["titre"]) for v in sec["verifs"]])
    _o, pages, _f = faire(NOTES["une poutre continue qui ne tient pas"])
    T = " ".join(pages[1:])
    el, r = barres(NOTES["une poutre continue qui ne tient pas"])[0]
    sec = NP.planche(el, r, NP.Planche())
    rouges = [v for v in sec["verifs"] if v["verdicts"] and v["verdicts"][0]["etat"] == "ko"]
    chk("la poutre continue qui ne tient pas : « NON CONFORME » en tête, le verdict rouge des vérifications dépassées et le signe « > » ; les "
        "portées L_1, L_2, L_3 au dessin et aux données",
        "NON CONFORME" in pages[1][:200] and sec["couleur"] == "ko" and len(rouges) >= 2 and all(" > " in v["verdicts"][0]["texte"] for v in rouges)
        and all(f"Travée {i}" in T for i in (1, 2, 3)), [(v["titre"], v["verdicts"][0]["texte"]) for v in rouges][:3])
    _o, pages, _f = faire(NOTES["une barre sans effort"])
    chk("la barre sans effort : « SANS EFFORT », « Aucun effort à vérifier » sous chaque méthode, pas de comparaison",
        "SANS EFFORT" in pages[1][:200] and " ".join(pages[1:]).count("Aucun effort à vérifier.") == 2 and "Comparaison des deux" not in " ".join(pages))
    etat = NOTES["un poteau comprimé et fléchi (efforts imposés, annexe A)"]
    octets, pages, _f = faire(etat)
    T = " ".join(pages[1:])
    el, r = barres(etat)[0]
    x = r["methodes"]["ec3"]
    v661, v662 = (next(v for v in x["verifications"] if v["id"] == i) for i in ("E-661", "E-662"))
    communes = set(NP.lignes_ecrites(x, v661)) & set(NP.lignes_ecrites(x, v662))
    chk("le poteau comprimé et fléchi : sa planche se POURSUIT (« (suite) » dans l'en-tête des pages suivantes) ; (6.62) ne répète pas les "
        f"{len(communes)} facteurs de (6.61) — une ligne le dit — et n'écrit que sa somme",
        len(pages) >= 3 and all("(suite)" in t[:200] for t in pages[2:]) and "(suite)" not in pages[1][:200] and len(communes) >= NP.COMMUNES_MIN
        and f"Les {len(communes)} facteurs écrits à la vérification" in T and T.count("0,79 + 0,21") == 2, (len(pages), len(communes), T.count("0,79 + 0,21")))
    chk("… « N_Ed/N_cr,y » n'est pas écrit deux fois (la formule en symboles répétait le symbole) ; la somme de (6.61) est celle du moteur",
        W.morceaux(next(f for f in x["calcul"] if f["cle"].endswith("rapport_y")))[0][0] == "N_Ed/N_cr,y"
        and len(W.morceaux(next(f for f in x["calcul"] if f["cle"].endswith("rapport_y")))[0]) == 2 and O.present(f"η = {C.fr(v661['valeur'], 3)}", T))

    print("== 5. le kit")
    ms = STY.Encre().ms()
    b = MX.measure(MX.parse(r"\bar{λ}_{LT,0}"), ms, 8.0)
    s_ = MX.measure(MX.parse(r"λ_{LT,0}"), ms, 8.0)
    chk("mathx : « \\bar{λ}_{LT,0} » — la barre ne change pas la largeur, elle monte au-dessus de la lettre, l'indice s'y accroche",
        abs(b.w - s_.w) < 1e-9 and b.a > s_.a and isinstance(MX.parse(r"\bar{λ}_{LT}").items[0], MX.Script), (b.w, s_.w, b.a, s_.a))
    chk("la note compose λ̄ : « λ̄_LT,0² » → « \\bar{λ}_{LT,0}^{2} » ; dans un texte, « λ̄_LT = 0,23 » devient un segment composé, « λ̄ » seul aussi",
        NP.tex("λ̄_LT,0²") == r"\bar{λ}_{LT,0}^{2}" and NP.riche("Déversement négligeable (λ̄_LT = 0,23 ≤ 0,40).") ==
        r"Déversement négligeable ($\bar{λ}_{LT}$ = 0,23 ≤ 0,40)." and NP.riche("λ̄ réduit") == r"$\bar{λ}$ réduit", NP.riche("(λ̄_LT = 0,23)"))
    long_ = "1 + (1,128 − 1) × ((2 − 1,6 / 1,128 × 1,0132^{2} × 0,74801 − 1,6 / 1,128 × 1,0132^{2} × 0,74801^{2}) × 0,21799 − 0)"
    parts, res = ["C_{yy}", "1 + (w_{y} − 1)·((2 − 1,6/w_{y}·C_{my}^{2}·λ_{max} − 1,6/w_{y}·C_{my}^{2}·λ_{max}^{2})·n_{pl} − b_{LT})", long_], r"\res{1,003}"
    avail = 225.0
    coupees = RI._lignes(parts, res, ms, 8.0, avail, 9.0, True)
    entieres = RI._lignes(parts, res, ms, 8.0, avail, 9.0, False)

    def large(t):
        return MX.width(RI.noeud(t), ms, 8.0)

    chk("riche : une ligne plus longue que la colonne se coupe DEVANT un opérateur — chaque ligne tient alors dans la colonne, à pleine "
        "taille ; sans l'option (les notes d'avant), elle reste entière et c'est sa taille qui se réduit",
        all(large(t) <= avail - dx + 1e-6 for t, dx in coupees) and any(large(t) > avail - dx + 1.0 for t, dx in entieres)
        and len(coupees) >= len(entieres) and all(t.split(" ")[0] in RI._OPERATEURS for t, dx in coupees[1:])
        and [dx for _t, dx in coupees] == [0.0, 9.0, 18.0], ([(round(large(t)), dx) for t, dx in coupees], [(round(large(t)), dx) for t, dx in entieres]))
    chk("… rien n'est perdu ni ajouté à la coupe : les lignes mises bout à bout redonnent la substitution ; une ligne coupée dans une parenthèse "
        "écrit les siennes en glyphes simples (le compositeur fermerait seul celle qui reste ouverte), et le refaire ne change plus rien",
        " ".join(t for t, _dx in coupees).replace(r"\t{(}", "(").replace(r"\t{)}", ")") == " = ".join(parts + [res])
        and any(r"\t{(}" in t for t, _dx in coupees) and all(RI._appariees(t) == t for t, _dx in coupees)
        and RI._appariees("(a + b)") == "(a + b)", [t for t, _dx in coupees])
    chk("… et la hauteur comptée pour faire basculer la colonne est celle des lignes dessinées (``lignes_suite``)",
        RI.lignes_suite(parts, res, ms, 8.0, avail, 9.0, couper=True) == [t for t, _dx in coupees]
        and RI.lignes_suite(parts, res, ms, 8.0, avail, 9.0) == [t for t, _dx in entieres])

    def planche_temoin(style, items, verdict):
        tampon = io.BytesIO()
        d = Doc(tampon, STY.A4L)
        d.warnings = []
        d.new_page(STY.A4L)
        style.planche(d, {"bureau": "B", "date": "1", "indice": "A"}, {"poutre": "P", "section": "S", "etat": "Vérifié", "blocs": [],
                                                                       "verifs": [{"num": 1, "titre": "T", "items": items, "verdicts": [verdict]}]})
        d.save()
        return tampon.getvalue()

    anciens = [("s", "Sous-titre"), ("v", "Libellé", "f_y", "235", "N/mm²"), ("f", "Une formule", "a = b + c"), ("p", "Un paragraphe."), ("k", 0)]
    # (le verdict d'AVANT porte déjà « gauche », « op », « droite », « ratio » — les deux membres de ce qu'il compare, comme la
    #  note de démonstration du kit, ndc_pdf/data.py : le mot calé à droite d'un verdict neuf ne s'appelle donc pas « droite »)
    VERDICT_AVANT = {"texte": "Hauteur de la poutre : 40 cm < hauteur minimale de la poutre : 73,2 cm", "ok": False,
                     "gauche": "40 cm", "op": "<", "droite": "73,2 cm", "ratio": 73.2 / 40.0}
    a1 = planche_temoin(STY.Encre(), anciens, VERDICT_AVANT)

    class Ancien(STY.Encre):                                   # la bande de verdict d'AVANT l'extension, recopiée
        def _verdict(self, d, fr, k):
            S = self
            col = S.coul_verdict(k)
            size = S.s_verd
            lines = d.wrap(k["texte"], S.f_b, size, fr.w - 15)
            h = len(lines) * size * 1.28 + 6.5
            fr.down(5)
            t = fr.y
            d.box(fr.x, t - h, fr.w, h, fill=STY.mix(col, "#FFFFFF", 0.90))
            d.box(fr.x, t - h + 1.5, 2.0, h - 3, fill=col)
            for i, ln in enumerate(lines):
                d.t(fr.x + 9, t - size * (1.02 + i * 1.28), ln, S.f_b, size, S.ink)
            fr.down(h)
            return fr.y

    a2 = planche_temoin(Ancien(), anciens, VERDICT_AVANT)

    def sans_dates(pdf):
        return re.sub(rb"/(CreationDate|ModDate) \([^)]*\)|/ID\s*\[[^\]]*\]", b"", pdf)

    chk("styles : une planche qui n'emploie AUCUN des genres neufs (« e », « pr », « tab », verdict riche ou à mot calé à droite) est dessinée à "
        "l'octet près comme par le code d'avant — les notes Poutre et Dalle ne changent pas ; un verdict d'avant, avec ses membres « gauche » et "
        "« droite », n'imprime PAS son membre de droite une seconde fois", bool(a1) and sans_dates(a1) == sans_dates(a2)
        and O.norm(" ".join(O.pages(a1, brut=True))).count("73,2 cm") == 1, (len(a1), len(a2)))
    neufs = [("pr", r"Un texte riche : $\bar{λ}_{LT}$ = 0,5."), ("e", None, parts, "1,003"),
             ("tab", [("SUJET", 120, "left"), ("TAUX", 60, "right")], [[("lab", "Flexion", True), ("eta", "73,8 %", True, False)]], "Une note."), ("k", 0)]
    d = Doc(io.BytesIO(), STY.A4L)
    d.warnings = []
    d.new_page(STY.A4L)
    S = NP.Planche()
    S.planche(d, {"bureau": "B", "date": "1", "indice": "A"}, {"poutre": "P", "section": "S", "etat": "Conforme", "couleur": "ok", "blocs": [],
                                                               "verifs": [{"num": 1, "titre": "T", "items": neufs, "verdicts": [
                                                                   {"texte": r"$M_{y,Ed}$ = 66,6 kN·m ≤ $M_{b,Rd}$ = 128,3 kN·m", "etat": "ok", "riche": True, "cale": "51,9 %"}]}]})
    chk("… les genres neufs se dessinent sans déborder (une substitution coupée, un texte riche, un tableau, un verdict riche avec son taux à droite)",
        not d.warnings, d.warnings)
    chk("… une planche garde DEUX colonnes de calcul et se poursuit sur une page suivante (jamais une troisième colonne) ; la réserve des données "
        "du panneau est celle du plus long de ses deux blocs, pas de leur somme",
        NP.Planche.n_cols == 2 and NP.Planche.planche_suite is True
        and NP.reserve_donnees([("A", [1, 2, 3]), ("B", [1] * 10)]) == NP.reserve_donnees([("B", [1] * 10)]) < 26 + 13 * (4 + 11))
    svg = DB.barre("t", *barres(NOTES["trois barres dans une note"])[2], largeur=NP.LARGEUR_DESSIN, legende=False)
    from ndc_pdf.svg import _a_des_parties, _parties
    import xml.etree.ElementTree as ET_
    textes = [el_ for el_ in ET_.fromstring(svg).iter() if el_.tag.endswith("text")]
    indices = [_parties(t) for t in textes if _a_des_parties(t)]
    chk("svg (mode riche) : l'indice d'un symbole du dessin (« L_2 = 6,00 m ») est une PARTIE plus petite, descendue, et la suite revient sur la "
        "ligne ; l'espace sans largeur qui porte ce retour n'est pas écrit ; l'aire d'une enveloppe est translucide",
        len(indices) == 3 and all(p[0][0] == "L" and p[1][1] < p[0][1] and p[1][2] > 0 and p[2][2] == 0 and "​" not in "".join(q[0] for q in p)
                                  for p in indices) and 'fill-opacity="0.13"' in svg, indices[:1])

    print("== 6. les refus")
    mauvaise = copy.deepcopy(NOTES["la barre neuve"])
    mauvaise["elements"][0]["profil"] = "IPE 9999"
    vide = {"version": 2, "elements": [], "actif": 0}
    r1, r2 = demander(mauvaise), demander(vide)
    chk("une barre que le moteur refuse (un profilé inconnu) : la note est REFUSÉE avec un message qui la nomme — jamais un PDF faux ; une "
        "note sans barre aussi", r1.get("ok") is False and "IPE 9999" in str(r1.get("erreur")) and "barre 1" in str(r1.get("erreur"))
        and r2.get("ok") is False and "barres" in str(r2.get("erreur")), (r1, str(r2)[:200]))
    chk("un fichier demandé qui n'existe pas est refusé", demander_autre() is False)

    print("== 7. comme dans le navigateur")
    etats = [NOTES["trois barres dans une note"], NOTES["un tube rond"]]
    s = O.sonde("profil", etats, INFOS, liste="elements")
    if "erreur" in s:
        chk("la note se fait comme dans le navigateur", False, s["erreur"])
    else:
        hors_kit = [f for f in s["note"] if not f.startswith(("ndc_pdf/", "site/note_pdf/"))]
        memes = [[O.norm(t) for t in O.pages(o, brut=True)] == faire(e)[1] for o, e in zip(s["sorties"], etats)]
        chk("sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet : la note se fait, son texte est celui de "
            "CPython ; en plus de l'écran et du kit des notes, elle ne charge du dépôt que son propre fichier",
            all(memes) and hors_kit == ["api_web/note_profil.py"] and not s["fonttools"] and not s["rl_accel"]
            and set(s["polices"]) <= set(s["polices_paquet"]) and "site/note_pdf" in s["pil"].replace(os.sep, "/"), (memes, hors_kit, s["polices"]))

    print("== 8. mutations")
    for nom, objet, attribut, fabrique, cibles in MUTATIONS:
        vrai = getattr(objet, attribut)
        setattr(objet, attribut, fabrique(vrai))
        P._RESULTATS.clear()
        try:
            rouges = [n for n, ok, _i in controles(cibles) if not ok]
        except Exception as err:  # noqa: BLE001 — une mutation qui fait planter la note est vue
            rouges = [f"exception : {err}"]
        finally:
            setattr(objet, attribut, vrai)
            P._RESULTATS.clear()
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges))
    temoin = [n for n, ok, _i in controles(["la barre neuve"]) if not ok]
    chk("témoin : sans mutation, aucun contrôle ne rougit", not temoin, temoin)

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print(f"   - {nom} | {str(info)[:300]}")
    return 1 if KO else 0


def demander_autre():
    r = json.loads(R.traiter(json.dumps({"module": "profil", "action": "telecharger", "etat": NOTES["la barre neuve"], "quoi": "autre"})))
    return r.get("ok")


# ------------------------------------------------------------------ 8. les mutations
def _une_barre_de_moins(vrai):
    return lambda etat, style=None: vrai(etat, style)[:-1]


def _autre_barre(vrai):
    def faux(el):
        el2 = copy.deepcopy(el)
        if el2["mode"] == "chargee":
            el2["cas"][0]["charges"][0]["v"] *= 1.2
        else:
            el2["efforts"]["elu"]["My"] *= 1.2
            el2["efforts"]["service"]["My"] *= 1.2
        return vrai(el2)
    return faux


def _taux_de_la_premiere(vrai):
    def faux(k, x, v, precedente=None):
        out = vrai(k, x, v, precedente)
        out["verdicts"][0]["cale"] = NP.taux_ecrit(x["verifications"][0]["taux"])
        return out
    return faux


def _formules_de_la_premiere(vrai):
    return lambda x, v: vrai(x, x["verifications"][0])


def _sans_dessin(vrai):
    return lambda c, svg, x, y, w, h, marge=2.0, riche=False: 1.0


def _toutes_les_methodes(vrai):
    def faux(el, r, style):
        el2 = dict(el, methodes=["historique", "ec3"])
        return vrai(el2, P.calculer(el2), style)
    return faux


def _sans_identification(vrai):
    return lambda etat, infos: {"nom_projet": "?", "partie": "?", "indice": "A"}


def _comparaison_fausse(vrai):
    def faux(r, largeur):
        r2 = dict(r, comparaison=[dict(li, historique=(li["historique"] or 0.0) + 0.1) for li in r["comparaison"]])
        return vrai(r2, largeur)
    return faux


class _SansCoupe(NP.Planche):
    """La planche sans la coupe des lignes longues : une substitution de l'annexe A sort de sa colonne."""

    def _items(self, d, fr, v, ms, cols, suivante=None):
        vrai = RI.formule_calee

        def entiere(*a, **k):
            k["couper"] = False
            k["min_size"] = 8.0
            return vrai(*a, **k)
        RI.formule_calee = entiere
        try:
            return super()._items(d, fr, v, ms, cols, suivante)
        finally:
            RI.formule_calee = vrai


POTEAU = ["un poteau comprimé et fléchi (efforts imposés, annexe A)"]
MUTATIONS = (
    ("une barre oubliée (sa planche n'est pas posée)", NP, "sections", _une_barre_de_moins, ["trois barres dans une note"]),
    ("la note calcule une autre barre que celle de l'écran (une charge × 1,2)", NP.P, "calculer", _autre_barre, ["la barre neuve"]),
    ("le taux d'une vérification est celui de la première de sa méthode", NP, "verification", _taux_de_la_premiere, ["la barre neuve"]),
    ("les formules d'une vérification sont celles de la première de sa méthode", NP, "lignes_ecrites", _formules_de_la_premiere, ["la barre neuve"]),
    ("le dessin du panneau n'est plus peint", NP, "peindre_svg", _sans_dessin, ["la barre neuve"]),
    ("une méthode non cochée est imprimée quand même", NP, "planche", _toutes_les_methodes, ["la méthode historique seule"]),
    ("l'identification de la note du site est perdue", NP, "infos_doc", _sans_identification, ["la barre neuve"]),
    ("la comparaison imprime d'autres taux que ceux du moteur (historique + 10 %)", NP, "comparaison_methodes", _comparaison_fausse, ["la barre neuve"]),
    ("la comparaison des deux méthodes n'est plus imprimée", NP.R, "comparable", lambda vrai: (lambda r: False), ["la barre neuve"]),
    ("les paramètres à confirmer ne sont plus imprimés", NP, "a_confirmer", lambda vrai: (lambda r, largeur: None), ["la barre neuve"]),
    ("ce que la note ne vérifie pas n'est plus dit", NP, "limites", lambda vrai: (lambda r: {"titre": "Remarques", "items": [], "verdicts": []}), ["la barre neuve"]),
    ("le périmètre non vérifié n'est plus nommé (les raidisseurs…)", NP, "LIMITES", lambda vrai: (), ["la barre neuve"]),
    ("la flèche n'est plus donnée", NP, "limites", lambda vrai: (lambda r: (lambda o: dict(o, items=o["items"][-1:]))(vrai(r))), ["la barre neuve"]),
    ("une ligne trop longue n'est plus coupée (elle sort de sa colonne)", NP, "Planche", lambda vrai: _SansCoupe, POTEAU),
    ("l'en-tête garde le résumé entier même quand il ne tient pas (le titre passe dessous)", NP, "entete",
     lambda vrai: (lambda el, r, mot, style: (vrai(el, r, mot, style)[0], NP.resume(el, r))), ["un nom de 40 caractères sur cinq travées"]),
    ("un nom de 60 caractères n'est plus cédé (il passe sous la pastille, dans l'en-tête et sur la garde)", NP, "_ceder",
     lambda vrai: (lambda texte, large, place: texte), ["un nom de 60 caractères"]),
    ("un nom de 60 caractères cède jusqu'à ne plus se lire (il n'en reste que dix lettres)", NP, "_ceder",
     lambda vrai: (lambda texte, large, place: texte if large(texte) <= place else texte[:10] + "…"), ["un nom de 60 caractères"]),
    ("le résumé écrit le nom de la méthode avant son taux (« Eurocode 3 9 % » se lit « 39 % »)", NP, "resume",
     lambda vrai: (lambda el, r, court=False: re.sub(r"(> \d+ %|[\d,]+ %) (historique|Eurocode 3)", r"\2 \1", vrai(el, r, court))),
     ["la barre neuve"]),
    ("le taux écrit sans décimale", NP, "taux_ecrit", lambda vrai: (lambda t: "—" if t is None else f"{t * 100:.0f} %"), ["la barre neuve"]),
    ("les deux membres d'un critère échangés", NP, "comparaison", lambda vrai: (lambda v: vrai(dict(v, valeur=v.get("limite"), limite=v.get("valeur")))),
     ["une poutre continue qui ne tient pas"]),
)


if __name__ == "__main__":
    sys.exit(main())
