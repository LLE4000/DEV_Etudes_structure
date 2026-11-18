# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Choix de profilé » (les barres en acier) sur le site — api_web/note_profil.py et ce qu'elle a
ajouté au kit des notes (ndc_pdf : la barre de λ̄, les substitutions repliées et coupées, le petit tableau, le verdict
riche, le peintre des dessins en mode riche) —, vérifiée en Python ordinaire, à travers le routeur comme le navigateur.

  1. TOUTE la note en un PDF valide : la garde du site (projet, partie, date, indice, une ligne par barre, sa page), puis
     la planche de chaque barre, dans l'ordre — sur seize notes : la barre neuve, la méthode historique seule, l'Eurocode
     seul, un poteau comprimé et fléchi en efforts imposés (l'annexe D ANB), le même avec γ_M0 ≠ γ_M1, un tube rectangulaire, un tube rond, une poutre
     continue qui ne tient pas, une barre sans effort, trois barres dans une note, une note de la version 1 relue, un nom
     de 40 caractères sur une poutre de cinq travées au plus long nom de profilé (le résumé de l'en-tête se raccourcit),
     un nom de 60 caractères, la borne du site (il cède dans l'en-tête et sur la garde, par « … ») ; et, pour l'âme
     (lot 4) : une poutre de reprise et son raidisseur, une charge au droit d'un appui (l'interaction, la force
     transmise d'une semelle à l'autre), une âme élancée (le voilement par cisaillement, des raidisseurs aux appuis).
  2. PARITÉ avec le moteur de l'écran (``profil.calculer``), barre par barre, par pdftotext : son nom, son profilé, son
     acier, son état ; les seules méthodes cochées ; chaque vérification — son titre, les deux membres de son critère,
     son taux — ; chaque RÉSULTAT et chaque TERME des formules que l'écran écrit ; le bilan des combinaisons ; L'ÂME ET
     SES RAIDISSEURS (EN 1993-1-5), imprimée quelle que soit la méthode — ses hypothèses, la décision de chaque lieu
     (sa force, « sans raidisseur » ou la paire de plats, son taux), chacune de ses vérifications, le tableau avant/après
     d'un raidisseur ; la SYNTHÈSE des deux méthodes (bloc 2, l'âme dans la colonne Eurocode 3, « — » pour une méthode
     qui n'a rien) ; les ÉCARTS avec les classeurs du bureau ; la torsion non couverte ; les décisions du bureau ; les
     paramètres à confirmer ; la flèche ; ce que la note ne vérifie pas ; et les valeurs du DESSIN.
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
  9. LE CRITÈRE DU LOT C (brief § 12, lot C, C5) : les notes de R2 (depuis les charges), du HEB 200 (efforts imposés) et
     du PRS-C (barre chargée, 8 maintiens, nature « exécution » : le lien C1 + C2), chacune dans les trois options — le
     PDF sort, les blocs de l'option sont là et les autres absents, la synthèse en tête, le statut imprimé suit les
     méthodes imprimées ; la page « Écarts » (6.7, dernier alinéa), la case « section soudée équivalente », le PRS au
     panneau.

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
from acier.barre import ec3_section as ES  # noqa: E402
from acier.barre import ecriture as W  # noqa: E402
from acier.barre import historique_deversement as HD  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_barre as DB  # noqa: E402
from api_web import note_profil as NP  # noqa: E402
from api_web import profil as P  # noqa: E402
from api_web import profil_mots as M  # noqa: E402
from api_web import profil_prs as PPR  # noqa: E402
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
    # (lot B : le seul paramètre « à confirmer » qui reste à l'interaction — γ_M0 ≠ γ_M1, brief § 7.8)
    "un poteau dont γ_M0 ≠ γ_M1 (n_pl à confirmer)": note(("nom", "PT2"), ("profil", "HEB 200"), *EFFORTS, ("parametres/gamma_M0", "1.05")),
    "un tube rectangulaire": note(("nom", "T1"), ("profil", "RHS 300x100x8"), *EFFORTS),
    "un tube rond": note(("nom", "R1"), ("profil", "CHS 219.1x6.3"), *EFFORTS),
    "une poutre continue qui ne tient pas": note(("nom", "C1"), ("modele", "continue"), ("travees/+", 1)),
    "une barre sans effort": note(("nom", "Z1"), ("mode", "efforts")),
    # (lot C, C0d — le harnais suit : ce que la note imprime est le choix de la NOTE, plus d'une barre ; la troisième ne coche plus « ec3 »)
    "trois barres dans une note": note(("cas/1/charges/+", "P"),
                                       barres=((("profil", "HEB 200"),) + EFFORTS, (("modele", "continue"), ("travees/+", 1)))),
    # (lot C, C0b) une console comprimée : l'Eurocode 3 ne couvre pas N + M en console — ses taux, montrés pour information, sans la
    # couleur « conforme », dans la comparaison aussi
    "une console comprimée (N + M non couvert)": note(("nom", "K1"), ("modele", "console"), ("travees/0", "3"), ("maintiens/0/-", "1"),
                                                     ("cas/0/N", "50")),
    "une note de la version 1": ANCIENNE,
    "un nom de 40 caractères sur cinq travées": note(("nom", "Poutre principale du plancher haut R+1 a"), ("profil", "SHS 400x400x12.5 (froid)"),
                                                    ("modele", "continue"), ("travees/+", 1), ("travees/+", 1), ("travees/+", 1)),
    "un nom de 60 caractères": note(("nom", NOM_60), ("profil", "SHS 400x400x12.5 (froid)"), ("modele", "continue"), ("travees/+", 1)),
    # (lot 4) une charge de 675 kN de calcul sur un HEB 300 : un raidisseur sous la charge, l'âme seule aux appuis (s_s = 200 mm)
    # (reprise du lot B : c, les s_s des charges et la semelle tenue sous la charge sont saisis — sans eux, décision impossible)
    "une poutre de reprise et son raidisseur": note(("nom", "PR1"), ("profil", "HEB 300"), ("travees/0", "4"), ("appuis/ss", "200"), ("appuis/c", "0"),
                                                   ("cas/0/charges/0/type", "P"), ("cas/0/charges/0/v", "300"), ("cas/1/charges/0/type", "P"),
                                                   ("cas/1/charges/0/v", "180"), ("cas/0/charges/0/ss", "0"), ("cas/1/charges/0/ss", "0"),
                                                   ("maintiens/0/x", "2")),
    # … une poutre continue dont l'appui du milieu porte aussi une charge : l'interaction avec la flexion, la force transmise (type b)
    "une charge au droit d'un appui": note(("nom", "CA1"), ("profil", "HEA 200"), ("modele", "continue"), ("travees/0", "5"), ("travees/1", "5"),
                                          ("appuis/ss", "80"), ("appuis/c", "20"), ("cas/0/charges/+", "P"), ("cas/0/charges/1/a", "5"),
                                          ("cas/0/charges/1/v", "20"), ("cas/0/charges/1/ss", "100")),
    # … une âme élancée (IPE 750x134 en S355) : le voilement par cisaillement vérifié, des raidisseurs aux appuis (lot A : avec
    # η = 1,00 pour un laminé — ANB:2011, 5.1(2) — l'IPE 750x147 ne le demande plus : h_w/t_w = 719/13,2 = 54,5 < 72ε/η = 58,6 ; l'IPE 750x134 : 719/12 = 59,9)
    "une âme élancée": note(("nom", "AE1"), ("profil", "IPE 750x134"), ("nuance", "S355"), ("travees/0", "8"), ("cas/0/charges/0/v", "60")),
    # (lot A) une nuance hors du domaine de la méthode historique (S460 : la NBN B 51-001 couvre AE235 à AE355) — le calcul est
    # imprimé, jamais « conforme » (brief § 9)
    "une nuance hors du domaine de la méthode historique (S460)": note(("nom", "HD1"), ("nuance", "S460")),
}


# ------------------------------------------------------------------ le CRITÈRE du lot C (brief § 12) : trois barres, trois options
def _q(v, a, b):
    return {"type": "q", "v": v, "a": a, "b": b, "niveau": None}


def _P(v, a, ss=None):
    return {"type": "P", "v": v, "a": a, "ss": ss, "niveau": None}


def barre_r2():
    """R2 (brief § 11.3) DEPUIS SES CHARGES : IPE 300 S235, 6 m sur deux appuis, G = 2,42 kN/m + 10 kN à 3 m, Q = 15 kN à 3 m
    (s_s = 100 mm sous la charge et aux appuis, c = 0), aucun maintien, charges sur la semelle haute."""
    b = P.barre_neuve("R2")
    b.update(poids_propre=False, maintiens=[], appuis={"mode": "poses", "ss": 100.0, "c": 0.0, "detail": None})
    b["cas"] = [dict(b["cas"][0], nom="G", charges=[_q(2.42, 0.0, 6.0), _P(10.0, 3.0, 100.0)]),
                dict(b["cas"][1], nom="Q", charges=[_P(15.0, 3.0, 100.0)])]
    return b


def barre_heb200():
    """HEB 200 S235 en EFFORTS IMPOSÉS (brief § 11.1 et 11.2) : L = 7,20 m, P = 432,72 kN de service et 649,08 kN tapé dans
    la colonne ELU — une donnée du test (la note n'écrit aucune « convention » entre les deux)."""
    b = P.barre_neuve("HEB 200")
    b.update(profil="HEB 200", mode="efforts", Lc=7.2, flambement={"y": {"L": 7.2, "u": 1.0, "courbe": None}, "z": {"L": 7.2, "u": 1.0, "courbe": None}},
             efforts={"service": {"N": 432.72, "My": 0.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}, "elu": {"N": 649.08, "My": 0.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}})
    return b


def barre_prsc(n=8):
    """PRS-C (brief § 11.9 ; la note du bureau du 16/11/2026) en BARRE CHARGÉE : 300 × 30 / 1 420 × 25 / 500 × 50, S235, 28 m,
    semelles oxycoupées, n maintiens de la semelle haute à L/(n + 1) ; l'acier au centre de gravité (le poids propre du
    module, permanent) ; le béton frais (20,80 kN/m), la charge de travail (2,40 kN/m) et la zone de travail (0,75 kN/m² ×
    3,0 m, de 12,5 à 15,5 m) en nature « exécution » (γ_Q = 1,5, ψ₀ = 1,0 saisi), sur la semelle haute — le LIEN C1 + C2.
    (À VÉRIFIER : le béton frais en « exécution », comme la note du bureau ; en permanent, M_Ed vaudrait 3 893,5 kN·m.)"""
    b = P.barre_neuve("PRS-C")
    b.update(profil="PRS", prs={"b1": 300.0, "t1": 30.0, "hw": 1420.0, "tw": 25.0, "b2": 500.0, "t2": 50.0, "a": None, "bords": "oxycoupes", "nom": None},
             nuance="S235", travees=[28.0], poids_propre=True, niveau="haut",
             maintiens=[{"x": 28.0 * i / (n + 1), "genre": "haut", "neglige": False} for i in range(1, n + 1)])
    b["cas"] = [dict(b["cas"][0], nom="E", nature="execution", psi0=1.0, charges=[_q(20.80 + 2.40, 0.0, 28.0), _q(2.25, 12.5, 15.5)])]
    return b


CRITERE = {f"{nom} · {imp}": P.etat_de([f()], impression=imp) for nom, f in (("R2", barre_r2), ("HEB 200", barre_heb200), ("PRS-C", barre_prsc))
           for imp in P.IMPRESSIONS}
# les blocs de la note par option (brief § 9 ; lot C, C5) — écrits ICI : (titre, imprimé avec cette option ?)
BLOCS = {"deux": {"synthese": True, "ecarts": True, "els": True, "ec3": True, "complementaires": False},
         "historique": {"synthese": False, "ecarts": True, "els": True, "ec3": False, "complementaires": True},
         "ec3": {"synthese": False, "ecarts": False, "els": False, "ec3": True, "complementaires": False}}


def controles_critere(nom):
    """[(nom, ok, info)] d'une note du critère : le PDF sort ; les blocs de son option sont là, les autres absents ; la synthèse
    vient en tête du détail ; le statut imprimé est celui des méthodes IMPRIMÉES et de l'âme ; la méthode non imprimée plus
    défavorable est dite ; la parité avec le moteur (``parite``)."""
    etat = CRITERE[nom]
    imp = etat[P.IMPRESSION]
    try:
        octets, pages, _f = faire(etat)
    except AssertionError as err:
        return [(f"critère « {nom} » : la note se fait", False, str(err)[:300])]
    T = " ".join(pages[1:])
    el, rv = barres(etat)[0]
    r = MO.verifier(el)
    attendu = BLOCS[imp]
    presents = {"synthese": SYNTHESE in T, "ecarts": ECARTS in T, "els": TETES["historique"] in T, "ec3": TETES["ec3"] in T,
                "complementaires": "Contrôles complémentaires EN 1993-1-5" in T}
    out = [(f"critère « {nom} » : un PDF ({len(pages)} pages) ; les blocs de l'option là, les autres absents ({presents})",
            octets[:5] == b"%PDF-" and len(pages) >= 2 and presents == attendu and DECISIONS in T and "Ce que la note ne vérifie pas" in T
            and "Torsion non couverte" in T and ("Paramètres à confirmer" in T) == bool((rv["methodes"].get("ec3") or {}).get("a_confirmer"))
            and "convention" not in T.lower(), presents)]
    if imp == "deux":
        out.append((f"critère « {nom} » : la synthèse en tête (bloc 2, après la barre, avant les écarts et le détail)",
                    0 < T.index("La barre et ses") < T.index(SYNTHESE) < T.index(ECARTS) < T.index(TETES["historique"]), T[:200]))
    mot = PR.mot_niveau(P._niveau(rv)).upper()
    mot_tout = PR.mot_niveau(P._niveau(r)).upper()
    out.append((f"critère « {nom} » : le statut imprimé (« {mot} ») est celui des méthodes imprimées et de l'âme — l'en-tête et la garde",
                mot in pages[1][:200] and mot in pages[0] and (mot == mot_tout or mot_tout not in pages[1][:200]), (mot, mot_tout, pages[1][:200])))
    autre = [k for k in r["methodes"] if k not in rv["methodes"]]
    if autre:
        # (« plus défavorable » : un statut pire que celui des méthodes imprimées, ou le même avec un taux plus fort — réécrit ici)
        x = r["methodes"][autre[0]]
        vus = RE.pire_statut([y["statut"] for y in rv["methodes"].values()])
        t_vus = max([y["taux_max"] for y in rv["methodes"].values() if y.get("taux_max") is not None] or [-1.0])
        dite = (x["statut"] != vus and RE.pire_statut([vus, x["statut"]]) == x["statut"]) or (
            x["statut"] == vus and (x.get("taux_max") or -1.0) > t_vus)
        ligne = {"historique": "Méthode ELS, non imprimé", "ec3": "Eurocode 3, non imprimé"}[autre[0]]
        out.append((f"critère « {nom} » : la méthode non imprimée, quand elle est plus défavorable, est dite en une ligne au bloc 1 ({dite})",
                    (ligne in T) == dite and (not dite or M.MOT_STATUT[x["statut"]].lower() in T), (ligne, dite)))
    # les décisions du bureau qui jouent (brief § 13.2) : R2 — η₁ (n° 1, la force et la flexion à mi-portée), le contact
    # inconnu (n° 2, type (c) aux appuis), le déversement des laminés (n° 4, Eurocode 3 imprimé) ; toujours le seuil (n° 7)
    jouent = DECISIONS_ATTENDUES[nom.split(" · ")[0]] - (set() if "ec3" in rv["methodes"] else {3, 4, 9, 10})
    dites = {int(m_) for m_ in re.findall(r"n° (\d+) · ", T)}
    out.append((f"critère « {nom} » : les décisions du bureau qui jouent, chacune avec son défaut « (appliqué) » et l'autre variante ({sorted(jouent)})",
                dites == jouent and T.count("(appliqué)") == len(jouent) - 1 and T.count("pour information") >= len(jouent) - 1, (sorted(dites), sorted(jouent))))
    # (lot C, C5) un message déjà dit n'est pas répété sur la planche : la remarque du type (c) des deux appuis de R2, une fois
    if nom.startswith("R2"):
        out.append((f"critère « {nom} » : la remarque du type (c), commune aux deux appuis, est imprimée une fois",
                    T.count("Type (c) : force près d") == 1, T.count("Type (c) : force près d")))
    ecarts_ = parite(etat, pages)
    out.append((f"critère « {nom} » : la parité avec le moteur de l'écran", not ecarts_, ecarts_[:4]))
    return out


DECISIONS_ATTENDUES = {"R2": {1, 2, 4, 7}, "HEB 200": {7}, "PRS-C": {7}}
# les choix du bureau, que la page « Écarts » ne présente jamais comme une correction (brief § 6.7, dernier alinéa)
CHOIX_DU_BUREAU = ("nominale", "s = 1,5", "C = 1", "C_d = 1", "courbes et classes", "non pondérées")


def controles_ecarts():
    """Brief § 6.7, dernier alinéa : la page « Écarts avec les classeurs du bureau » ne présente jamais un choix du bureau
    comme une correction — sa source (``historique_deversement.CORRECTIONS``) n'en nomme aucun, et chacune de ses lignes est
    une correction que le moteur rend pour la barre (ou dit qu'il n'y en a pas)."""
    mauvais = [k for k, t in HD.CORRECTIONS.items() if any(c in t for c in CHOIX_DU_BUREAU)]
    faux = []
    for etat in [CRITERE["R2 · deux"], CRITERE["PRS-C · historique"], NOTES["la barre neuve"], NOTES["une nuance hors du domaine de la méthode historique (S460)"]]:
        for el, rv in barres(etat):
            corr = [c["texte"][:1].upper() + c["texte"][1:] for c in (rv["methodes"].get("historique") or {}).get("corrections") or []]
            for it in NP.ecarts(rv, {})["items"]:
                t = it[2]
                if any(c in t for c in CHOIX_DU_BUREAU) or not (t == NP.AUCUN_ECART or any(t.startswith(c) for c in corr)):
                    faux.append((el["nom"], t))
    return [("6.7 : la page « Écarts » ne présente jamais un choix du bureau comme une correction (ni f_y nominale, ni s, ni C, ni C_d, ni "
             "« courbes et classes ») — CORRECTIONS n'en nomme aucun, et chaque ligne est une correction rendue par le moteur", not mauvais and not faux,
             (mauvais, faux[:3]))]


PRS_A = {"b1": 300.0, "t1": 30.0, "hw": 540.0, "tw": 15.0, "b2": 300.0, "t2": 30.0, "a": 6.0, "bords": "oxycoupes", "nom": None}


def _prs_a(soudee):
    b = P.barre_neuve("PRS-A")
    b.update(profil="PRS", prs=dict(PRS_A), nuance="S355", travees=[10.0], maintiens=[],
             parametres={"prs_soudee_equivalente": True} if soudee else {})
    b["cas"] = [dict(b["cas"][0], charges=[_q(20.0, 0.0, 10.0)]), dict(b["cas"][1], charges=[_q(10.0, 0.0, 10.0)])]
    return P.etat_de([b], impression="ec3")


def controles_soudee():
    """Autres tests PRS, n° 2 (brief § 11.8) : la case « section soudée équivalente » d'un PRS bisymétrique — cochée, le § 6.3.2.3
    décide et la note l'imprime (hypothèse et décision n° 10) ; non cochée, le § 6.3.2.2 décide, le § 6.3.2.3 pour information."""
    out = []
    for soudee in (True, False):
        etat = _prs_a(soudee)
        _o, pages, _f = faire(etat)
        T = " ".join(pages[1:])
        el, rv = barres(etat)[0]
        D = rv["methodes"]["ec3"]["grandeurs"]["deversement"]
        if soudee:
            ok = (D["methode"] == "§ 6.3.2.3" and "déclaré « section soudée équivalente » : § 6.3.2.3" in T
                  and "n° 10 · PRS bisymétrique « soudée équivalente » : § 6.3.2.3 (appliqué) ; § 6.3.2.2 : M" in T)
        else:
            ok = D["methode"] == "§ 6.3.2.2" and "n° 10 · PRS bisymétrique : § 6.3.2.2 (appliqué) ; § 6.3.2.3 : M" in T and "pour information" in T
        out.append((f"PRS-A bisymétrique, case « section soudée équivalente » {'cochée : le § 6.3.2.3 décide' if soudee else 'non cochée : le § 6.3.2.2 décide, le § 6.3.2.3 pour information'}"
                    " — la note le dit (hypothèse, décision n° 10)", ok, (D["methode"], [l for l in T.split("n° ") if l.startswith("10")][:1])))
    return out


def controles_prs_panneau():
    """Le PRS-C dans le PANNEAU (brief § 18.7, charte E1 : un seul endroit) : le croquis sous la barre (« LE PRS » au titre),
    ses plats, puis ses propriétés — celles de ``profil_prs.proprietes`` (z_s, z_j, ψ_f, f_y de chaque plat, f_y de calcul)."""
    etat = CRITERE["PRS-C · deux"]
    _o, pages, _f = faire(etat)
    T = " ".join(pages[1:])
    el, rv = barres(etat)[0]
    props = PPR.proprietes(PPR.section(el))
    manque = [(s_, v) for s_, v, _u, _o2 in props if O.norm(v) not in T]
    return [("PRS-C : le croquis coté dans le panneau, sous la barre (« LE PRS »), ses plats et le tableau de ses propriétés (profil_prs.proprietes)",
             "L E P R S" in T and "300 × 30" in T and "1 420 × 25" in T and "500 × 50" in T and not manque
             and all(x in T for x in ("Semelle haute", "De calcul", "Wagner")), manque[:4])]


# ------------------------------------------------------------------ la GARDE DES TEXTES (lot C, C5 ; règle du propriétaire du 16/11/2026)
LIGNE_MAX = 120
# exempté PAR SON NOM : le texte que le brief dicte (§ 6.4), repris tel quel
EXEMPTES = {HD.MESSAGE_CD: "MESSAGE_CD (brief § 6.4, 127 caractères)"}
# des textes d'AUTRES blocs de la vague 2, lus mot pour mot par leurs contrôles : à raccourcir à la fusion, puis à retirer d'ici
# (rapport du bloc C5, « hors_bloc ») — le seuil local de la vérification -FM (C4) ; la ligne « … sans maintien (EN 1993-1-5,
# 6.1(1)) — 37 % si elle est tenue ; … » (test_api_web_profil.py de C3, le harnais « barres » de C4)
A_LA_FUSION = ("Seuil de résistance locale à M_Ed", "sans maintien (EN 1993-1-5, 6.1(1)) — ")


def _exempte(t):
    return t in EXEMPTES or any(m in t for m in A_LA_FUSION)


def _barres_essai():
    """Les quinze barres d'essai (tests/test_barre_comparaison.py, la même liste) et les trois du critère."""
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import test_barre_comparaison as TC
    out = dict(TC.BARRES)
    for nom in ("R2 · deux", "HEB 200 · deux", "PRS-C · deux"):
        out[nom] = P.normaliser(copy.deepcopy(CRITERE[nom]))["elements"][0]
    return out


def textes_ecran(r):
    """Les blocs de texte d'un résultat du moteur, tels que l'écran et la note les montrent : [(bloc, [messages])] — la tête
    de chaque méthode et de l'âme (hypothèses, avertissements), chaque vérification (ses messages)."""
    out = []
    for k, x in list(r["methodes"].items()) + [("âme", r["ame"])]:
        out.append((k, list(x.get("hypotheses") or []) + [a["message"] for a in x.get("alertes") or []]))
        out += [(v["id"], list(v["messages"])) for v in x["verifications"]]
    return out


def textes_note(etat):
    """Les blocs de texte de la note : [(titre du bloc, [lignes de texte])] — le texte des lignes « pr » (le troisième
    élément, ``note_profil.pr``)."""
    e = P.normaliser(copy.deepcopy(etat))
    out = []
    for el in e["elements"]:
        sec = NP.planche(el, P.calculer(el), NP.Planche(), e[P.IMPRESSION])
        out += [(v["titre"], [i[2] for i in v["items"] if i[0] == "pr" and len(i) > 2]) for v in sec["verifs"]]
    return out


def controles_garde():
    """LA GARDE DES TEXTES, à l'écran ET dans la note (les quinze barres d'essai, les trois du critère ; les neuf notes du
    critère et les notes de ce fichier) : chaque message tient en 120 caractères ; aucun message n'est répété dans un bloc ;
    les limites (« Tout le calcul » de l'écran, la fin de la note) aussi. Exempté par son nom : ``MESSAGE_CD``."""
    longs, redits = [], []

    def lire(ou, blocs):
        for b, ts in blocs:
            vus = set()
            for t in ts:
                if len(t) > LIGNE_MAX and not _exempte(t):
                    longs.append((len(t), ou, b, t[:90]))
                if t in vus:
                    redits.append((ou, b, t[:90]))
                vus.add(t)

    for nom, b in _barres_essai().items():
        lire(f"écran · {nom}", textes_ecran(MO.verifier(b)))
    for nom, etat in list(CRITERE.items()) + [(k, NOTES[k]) for k in ("la barre neuve", "une poutre de reprise et son raidisseur", "une âme élancée",
                                                                        "une console comprimée (N + M non couvert)", "un poteau dont γ_M0 ≠ γ_M1 (n_pl à confirmer)",
                                                                        "une nuance hors du domaine de la méthode historique (S460)")]:
        lire(f"note · {nom}", textes_note(etat))
    lire("limites", [("LIMITES", [t[:1].upper() + t[1:] + "." for t in M.limites(True)])])
    longs.sort(reverse=True)
    return [(f"garde des textes : chaque message tient en {LIGNE_MAX} caractères, à l'écran et dans la note (quinze barres d'essai, trois du "
             f"critère, neuf notes du critère et six autres ; exempté par son nom : MESSAGE_CD)", not longs, longs[:4]),
            ("garde des textes : aucun message répété dans un bloc (écran et note)", not redits, redits[:4])]


SPECIAUX = {"écarts (6.7)": controles_ecarts, "PRS « soudée équivalente »": controles_soudee, "PRS-C au panneau": controles_prs_panneau,
            "garde des textes": controles_garde}


def demander(etat, infos=INFOS):
    return json.loads(R.traiter(json.dumps({"module": "profil", "action": "telecharger", "etat": etat, "quoi": "note_pdf", "infos": infos})))


def faire(etat, infos=INFOS):
    """(octets, pages en texte brut, nom du fichier) ; lève si la note est refusée."""
    r = demander(etat, infos)
    assert r.get("ok"), r
    octets = base64.b64decode(r["fichier"]["base64"])
    return octets, [O.norm(t) for t in O.pages(octets, brut=True)], r["fichier"]["nom"]


def vue(r, impression):
    """Ce que la note IMPRIME d'un résultat (lot C, C0d : le moteur calcule toujours les deux méthodes ; la note imprime les
    deux, ou l'une — ``etat["impression"]``) : les méthodes imprimées, la comparaison si ce sont les deux, le statut le plus
    défavorable des méthodes imprimées et de l'âme. Réécrit ici : la note n'est pas son propre témoin."""
    garde = ("historique", "ec3") if impression == "deux" else (impression,)
    if tuple(r["methodes"]) == garde:
        return r
    ms = {k: x for k, x in r["methodes"].items() if k in garde}
    v = dict(r, methodes=ms, comparaison=r["comparaison"] if len(ms) == 2 else [],
             statut=RE.pire_statut([x["statut"] for x in ms.values()] + [r["ame"]["statut"]]))
    if r.get("statique"):                   # les diagrammes des seules méthodes imprimées (service pour l'ELS, ELU pour l'Eurocode 3)
        v["statique"] = dict(r["statique"], enveloppes={k: e for k, e in r["statique"]["enveloppes"].items() if k in garde})
    return v


def barres(etat):
    """[(barre relue, résultat du MOTEUR tel que la note l'imprime)] de la note — le moteur lui-même (``profil.calculer`` n'en
    est que la mémoire) : une note qui calculerait autre chose que l'écran ne trouve pas ici son propre écho."""
    e = P.normaliser(copy.deepcopy(etat))
    return [(el, vue(MO.verifier(el), e[P.IMPRESSION])) for el in e["elements"]]


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


# les noms des méthodes, écrits ICI (la note n'est pas son propre témoin) : la méthode du bureau se nomme « Méthode ELS
# (NBN B 51-001) » en tête de son bloc — sa norme y est déjà — et « ELS » là où la place manque ; sa clé reste « historique »
TETES = {"historique": "Méthode ELS (NBN B 51-001)", "ec3": "Eurocode 3 — NBN EN 1993-1-1"}
# (lot C, C5) les blocs de la note, dans l'ordre du § 9 du brief : la barre, la synthèse, les écarts, le détail, les décisions
SYNTHESE, ECARTS, DECISIONS = "Synthèse des deux méthodes", "Écarts avec les classeurs du bureau", "Décisions du bureau appliquées"
COURTS = {"historique": "ELS", "ec3": "Eurocode 3"}


def resume_attendu(el, r, court=False):
    """Le résumé d'une barre, sous son nom : « S235 · 6,00 m · 74 % ELS · 58 % Eurocode 3 » — ``court`` : ses seuls
    taux. Réécrit ici, sur le résultat du moteur : la note n'est pas son propre témoin."""
    mots = [] if court else [el["nuance"]] + ([" + ".join(M.m(x) for x in el["travees"]) + " m"] if el["mode"] == "chargee" else [])
    for k, nom in COURTS.items():
        t = (r["methodes"].get(k) or {}).get("taux_max")
        if t is not None:
            mots.append((f"> {C.PLAFOND_TAUX} %" if C.pct(t) > C.PLAFOND_TAUX else f"{C.pc(C.pct(t))} %") + f" {nom}")
    k = sum(1 for L in r["ame"]["lieux"] if L["decision"] == "raidisseur")
    if k:
        mots.append(f"{k} raidisseur" + ("s" if k > 1 else ""))
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
        # (la f_y de l'Eurocode 3 : le tableau de l'ANB, la table unique du lot A ; une note « méthode ELS seule » dit la sienne, la
        # NOMINALE — lot B, brief v12 § 6.1)
        veut("… et la source de sa f_y : le tableau de l'ANB (la table unique, lot A), ou « nominale » pour la méthode ELS seule",
             f"Acier {el['nuance']} · " + ("tableau 3." if "ec3" in r["methodes"] else "nominale"))
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
            if el["poids_propre"]:                              # (lot A) le poids propre du MOTEUR : 78,6 kN/m³ × A
                veut("le poids propre, celui du moteur", f"Poids propre du profilé (78,6 kN/m³ × A = {M.n(r['poids_propre'], 3)} kN/m)")
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
        # (la note ne dit plus « historique », nulle part : ni un titre, ni une phrase, ni un message du moteur)
        veut("le nom de la méthode du bureau : « ELS », jamais « historique »", "historique", "istorique" not in T.lower())
        for k in ("historique", "ec3"):
            tete = TETES[k]
            if k not in r["methodes"]:
                veut("une méthode non cochée n'est pas imprimée", tete, tete not in T)
                continue
            x = r["methodes"][k]
            veut("la tête de la méthode", tete)
            if el["mode"] == "efforts" and k == "historique":
                veut("les efforts imposés de service : « (méthode ELS) »", "de service (méthode ELS)")
            veut("… sans répéter sa norme", f"{tete} — {x['norme']}", O.norm(f"{tete} — {x['norme']}") not in T)
            if x["statut"] == RE.HORS_DOMAINE:                 # (lot A) le mot de la méthode, jamais un mot de taux
                veut("une méthode hors de son domaine est dite « Hors domaine »", "Hors domaine — ")
                if comparables(r):                             # … et ses taux de la comparaison, sans la couleur « conforme »
                    tab = next(i for i in NP.comparaison_methodes(r, 160.0)["items"] if i[0] == "tab")[2]
                    col = 1 if k == "historique" else 2
                    veut("une méthode hors de son domaine : ses taux de la comparaison sans la couleur « conforme » (comme l'écran, « v »)",
                         k, all(li[col][0] == "num" for li in tab))
            if x.get("taux_max") is not None:
                nombre("le taux de tête de la méthode", taux(x["taux_max"]))
            if len(x.get("combinaisons") or []) >= 2:
                veut("la combinaison qui gouverne", x["combinaison"])
            for v in x["verifications"]:            # (un titre trop long pour la colonne perd le nom de sa méthode : lot C, C5)
                veut("le titre d'une vérification", f"{COURTS[k]} · {v['titre'][:1].lower() + v['titre'][1:]}",
                     O.norm(f"{COURTS[k]} · {v['titre'][:1].lower() + v['titre'][1:]}") in T or (len(v["titre"]) > 40 and O.norm(v["titre"]) in T))
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
                for msg in v["messages"]:        # (lot C, C5 : dit une fois sur la planche — peut-être dans un texte plus long)
                    veut(f"un message de {v['id']}", msg, nu(msg.split(" (")[0]).lower() in nu(T).lower())
        # ---- l'âme et ses raidisseurs (EN 1993-1-5) : imprimée quelle que soit la méthode cochée
        a = r["ame"]
        seule = list(r["methodes"]) == ["historique"]
        veut("la tête du bloc de l'âme", "Contrôles complémentaires EN 1993-1-5" if seule else f"Âme et raidisseurs — {a['norme']}")
        veut("… sous son seul titre", "l'autre titre", ("Contrôles complémentaires" in T) == seule)
        for h in a["hypotheses"]:
            veut("une hypothèse de l'âme", h, nu(h.split(" (")[0])[:60] in nu(T))
        if a["statut"] == RE.BLOQUE and a["verifications"]:          # (lot C, C0c) une donnée manque : « Bloqué », le mot de l'écran
            veut("l'âme bloquée est dite « Bloqué »", "Bloqué — ")
        for L in a["lieux"]:
            veut("un lieu de l'âme", L["nom"])
            # (2e reprise du lot B) la décision telle que l'écran la dit (profil_mots.decision_lieu, la source unique) : la paire de
            # plats, « sans raidisseur » — avec la borne d'une donnée non saisie —, ou « décision impossible : … » avec son taux
            decision = M.decision_lieu(L)
            veut(f"la décision de {L['nom']}", decision, nu(decision) in nu(T))
            nombre(f"la force de {L['nom']}", f"{M.r1(L['F_Ed'])} kN")
            if L["taux"] is not None:
                nombre(f"le taux de {L['nom']}", taux(L["taux"]))
                # sa LIGNE au tableau des décisions : le lieu, sa force, sa décision, son taux — côte à côte
                veut(f"la ligne de {L['nom']} au tableau des décisions", f"{L['nom']} {M.r1(L['F_Ed'])} kN {decision} {taux(L['taux'])}",
                     nu(f"{L['nom']}{M.r1(L['F_Ed'])}kN{decision}{taux(L['taux'])}") in nu(T))
        if a["verifications"] and a.get("taux_max") is not None:
            nombre("le taux de tête de l'âme", taux(a["taux_max"]))
        for v in a["verifications"]:
            veut("le titre d'une vérification de l'âme", v["titre"])
            if v["taux"] is not None:
                nombre(f"le taux de {v['id']}", taux(v["taux"]))
            if v["valeur"] is not None and not PR.interaction(v):
                signe = "≤" if v["statut"] != RE.NOK else ">"
                nombre(f"la valeur de {v['id']}, puis le sens de la comparaison", f"{M.r1(v['valeur'])} {v['unite']} {signe}".replace("  ", " "))
                nombre(f"la limite de {v['id']}", f"{M.r1(v['limite'])} {v['unite']}".strip())
            elif v["valeur"] is not None:
                nombre(f"la somme de {v['id']}", f"η = {C.fr(v['valeur'], 3)}")
            for f, syms in PR.formules_de(a, v, taux=False):
                _parts, res = W.morceaux(f, syms)
                nombre(f"le résultat de {f['cle']}", nombre_de(res))
                for t in f["termes"]:
                    nombre(f"un terme de {f['cle']}", W.nombre(t["valeur"]))
            for msg in v["messages"]:
                veut(f"un message de {v['id']}", msg, nu(msg.split(" (")[0])[:60].lower() in nu(T).lower())
        for x_ in a["alertes"]:
            veut("un avertissement de l'âme", x_["message"], nu(x_["message"])[:60] in nu(T))
        if any(L.get("avec") for L in a["lieux"]):          # (lot C, C5 ; brief § 15.4) sans et avec le raidisseur proposé, en tableau
            veut("le tableau avant/après du raidisseur", "SANS RAIDISSEUR", "SANS RAIDISSEUR" in T)
        if comparables(r):
            # (le bloc peut passer d'une page à la suivante : l'en-tête de la page de suite, avec son résumé, tombe alors dedans)
            bloc = entre(T, SYNTHESE, (ECARTS, TETES["historique"], TETES["ec3"]))
            for res in (resume_attendu(el, r), resume_attendu(el, r, court=True)):
                bloc = bloc.replace(O.norm(res), "")
            veut("la synthèse", SYNTHESE, bool(bloc))
            ecrits_attendus = 0
            for li in r["comparaison"]:
                # (la colonne Eurocode 3 : l'Eurocode 3, ou l'âme — commune aux deux méthodes — quand il n'a rien ou qu'elle est plus chargée)
                e_, a_ = li["ec3"], li.get("commun")
                ce = a_ if a_ is not None and (e_ is None or (a_["taux"] is not None and (e_["taux"] is None or a_["taux"] > e_["taux"]))) else e_
                if li["historique"] is None and ce is None:
                    continue
                veut("un sujet de la synthèse", li["sujet"], O.norm(li["sujet"].split(",")[0]) in bloc)
                for k, c_ in (("historique", li["historique"]), ("ec3", ce)):
                    if c_ is not None and c_["taux"] is not None:
                        ecrits_attendus += 1
                        veut(f"un taux de la synthèse ({li['sujet']}, {k})", taux(c_["taux"]), O.present(taux(c_["taux"]), bloc))
            ecrits = len(re.findall(r"\d %", bloc))
            veut("la synthèse n'écrit que ses taux", ecrits, ecrits == ecrits_attendus)
            # (lot C, C0b) un taux sous un statut bloqué, non couvert ou hors domaine : imprimé sans la couleur « conforme » ; (C5) la
            # colonne ELS dit « — » quand la méthode n'a rien sur le sujet (l'âme : « Contrôles complémentaires »)
            tab = next(i for i in NP.comparaison_methodes(r, 160.0)["items"] if i[0] == "tab")[2]
            lignes_ = [li for li in r["comparaison"] if li["historique"] is not None or li["ec3"] is not None or li.get("commun") is not None]
            for li, ligne in zip(lignes_, tab):
                if li["historique"] is None:
                    veut(f"la colonne ELS dit « — » ({li['sujet']})", "—", ligne[1] == ("num", "—"))
                for col, k in ((1, "historique"), (2, "ec3")):
                    c_ = li[k]
                    if c_ is not None and c_["taux"] is not None and c_["statut"] in (RE.BLOQUE, RE.NON_COUVERT, RE.HORS_DOMAINE):
                        veut(f"un taux sous « {c_['statut']} » imprimé sans la couleur « conforme » ({li['sujet']}, {k})", "num",
                             ligne[col][0] == "num")
        else:
            veut("pas de synthèse sans les deux méthodes", "Synthèse", SYNTHESE not in T)
        # (lot C, C5) les écarts avec les classeurs du bureau : avec la méthode ELS imprimée, une ligne par correction qui joue
        if "historique" in r["methodes"]:
            bloc = entre(T, ECARTS, (TETES["historique"],))
            corr = r["methodes"]["historique"].get("corrections") or []
            veut("les écarts avec les classeurs du bureau", ECARTS, bool(bloc))
            for c_ in corr:
                veut("une correction dans les écarts", c_["texte"], nu(c_["texte"][1:40]) in nu(bloc))
            if not corr:
                veut("aucune correction : la page le dit", NP.AUCUN_ECART, O.norm(NP.AUCUN_ECART) in bloc)
        else:
            veut("pas d'écarts sans la méthode ELS", ECARTS, ECARTS not in T)
        veut("la torsion non couverte (hypothèses communes)", "Torsion non couverte", "Torsion non couverte" in T)
        veut("les décisions du bureau appliquées, dont le seuil « à la limite » (n° 7)", "n° 7", DECISIONS in T and "n° 7 · « à la limite »" in T)
        ac = (r["methodes"].get("ec3") or {}).get("a_confirmer") or []
        veut("les paramètres à confirmer", "Paramètres à confirmer", ("Paramètres à confirmer" in T) == bool(ac))
        for p in ac:
            veut("la valeur d'un paramètre à confirmer", PR.valeur_param(p["valeur"]))
        fl = (r.get("statique") or {}).get("fleche")
        if fl and fl.get("valeur"):
            nombre("la flèche", f"{C.fr(abs(fl['valeur']), 1)} mm")
        veut("ce que la note ne vérifie pas", "Ce que la note ne vérifie pas")
        veut("… une ligne par limite, celles d'un PRS en plus", "les soudures âme-semelles", ("Les soudures âme-semelles du PRS" in T) == bool(el.get("prs")))
        veut("… dont les raidisseurs intermédiaires d'une âme élancée, les soudures et les assemblages", "raidisseurs intermédiaires")
        # (lot C, reprise de C0) les tableaux de l'âme (ses lieux) et le bilan des combinaisons (d'une méthode, de l'âme) : un taux sous un
        # statut bloqué, non couvert ou hors domaine est imprimé SANS la couleur « conforme » (à l'encre, « num ») ; un lieu qui passe,
        # en vert ; un refus, en rouge — la couleur de l'écran (profil_mots.couleur_taux)
        tabs_ = [i for i in NP.ame(el, r, 160.0)["items"] if i[0] == "tab"]
        if r["ame"]["lieux"]:
            for L, ligne in zip(r["ame"]["lieux"], tabs_[0][2]):
                c_ = ligne[3]
                attendu = ("num" if L["statut"] in M.STATUTS_A_VERIFIER else ("eta", L["statut"] != RE.NOK)) if L["taux"] is not None else None
                if attendu is not None:
                    veut(f"le taux du lieu « {L['nom']} » sous « {L['statut']} » à la couleur de son statut", attendu,
                         (c_[0] == "num") if attendu == "num" else (c_[0] == "eta" and c_[2] == attendu[1]))
        for x_ in list(r["methodes"].values()) + [r["ame"]]:
            t_ = NP._tableau_combinaisons(x_, 160.0)
            for ligne in (t_[2] if t_ else []):
                c_ = next(c for c in x_["combinaisons"] if c["nom"] == ligne[0][1])
                if c_["taux"] is not None and c_["statut"] in M.STATUTS_A_VERIFIER:
                    veut(f"le taux de la combinaison « {c_['nom']} » sous « {c_['statut']} » imprimé sans la couleur « conforme »", "num",
                         ligne[2][0] == "num")
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
    """[(nom, ok, info)] sur les notes ``noms`` (toutes par défaut) — une note du critère (``CRITERE``) et les contrôles
    nommés (``SPECIAUX``) aussi."""
    out = []
    for nom in (noms or NOTES):
        if nom in CRITERE:
            out += controles_critere(nom)
            continue
        if nom in SPECIAUX:
            out += SPECIAUX[nom]()
            continue
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
    print("== 1, 2, 3, 4. seize notes : un PDF valide, les chiffres du moteur, la mise en page, l'identification")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 9. le CRITÈRE du lot C (brief § 12) : R2, HEB 200 et PRS-C, chacune dans les trois options")
    for nom, ok, info in controles(list(CRITERE)):
        chk(nom, ok, info)
    # les chiffres que ces notes impriment, ceux du moteur — contre le brief (règle 7 : ≤ 1 %)
    vr = {n_: barres(CRITERE[f"{n_} · deux"])[0][1] for n_ in ("R2", "HEB 200", "PRS-C")}

    def vf(r, k, i):
        return next(v for v in r["methodes"][k]["verifications"] if v["id"] == i)
    my, vz = vf(vr["R2"], "ec3", "E-My"), vf(vr["R2"], "ec3", "E-Vz")
    nbz, hnz = vf(vr["HEB 200"], "ec3", "E-Nb-z"), vf(vr["HEB 200"], "historique", "H-N-z")
    mb, dv = vf(vr["PRS-C"], "ec3", "E-Mb"), vf(vr["PRS-C"], "historique", "H-DEV")
    chk("les chiffres du critère : R2 depuis les charges M_Ed = 68,70 kN·m, V_Ed = 27,80 kN (brief § 11.3) ; HEB 200 N_b,z,Rd = 570,1 kN (11.2, "
        "± 0,2 %), méthode ELS 1,118 (11.1, ± 0,2 %) ; PRS-C (lien C1 + C2, nature « exécution ») M_Ed = 4 199,4 kN·m (± 0,03 %), taux 0,967, "
        "méthode ELS σ_adm,d = 89,68 N/mm² (11.9)",
        abs(my["valeur"] - 68.70) < 0.005 and abs(vz["valeur"] - 27.80) < 0.005 and abs(nbz["limite"] / 570.1 - 1) < 0.002
        and abs(hnz["taux"] / 1.118 - 1) < 0.002 and abs(mb["valeur"] / 4199.4 - 1) < 0.0003 and abs(mb["taux"] - 0.967) <= 0.001
        and abs(dv["limite"] - 89.68) <= 0.01, (my["valeur"], vz["valeur"], nbz["limite"], hnz["taux"], mb["valeur"], mb["taux"], dv["limite"]))
    print("== 9 bis. la page « Écarts », la case « section soudée équivalente », le PRS au panneau")
    for nom, ok, info in controles(list(SPECIAUX)):
        chk(nom, ok, info)

    print("== 2 bis. ce que la note imprime, cas par cas")
    # (lot B, brief v12 § 6.1, test 37) la méthode ELS prend la f_y NOMINALE, l'Eurocode 3 la table selon l'épaisseur : la ligne
    # « Acier » des données dit celle de l'Eurocode 3, les hypothèses de la méthode ELS la sienne
    hp = dict(P.barre_neuve(), profil="HEB 300", mode="efforts", Lc=5.0, flambement={"y": {"L": 5.0, "u": 1.0}, "z": {"L": 5.0, "u": 1.0}},
              efforts={"service": {"N": 1000.0, "My": 50.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}, "elu": {"N": 1400.0, "My": 70.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}})
    rh = MO.verifier(hp)
    dn = NP.donnees(hp, rh)
    acier = next(it for it in dn["items"] if it[0] == "v" and it[1].startswith("Acier"))
    seule = NP.donnees(hp, vue(MO.verifier(hp), "historique"))       # (lot C, C0d : la note qui n'imprime que l'ELS)
    acier_els = next(it for it in seule["items"] if it[0] == "v" and it[1].startswith("Acier"))
    tete = NP.methode(hp, "historique", rh["methodes"]["historique"], 300.0)
    chk("HEB 300 S235 (t_f = 19 mm), les deux méthodes : la ligne « Acier » dit la f_y de l'Eurocode 3 (225, tableau 3.3 ANB) ; la méthode ELS "
        "écrit la sienne dans son hypothèse (235 / 1,5, « f_y nominale, comme les classeurs du bureau ») ; seule, la ligne dit « nominale, 235 »",
        acier[1] == "Acier S235 · tableau 3.3 ANB" and acier[3] == "225" and acier_els[1] == "Acier S235 · nominale" and acier_els[3] == "235"
        and any("235 / 1,5 = 156,7 N/mm² ($f_{y}$ nominale, comme les classeurs du bureau)." in it[1] for it in tete["items"][:3] if it[0] == "pr"),
        (acier, acier_els, tete["items"][:2]))
    _o, pages, _f = faire(NOTES["la barre neuve"])
    T = " ".join(pages[1:])
    el, r = barres(NOTES["la barre neuve"])[0]
    h, e = r["methodes"]["historique"], r["methodes"]["ec3"]
    chk("la barre neuve (lot C, C5 : l'ordre du § 9) : la barre, la SYNTHÈSE, les écarts avec les classeurs, la méthode ELS (NBN B 51-001), "
        "l'Eurocode 3, l'âme, les décisions du bureau, ce qui n'est pas vérifié — et les "
        "taux de tête sont ceux de l'écran (88,4 % — lot B : le déversement des classeurs ; 57,6 % — lot C : le poids propre au centre de gravité, 57,7 % avant) ; plus de « paramètres à confirmer » (lot A : le couple (λ̄_LT,0 ; β) est celui de "
        "l'ANB:2018, 6.3.2.3(1), lu) ; la nuance dit son tableau de l'ANB (la table unique)",
        0 < T.index("La barre et ses charges") < T.index(SYNTHESE) < T.index(ECARTS) < T.index("Méthode ELS (NBN B 51-001)")
        < T.index("Eurocode 3 — NBN EN 1993-1-1") < T.index("Âme et raidisseurs — NBN") < T.index(DECISIONS)
        < T.index("Ce que la note ne vérifie pas") and "Paramètres à confirmer" not in T and "Acier S235 · tableau 3.3 ANB" in T
        and taux(h["taux_max"]) == "88,4 %" and taux(e["taux_max"]) == "57,6 %" and "88,4 %" in T and "57,6 %" in T, T[:300])
    chk("… la méthode ELS s'écrit comme les classeurs du bureau (lot B, brief v12 § 6.4) : σ_adm = f_y/s = 235 / 1,5 = 156,7 N/mm² avec la f_y "
        "NOMINALE (« f_y nominale, comme les classeurs du bureau », test 37), le déversement des classeurs (λ_e,d, λ̄_d, Φ_d, φ_d, σ_adm,d) et "
        "M/W ≤ σ_adm,d, l'origine du 3,9, la correction du classeur qui joue (« Correction du classeur : … ») ; aucune mention « plus sévère "
        "que l'EC3 » (test 12 du § 11.5)",
        all(x in T for x in ("235 / 1,5", "156,7 N/mm", "nominale, comme les classeurs du bureau", "92,35", "0,983", "2,2326", "0,6081", "95,26 N/mm",
                             "84,19 N/mm", "3,9 = √(3π·√2,6) arrondi", "Correction du classeur : classe de la ligne « axe faible » seule"))
        and "plus sévère" not in T, T[1500:2200])
    chk("… la flèche de service est donnée (10,0 mm, L/599) et dite non vérifiée ; le poids propre est compté, et sa valeur dite (78,6 kN/m³ × A, "
        "la valeur du moteur)",
        "10,0 mm" in T and "L/599" in T and "elle n'est pas vérifiée" in T and "Poids propre du profilé (78,6 kN/m³ × A = 0,423 kN/m) : compté" in T)
    sec = NP.planche(el, r, NP.Planche())
    chk("… sa planche : un bloc de données, une tête par méthode, la tête de l'âme, une vérification par bloc numéroté de 1 à n, chacune finie "
        "par son verdict",
        [v["num"] for v in sec["verifs"]] == list(range(1, len(sec["verifs"]) + 1))
        and len(sec["verifs"]) == 1 + sum(1 + len(x["verifications"]) for x in list(r["methodes"].values()) + [r["ame"]]) + 4
        and all(v["items"][-1] == ("k", 0) and len(v["verdicts"]) == 1 for v in sec["verifs"]
                if " · " in v["titre"] or " — NBN" in v["titre"] or v["titre"] in TETES.values() or " : âme" in v["titre"])
        # (reprise du lot B, le harnais suit : s_s et c non saisis → décision impossible (brief § 8.3, point 5), les vérifications à
        #  s_s = c = 0 imprimées pour information)
        and [v["titre"] for v in sec["verifs"]][10:13] == ["Âme et raidisseurs — NBN EN 1993-1-5", "Appui A : critère élastique à la jonction (6.2.1(5))",
                                                          "Appui A : âme sous la réaction"], [(v["num"], v["titre"]) for v in sec["verifs"]])
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
    chk("la barre sans effort : « SANS EFFORT », « Aucun effort à vérifier » sous chaque méthode, pas de synthèse",
        "SANS EFFORT" in pages[1][:200] and " ".join(pages[1:]).count("Aucun effort à vérifier.") == 2 and SYNTHESE not in " ".join(pages))
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
def _source_mutee(module, ancien, nouveau):
    """Une copie d'un fichier, son TEXTE muté (une seule occurrence de ``ancien`` : sinon la suite rougit, la mutation ne
    passerait pas pour rouge sans avoir rien muté)."""
    import importlib.util
    src = open(module.__file__, encoding="utf-8").read()
    if src.count(ancien) != 1:
        chk(f"le texte à muter est dans {module.__name__}, une seule fois : « {ancien.strip()[:70]} »", False, src.count(ancien))
        raise ValueError(f"texte à muter absent ou répété dans {module.__name__}")
    spec = importlib.util.spec_from_file_location(module.__name__ + "_mute", module.__file__)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module.__package__
    exec(compile(src.replace(ancien, nouveau), module.__file__, "exec"), mod.__dict__)
    return mod


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
    def faux(k, x, v, precedente=None, *a, **kw):
        out = vrai(k, x, v, precedente, *a, **kw)
        out["verdicts"][0]["cale"] = NP.taux_ecrit(x["verifications"][0]["taux"])
        return out
    return faux


def _formules_de_la_premiere(vrai):
    return lambda x, v: vrai(x, x["verifications"][0])


def _sans_dessin(vrai):
    return lambda c, svg, x, y, w, h, marge=2.0, riche=False: 1.0


def _toutes_les_methodes(vrai):
    def faux(el, r, style, impression="deux"):
        return vrai(el, r, style, "deux")
    return faux


def _sans_identification(vrai):
    return lambda etat, infos: {"nom_projet": "?", "partie": "?", "indice": "A"}


def _comparaison_fausse(vrai):
    def faux(r, largeur, num=None):
        r2 = dict(r, comparaison=[dict(li, historique=dict(li["historique"], taux=(li["historique"]["taux"] or 0.0) + 0.1) if li["historique"] else None)
                                  for li in r["comparaison"]])
        return vrai(r2, largeur, num)
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


def _synthese_a_la_fin(vrai):
    def faux(el, r, style, impression="deux"):
        out = vrai(el, r, style, impression)
        vs = out["verifs"]
        k = next((i for i, v in enumerate(vs) if v["titre"] == SYNTHESE), None)
        if k is not None:
            vs.append(vs.pop(k))
            for i, v in enumerate(vs, 1):
                v["num"] = i
        return out
    return faux


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
    # (lot A : la barre neuve n'a plus de paramètre à confirmer ; lot B : le poteau comprimé et fléchi non plus — seul γ_M0 ≠ γ_M1 en garde un)
    ("les paramètres à confirmer ne sont plus imprimés", NP, "a_confirmer", lambda vrai: (lambda r, largeur: None),
     ["un poteau dont γ_M0 ≠ γ_M1 (n_pl à confirmer)"]),
    ("ce que la note ne vérifie pas n'est plus dit", NP, "limites", lambda vrai: (lambda el, r: {"titre": "Remarques", "items": [], "verdicts": []}),
     ["la barre neuve"]),
    ("le périmètre non vérifié n'est plus nommé (les raidisseurs…)", M, "LIMITES", lambda vrai: (), ["la barre neuve"]),
    ("la flèche n'est plus donnée", NP, "limites", lambda vrai: (lambda el, r: (lambda o: dict(o, items=o["items"][1:]))(vrai(el, r))), ["la barre neuve"]),
    ("une ligne trop longue n'est plus coupée (elle sort de sa colonne)", NP, "Planche", lambda vrai: _SansCoupe, POTEAU),
    ("l'en-tête garde le résumé entier même quand il ne tient pas (le titre passe dessous)", NP, "entete",
     lambda vrai: (lambda el, r, mot, style: (vrai(el, r, mot, style)[0], NP.resume(el, r))), ["un nom de 40 caractères sur cinq travées"]),
    ("un nom de 60 caractères n'est plus cédé (il passe sous la pastille, dans l'en-tête et sur la garde)", NP, "_ceder",
     lambda vrai: (lambda texte, large, place: texte), ["un nom de 60 caractères"]),
    ("un nom de 60 caractères cède jusqu'à ne plus se lire (il n'en reste que dix lettres)", NP, "_ceder",
     lambda vrai: (lambda texte, large, place: texte if large(texte) <= place else texte[:10] + "…"), ["un nom de 60 caractères"]),
    ("le résumé écrit le nom de la méthode avant son taux (« Eurocode 3 9 % » se lit « 39 % »)", NP, "resume",
     lambda vrai: (lambda el, r, court=False: re.sub(r"(> \d+ %|[\d,]+ %) (ELS|Eurocode 3)", r"\2 \1", vrai(el, r, court))),
     ["la barre neuve"]),
    ("le taux écrit sans décimale", NP, "taux_ecrit", lambda vrai: (lambda t: "—" if t is None else f"{t * 100:.0f} %"), ["la barre neuve"]),
    # ---- l'âme et ses raidisseurs (lot 4)
    ("le bloc de l'âme n'est plus imprimé", NP, "ame", lambda vrai: (lambda el, r, largeur: {"titre": "Remarques", "items": [], "verdicts": []}),
     ["la barre neuve"]),
    ("le tableau des décisions (un lieu, sa force, sa décision, son taux) n'est plus imprimé", NP, "ame",
     lambda vrai: (lambda el, r, largeur: (lambda o: dict(o, items=[i for i in o["items"] if i[0] != "tab"]))(vrai(el, r, largeur))),
     ["une poutre de reprise et son raidisseur"]),
    ("une note « méthode historique seule » perd son titre « Contrôles complémentaires EN 1993-1-5 »", NP, "titre_ame",
     lambda vrai: (lambda r: f"{M.TITRE_AME} — {r['ame']['norme']}"), ["la méthode historique seule"]),
    # (lot C, C5) l'âme, commune aux deux méthodes, dans la colonne Eurocode 3 de la synthèse quand il n'a rien sur le sujet
    ("l'âme n'est plus dans la synthèse (colonne Eurocode 3)", NP, "_colonne_ec3", lambda vrai: (lambda li: (li.get("ec3"), True)),
     ["une charge au droit d'un appui"]),
    ("le résumé ne dit plus le raidisseur que la barre demande", NP, "resume",
     lambda vrai: (lambda el, r, court=False: vrai(el, r, court).replace(" · 1 raidisseur", "")), ["une poutre de reprise et son raidisseur"]),
    ("les vérifications de l'âme ne sont plus imprimées (sa tête seule)", NP, "planche",
     lambda vrai: (lambda el, r, style, impression="deux": vrai(el, dict(r, ame=dict(r["ame"], verifications=[])), style, impression)), ["une âme élancée"]),
    # (lot A) la source de f_y, le mot « Hors domaine », le poids volumique du moteur
    ("la nuance ne dit plus le tableau de l'ANB d'où vient sa f_y", ES, "tableau_fy", lambda vrai: (lambda nuance, s: None), ["la barre neuve"]),
    ("une méthode hors de son domaine est dite d'un mot de taux (« À vérifier »)", NP.R, "mot_methode",
     lambda vrai: (lambda x: NP.R.mot_niveau(NP.R.niveau_methode(x))), ["une nuance hors du domaine de la méthode historique (S460)"]),
    ("la comparaison peint le taux d'une méthode hors domaine de la couleur « conforme »", NP.R, "niveau_taux",
     lambda vrai: (lambda case: C.niveau(case["taux"]) if case and case["taux"] is not None else "c"),
     ["une nuance hors du domaine de la méthode historique (S460)"]),
    # (lot C, C0c) le statut « bloqué » dit d'un mot de niveau (« À vérifier ») dans la note
    ("l'âme bloquée (décision impossible) est dite « À vérifier »", NP.R, "mot_methode",
     lambda vrai: (lambda x: NP.R.mot_niveau(NP.R.niveau_methode(x)) if x["statut"] == RE.BLOQUE else vrai(x)), ["la barre neuve"]),
    # (lot C, C0b) la même, sur la console comprimée : l'Eurocode 3 non couvert, ses taux en vert
    ("la comparaison peint un taux non couvert de la couleur « conforme »", NP.R, "niveau_taux",
     lambda vrai: (lambda case: C.niveau(case["taux"]) if case and case["taux"] is not None else "c"), ["une console comprimée (N + M non couvert)"]),
    # (lot C, reprise de C0) les lieux de l'âme et le bilan des combinaisons peints au taux seul — le défaut mesuré : 156,7 % « décision
    # impossible » en vert aux appuis de la barre neuve ; l'Eurocode 3 « non couvert » de la console, ses combinaisons en vert
    ("les lieux de l'âme peints « conformes » dès qu'ils ne refusent pas (L[\"statut\"] != RE.NOK)", NP, "ame",
     lambda vrai: _source_mutee(NP, 'case_taux(L["statut"], L["taux"], gov)', '("eta", taux_ecrit(L["taux"]), L["statut"] != RE.NOK, gov)').ame,
     ["la barre neuve"]),
    ("le bilan des combinaisons peint « conforme » dès qu'il ne refuse pas (c[\"statut\"] != RE.NOK)", NP, "_tableau_combinaisons",
     lambda vrai: _source_mutee(NP, 'case_taux(c["statut"], c["taux"], gov)', '("eta", taux_ecrit(c["taux"]), c["statut"] != RE.NOK, gov)')._tableau_combinaisons,
     ["une console comprimée (N + M non couvert)"]),
    ("la note oublie la couleur d'un taux sous son statut (« eta » au lieu de « num »)", NP, "comparaison_methodes",
     lambda vrai: (lambda r, largeur, num=None: (lambda o: dict(o, items=[("tab", i[1], [[li[0]] + [("eta", c[1], True, False) if c[0] == "num" and "%" in c[1] else c
                                                                              for c in li[1:]] for li in i[2]]) if i[0] == "tab" else i
                                                                for i in o["items"]]))(vrai(r, largeur, num))),
     ["une console comprimée (N + M non couvert)"]),
    ("le poids propre est dit avec un poids volumique recopié (77 kN/m³) et non celui du moteur", NP, "ST",
     lambda vrai: type("ST", (), {"POIDS_VOLUMIQUE": 77.0}), ["la barre neuve"]),
    # (deuxième passe, B4) le nom de la méthode du bureau : « Méthode ELS (NBN B 51-001) », « ELS » là où la place manque
    ("l'ancien nom revient en tête de la méthode (« Méthode historique »)", M, "METHODES", lambda vrai: dict(vrai, historique="Méthode historique"),
     ["la barre neuve"]),
    ("l'ancien nom court revient dans le titre des vérifications et la comparaison (« Historique »)", M, "METHODES_COURT",
     lambda vrai: dict(vrai, historique="Historique"), ["la barre neuve"]),
    ("le résumé de la barre redit « 74 % historique »", NP, "COURT", lambda vrai: dict(vrai, historique="historique"), ["la barre neuve"]),
    ("la tête de la méthode ELS répète sa norme (« Méthode ELS (NBN B 51-001) — NBN B 51-001 »)", NP, "methode",
     lambda vrai: (lambda el, k, x, largeur: (lambda o: dict(o, titre=f"{M.METHODES[k]} — {x['norme']}"))(vrai(el, k, x, largeur))), ["la barre neuve"]),
    ("les efforts imposés redisent « de service (méthode historique) »", NP, "donnees",
     lambda vrai: (lambda el, r, autre=None: (lambda o: dict(o, items=[(i[0], i[1].replace("(méthode ELS)", "(méthode historique)")) + tuple(i[2:])
                                                                      if i[0] == "pr" else i for i in o["items"]]))(vrai(el, r, autre))), POTEAU),
    ("les deux membres d'un critère échangés", NP, "comparaison", lambda vrai: (lambda v: vrai(dict(v, valeur=v.get("limite"), limite=v.get("valeur")))),
     ["une poutre continue qui ne tient pas"]),
    # ---- (lot C, C5) les trois options, la synthèse en tête, le statut imprimé, les écarts, les décisions, la torsion, le PRS
    ("la synthèse n'est plus en tête (elle passe à la fin de la planche)", NP, "planche", _synthese_a_la_fin, ["R2 · deux"]),
    ("une option imprime une méthode non choisie", NP, "imprime", lambda vrai: (lambda r, impression: r), ["PRS-C · historique"]),
    ("le statut imprimé suit une méthode non imprimée", NP, "imprime",
     lambda vrai: (lambda r, impression: dict(vrai(r, impression), statut=r["statut"])), ["PRS-C · ec3"]),
    ("la méthode non imprimée plus défavorable n'est plus dite", NP, "autre_methode", lambda vrai: (lambda r, rv: None), ["PRS-C · ec3"]),
    ("les écarts avec les classeurs du bureau ne sont plus imprimés", NP, "ecarts",
     lambda vrai: (lambda r, num=None: {"titre": "Remarques", "items": [], "verdicts": []}), ["R2 · historique"]),
    ("un choix du bureau présenté comme une correction (« f_y nominale »)", NP, "ecarts",
     lambda vrai: (lambda r, num=None: (lambda o: dict(o, items=o["items"] + [NP.pr("F_y nominale, comme les classeurs du bureau.")]))(vrai(r, num))),
     ["écarts (6.7)"]),
    ("la torsion non couverte n'est plus dite", NP, "communes", lambda vrai: (lambda el, r: [h for h in vrai(el, r) if "orsion" not in h]), ["HEB 200 · ec3"]),
    ("les décisions du bureau perdent leurs variantes (seul le seuil n° 7 reste)", NP, "decisions",
     lambda vrai: (lambda el, r, num: (lambda o: dict(o, items=o["items"][-1:]))(vrai(el, r, num))), ["R2 · deux"]),
    ("« section soudée équivalente » cochée n'est plus dite par la décision n° 10", NP, "decisions",
     lambda vrai: (lambda el, r, num: (lambda o: dict(o, items=[i[:1] + tuple(t.replace(" « soudée équivalente »", "") for t in i[1:]) if i[0] == "pr" else i
                                                         for i in o["items"]]))(vrai(el, r, num))), ["PRS « soudée équivalente »"]),
    ("le PRS n'est plus au panneau (ses propriétés du catalogue le plus proche)", NP, "blocs_panneau",
     lambda vrai: (lambda el: vrai(dict(el, prs=None, profil="IPE 750x147"))), ["PRS-C au panneau"]),
    ("le tableau avant/après du raidisseur n'est plus imprimé", NP, "avant_apres", lambda vrai: (lambda L, largeur: []),
     ["une poutre de reprise et son raidisseur"]),
    ("un message déjà dit est répété (la remarque du type (c) à chaque appui)", NP, "redit", lambda vrai: (lambda t, dits: False), ["R2 · deux"]),
    # ---- (lot C, C5) la garde des textes : un texte de 121 caractères rougit (dans la note, à l'écran) ; une redite dans un bloc aussi
    ("une hypothèse commune de 121 caractères dans la note", NP, "communes", lambda vrai: (lambda el, r: vrai(el, r) + ["x" * 120 + "."]), ["garde des textes"]),
    ("une correction des classeurs de 121 caractères à l'écran (avec son préfixe)", HD, "CORRECTIONS",
     lambda vrai: dict(vrai, axe_faible="x" * (LIGNE_MAX + 1 - len("Correction du classeur : "))), ["garde des textes"]),
    ("une ligne répétée dans un bloc de la note", NP, "communes", lambda vrai: (lambda el, r: vrai(el, r) + vrai(el, r)[-1:]), ["garde des textes"]),
)


if __name__ == "__main__":
    sys.exit(main())
