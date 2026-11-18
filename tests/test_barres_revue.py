# -*- coding: utf-8 -*-
"""Barres acier — la REVUE FINALE du lot C (bloc C8) : le script tools/comparaison_barres.py, son rapport daté, les listes
restantes et la copie nommée du § 5.8. docs/validation/barres_revue_finale_*.md ; docs/acier/BARRES_ACIER.md § 5.8 et § 10.

  1. Le rapport daté EXISTE (comme tests/test_pied_poteau_validation.py) — il n'est JAMAIS comparé à l'égalité : une
     mesure datée ne se réécrit pas, et le prochain changement du moteur la ferait rougir pour toujours.
  2. Le script, REJOUÉ : chaque cas a ses deux méthodes ou un statut NOMMÉ, jamais rien ; chaque valeur du brief à sa
     tolérance (1 % au plus, règle 7) ; la colonne ELS porte la méthode ELS et la colonne EC3 l'Eurocode 3 ; une phrase
     par écart de plus de 10 points ; la donnée du rapport (N_Ed = 1,5 × P) déclarée en tête.
  3. Les listes, chacune tirée de SA source : À CONFIRMER = parametres.a_confirmer() (npl_gamma seul, gardé vivant par
     γ_M0 = 1,00 et γ_M1 = 1,10) ; DÉCISION BUREAU = les entrées « decision » de la table (n° 1, 2, 3, 4, 9, 10), n° 7 dans
     sa source du site ; aucun défaut de décision écrit en dur hors de la table ; tout fichier qui dit « à vérifier » est classé.
  4. Le § 5.8 de la doc = acier/barre/parametres.py (copie NOMMÉE, réécrite par « --parametres »).
  5. MUTATIONS du script : chaque altération plausible (colonnes ELS et EC3 échangées, un sujet perdu…) fait rougir.

    /usr/bin/python3.12 -B tests/test_barres_revue.py
"""
import glob
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tools"))

import comparaison_barres as T0  # noqa: E402
from acier.barre import parametres as PA  # noqa: E402
from acier.garde_corps import membres as ME  # noqa: E402

OK, KO = [], []
SCRIPT = os.path.join(RACINE, "tools", "comparaison_barres.py")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info and not cond else ""))


def proche(a, b, tol=0.01):
    return a is not None and b is not None and abs(a / b - 1.0) <= tol


def _nombre(cellule):
    m = re.search(r"(-?\d[\d  ]*,\d+|-?\d+) %", cellule)
    return None if not m else float(m.group(1).replace(" ", "").replace(" ", "").replace(",", ".")) / 100.0


# ------------------------------------------------------------------ 2. le script rejoué (les contrôles que les mutations rejouent)
def controles(T):
    """[(nom, ok, info)] — ``T`` : le module du script (celui du dépôt, ou une copie mutée)."""
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    cs = T.preparer()
    noms = {x["nom"] for x in cs}
    exiges = ("HEB 200", "HEA 120 Chancel", "CHS 168,3 × 6 (A du classeur)", "CHS 168,3 × 6 (A = 3 059 mm²)", "REALEX Ø610 × 12,5",
              "FLAMBECBD (HEA 160)", "FLAMBEBAD (IPE 270)", "p. 6, caisson soudé", "p. 8, cornière L 60 × 60 × 8", "p. 14, HEA 220",
              "p. 23, poutre BC HEA 200", "p. 25, colonne AB HEA 300", "PRS-A, oxycoupé, 10 m", "PRS-B, monosymétrique, 10 m",
              "PRS-C, 8 maintiens", "PRS-C, maintiens aux tiers", "PRS-C, 18 maintiens", "S2, poteau HEB 300", "S3, continue IPE 360, 2 × 6 m")
    c("les cas de la mission sont là : 11.1, la doc p. 6 à 25, 11.1 bis, R2/R3 aux deux niveaux, PRS-A, B, C (8, tiers, 18), S1 à S3",
      all(n in noms for n in exiges) and sum(1 for x in cs if x["groupe"] == T.G3) >= 8 and sum(1 for x in cs if x["nom"].startswith("R2 et R3")) == 2
      and sum(1 for x in cs if x["nom"].startswith("S1")) == 2 and {x["groupe"] for x in cs} == set(T.GROUPES), sorted(set(exiges) - noms))
    vides = [(x["nom"], k) for x in cs for k in ("historique", "ec3") if x[k] is None and not x["absent"].get(k)]
    c("chaque cas a ses DEUX méthodes, ou un statut NOMMÉ (non couvert, non calculé, sans objet, fonction seule) — jamais rien", not vides, vides)
    sans = [x["nom"] for x in cs if x["capacite"] is None and not T.lignes(x)]
    c("chaque cas chargé a ses lignes de sujets (les sept sujets du moteur, comparaison.SUJETS)", not sans, sans)
    hors = T.hors_tolerance(cs)
    tol = [(x["nom"], b["quoi"]) for x in cs for b in x["brief"] if not 0.0 < b["tol"] <= 0.01]
    n_brief = sum(len(x["brief"]) for x in cs)
    c(f"chaque valeur du brief à sa tolérance — 1 % au plus (règle 7), plus serrée pour une non-régression ({n_brief} valeurs)",
      not hors and not tol and n_brief >= 80, (hors, tol))
    r4 = [b for b in T.R4_LIGNES if not proche(b["obtenu"], b["attendu"], b["tol"])]
    c("R4 (EN 1993-1-5) : les six F_Rd du brief à 1 % (283,2 ; 387,8 ; 425,1 ; 248,9 ; 199,82 ; 335,9 kN)", len(T.R4_LIGNES) == 6 and not r4, r4)
    heb = next(x for x in cs if x["nom"] == "HEB 200")
    li = next((li for li in T.lignes(heb) if li["sujet"] == "Flambement, axe faible"), None)
    c("HEB 200, flambement z : colonne ELS = la méthode ELS (1,118 du brief, à 1 %), colonne EC3 = l'Eurocode 3 (649,08/570,1 = 1,139) — "
      "leurs vérifications H-N-z et E-Nb-z, écart en points",
      li is not None and proche(li["historique"]["taux"], 1.118) and proche(li["ec3"]["taux"], 1.139) and li["historique"]["gouvernant"] == "H-N-z"
      and li["ec3"]["gouvernant"] == "E-Nb-z" and proche(li["ecart"], 100 * (li["ec3"]["taux"] - li["historique"]["taux"]), 1e-9), li)
    mal = [(x["nom"], li["sujet"]) for x in cs for li in T.lignes(x)
           if (li["historique"] and not li["historique"]["gouvernant"].startswith("H-")) or (li["ec3"] and not li["ec3"]["gouvernant"].startswith("E-"))]
    c("chaque ligne : la colonne ELS ne porte que des vérifications H-…, la colonne EC3 que des E-… (aucune colonne échangée)", not mal, mal[:4])
    texte = T.rapport(cs, "18/11/2026")
    ligne = next((li for li in texte.splitlines() if li.startswith("| HEB 200 | Flambement, axe faible |")), "")
    cel = [x.strip() for x in ligne.strip().strip("|").split("|")]
    c("le rapport ÉCRIT la ligne du HEB 200 dans le bon ordre : ELS ≈ 111,8 %, Eurocode 3 ≈ 113,9 % (à 1 %), écart, puis « H-N-z / E-Nb-z »",
      len(cel) == 6 and proche(_nombre(cel[2]), 1.118) and proche(_nombre(cel[3]), 1.139) and cel[5] == "H-N-z / E-Nb-z", cel)
    ecarts = [(x, li) for x in cs for li in T.lignes(x) if li["ecart"] is not None and abs(li["ecart"]) > T.SEUIL_ECART]
    c(f"une phrase par écart de plus de {T.SEUIL_ECART:g} points, dans le rapport ({len(ecarts)} écarts)",
      ecarts and all(T.phrase_ecart(x, li) and T.phrase_ecart(x, li) in texte for x, li in ecarts)
      and all(T.phrase_ecart(x, li) is None for x in cs for li in T.lignes(x) if li["ecart"] is None or abs(li["ecart"]) <= T.SEUIL_ECART))
    pa = next(x for x in cs if x["nom"] == "PRS-A, oxycoupé, 10 m")["capacite"]
    c("PRS-A oxycoupé, sans charge : capacité ELS 90,31 × 5 534,1·10³ = 499,8 kN·m de service ; × 1,5 = 749,7 contre M_b,Rd 797,39 kN·m",
      proche(pa["M_service"], 499.8) and proche(pa["M_els_s"], 749.7) and proche(pa["M_b_Rd"], 797.39) and proche(pa["rapport"], 749.7 / 797.39), pa)
    c("la DONNÉE du rapport (N_Ed = 1,5 × P) est déclarée en tête, une fois, et dite donnée (pas une règle du module)",
      texte.count(T.DONNEE) == 1 and texte.index(T.DONNEE) < texte.index("## 0.") and "pas une règle du module" in T.DONNEE and T.S == 1.5)
    q = {x["cas"]: x for x in T.QUALIF}
    l17 = next(x for k, x in q.items() if k.startswith("L17a IPE 360"))
    c("qualification : L1 à L16 et L17b, L17c attendus ; L17a reçu (221,1) ; non-régression 223,30 kN·m (catalogue, G = 80 769) à 0,1 %",
      sum(1 for x in T.QUALIF if x["exterieur"] is None) == 18 and proche(l17["site"], 223.30, 0.001) and l17["exterieur"] == 221.1
      and all(proche(x["site"], x["attendu"], 0.001) for x in T.QUALIF), [(x["cas"], x["site"], x["attendu"]) for x in T.QUALIF if not proche(x["site"], x["attendu"], 0.001)])
    c("qualification : ECCS 17 à 20 et 22, JRC « bloqué — données à fournir » ; ECCS 21 dans le rapport",
      all(f"| ECCS {n} (" in texte for n, _ in T.ECCS) and "JRC EUR 22898" in texte and len(T.ECCS21) == 14)
    c("À CONFIRMER, tiré de SA source : parametres.a_confirmer() = npl_gamma seul ; GARDÉ VIVANT : avec γ_M0 = 1,00 et γ_M1 = 1,10, le moteur "
      "le rend dans ses hypothèses", [p["cle"] for p in PA.a_confirmer()] == ["npl_gamma"] and [p["cle"] for p in T.GAMMA] == ["npl_gamma"]
      and "`npl_gamma`" in texte)
    c("DÉCISION BUREAU, tirées de la table : n° 1, 2, 3, 4, 9, 10 (statut « decision ») ; n° 7 dans sa source du site (membres.SEUIL_ATTENTION, "
      "95 %) ; n° 5 tranché ; n° 6 et 8 hors code — toutes dans le rapport",
      [d["point"] for d in PA.decisions()] == [1, 2, 3, 4, 9, 10] and all(f"`parametres.{d['cle']}`" in texte for d in PA.decisions())
      and ME.SEUIL_ATTENTION == 0.95 and "membres.SEUIL_ATTENTION" in texte and all(f"| {n} |" in texte for n in (5, 6, 7, 8)))
    vus = T.a_verifier_du_code()
    c("À VÉRIFIER : chaque fichier du moteur et de l'écran du profil qui dit « à vérifier » (sans accents ni casse) est CLASSÉ dans la "
      "liste du script, et aucun classement n'est mort", vus and set(vus) == set(T.FICHIERS_A_VERIFIER),
      (sorted(set(vus) - set(T.FICHIERS_A_VERIFIER)), sorted(set(T.FICHIERS_A_VERIFIER) - set(vus))))
    absents = [b for _, b, _, _ in T.A_VERIFIER if not os.path.exists(os.path.join(RACINE, b))]
    c("À VÉRIFIER : chaque hypothèse nomme un fichier qui existe, et dit son effet et sa réponse", not absents and all(all(x) for x in T.A_VERIFIER), absents)
    c("questions au bureau Q1 à Q14 rangées en trois groupes (tranchées, ouvertes, nouvelles), chacune une fois",
      len(T.QUESTIONS) == 3 and sorted(int(m) for _, qs in T.QUESTIONS for q_ in qs for m in re.findall(r"\bQ(\d+)", q_))
      == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    c("le défaut non conservatif trouvé (débord gauche : charges bornées à Σ portées), corrigé à l'intégration, est en tête du rapport, avant le § 1",
      texte.index("Défaut non conservatif") < texte.index("## 1."))
    return C


# ------------------------------------------------------------------ 4. la garde : aucun défaut de décision en dur
FICHIERS_DECISION = sorted(set(glob.glob("acier/barre/*.py")) - {"acier/barre/parametres.py"}) + sorted(
    glob.glob("api_web/profil*.py")) + ["api_web/note_profil.py", "api_web/dessin_barre.py"]


def defauts_en_dur(textes):
    """[(fichier, motif)] : un défaut de DÉCISION BUREAU écrit hors de la table — un repli « .get(clé, …) », une clé posée
    dans un dictionnaire ou en argument, ou le seuil n° 7 recopié (0.95) au lieu d'être lu dans sa source du site."""
    out = []
    for f, t in textes.items():
        for d in PA.decisions():
            k = d["cle"]
            for motif in (rf"\.get\(\s*[\"']{k}[\"']\s*,", rf"[\"']{k}[\"']\s*:\s*\S", rf"\b{k}\s*=\s*\S"):
                if re.search(motif, t):
                    out.append((f, k))
        if re.search(r"(?<![\d.])0\.95(?![\d])", t):
            out.append((f, "0.95"))
    return out


def lire(fichiers):
    return {f: open(f, encoding="utf-8").read() for f in fichiers}


# ------------------------------------------------------------------ 5. mutations du script
def muter(ancien, nouveau):
    """Les contrôles qui rougissent quand le texte ``ancien`` du script devient ``nouveau`` ; None s'il n'y est pas une fois."""
    src = open(SCRIPT, encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    mod = types.ModuleType("comparaison_barres_mute")
    mod.__file__ = SCRIPT
    try:
        exec(compile(src.replace(ancien, nouveau), SCRIPT, "exec"), mod.__dict__)
        return [n for n, ok, _ in controles(mod) if not ok]
    except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
        return [f"exception {type(e).__name__}: {e}"]


MUTATIONS = [
    ("colonnes ELS et EC3 échangées (les données)", '(("historique", h, c["historique"]), ("ec3", e, c["ec3"]),',
     '(("historique", e, c["ec3"]), ("ec3", h, c["historique"]),'),
    ("colonnes ELS et EC3 échangées (l'écriture du tableau)", "| {sujet} | {els} | {ec3} |", "| {sujet} | {ec3} | {els} |"),
    ("un sujet disparaît (le flambement de l'axe faible)", "    for nom, h, e, a in CP.SUJETS:\n", "    for nom, h, e, a in CP.SUJETS[1:]:\n"),
    ("la donnée du rapport : 1,35 au lieu de 1,5", "S = 1.5                          #", "S = 1.35                         #"),
    ("la tolérance du brief : 10 % au lieu de 1 %", "TOL_BRIEF = 0.01", "TOL_BRIEF = 0.10"),
    ("une méthode sans résultat ni motif (REALEX : la nuance absente tue)",
     'e, me = None, f"non couvert : f_y = {fr(fy, 0)} N/mm² n\'est pas une nuance du module (S235 à S460)"', "e, me = None, None"),
    ("la phrase d'un écart de plus de 10 points se tait", "    if li[\"ecart\"] is None or abs(li[\"ecart\"]) <= SEUIL_ECART:\n        return None",
     "    if True:\n        return None"),
    ("la capacité ELS sans le facteur s", '"M_els_s": None if Ms is None else S * Ms', '"M_els_s": Ms'),
    # (intégration du lot C, le harnais suit : le défaut est corrigé, la tête du rapport le dit — trouvé, puis corrigé)
    ("le défaut non conservatif quitte la tête du rapport", '    w("- **Défaut non conservatif trouvé par la revue, corrigé', '    w("- **Défaut trouvé par la revue, corrigé'),
]


def main():
    print("== 1. le rapport daté")
    dates = sorted(glob.glob(os.path.join("docs", "validation", "barres_revue_finale_*.md")))
    chk("le rapport daté de la revue finale EXISTE (docs/validation/barres_revue_finale_AAAA-MM-JJ.md) — jamais comparé à l'égalité",
        bool(dates) and all(re.fullmatch(r"barres_revue_finale_\d{4}-\d\d-\d\d\.md", os.path.basename(f)) for f in dates), dates)
    if dates:
        t = open(dates[0], encoding="utf-8").read()
        chk("il dit sa date, la donnée du rapport en tête, et ses cinq parties (cas, écarts avec les classeurs, qualification, listes, avant/après)",
            t.startswith("# ") and "Mesure datée du " in t and "N_Ed = 1,5 × P" in t[:1500]
            and all(f"## {n}." in t for n in (1, 2, 3, 4, 5)), [n for n in (1, 2, 3, 4, 5) if f"## {n}." not in t])

    print("== 2. le script rejoué")
    base = controles(T0)
    for nom, ok, info in base:
        chk(nom, ok, info)

    print("== 3. le § 5.8 de la doc : la copie NOMMÉE de acier/barre/parametres.py")
    doc = open(T0.DOC, encoding="utf-8").read()
    bloc = T0.bloc_58(doc)
    chk("le § 5.8 de docs/acier/BARRES_ACIER.md est la table de parametres.py, ligne pour ligne (sinon : tools/comparaison_barres.py --parametres)",
        bloc == T0.table_parametres(), None if bloc is None else [x for x in T0.table_parametres() if x not in bloc][:3])
    faux = doc.replace("| `gamma_G_sup` | γ_G,sup | 1,35 |", "| `gamma_G_sup` | γ_G,sup | 1,5 |")
    chk("la garde du § 5.8 voit une valeur changée dans la doc (γ_G,sup 1,5 au lieu de 1,35) et une clé de la table absente de la doc",
        faux != doc and T0.bloc_58(faux) != T0.table_parametres()
        and T0.bloc_58(doc.replace("| `xi` |", "| `xi_` |")) != T0.table_parametres())
    chk("chaque clé de la table est au § 5.8, et chaque DÉCISION y dit son numéro et sa variante",
        bloc is not None and all(any(li.startswith(f"| `{k}` |") for li in bloc) for k in PA.PARAMETRES)
        and all(any(li.startswith(f"| `{d['cle']}` |") and f"décision n° {d['point']}" in li and "variante" in li for li in bloc) for d in PA.decisions()))

    print("== 4. aucun défaut de décision écrit en dur hors de la table")
    textes = lire(FICHIERS_DECISION)
    trouves = defauts_en_dur(textes)
    chk(f"aucun défaut de DÉCISION BUREAU écrit en dur hors de parametres.py ({len(textes)} fichiers du moteur et de l'écran) : ni repli "
        "« .get(clé, …) », ni clé posée, ni le seuil n° 7 recopié", not trouves, trouves)
    lus = [d["cle"] for d in PA.decisions() if not any(re.search(rf"[\"']{d['cle']}[\"']", t) for t in textes.values())]
    chk("chaque décision de la table est LUE par un moteur ou l'écran (sa clé y est nommée)", not lus, lus)
    for nom, f, a, b in (("un repli .get(…, « elastique »)", "acier/barre/ame.py", "P['eta1_ame']", "P.get('eta1_ame', 'elastique')"),
                         ("le seuil 0.95 recopié", "api_web/profil_resultat.py", "\n", "\nSEUIL = 0.95\n")):
        mute = dict(textes)
        mute[f] = mute[f].replace(a, b, 1)
        chk(f"mutation de la garde « {nom} » ({f}) : elle rougit", mute[f] != textes[f] and defauts_en_dur(mute), nom)

    print("== 5. mutations du script")
    for nom, ancien, nouveau in MUTATIONS:
        rouges = muter(ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans le script, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, le script du dépôt repasse tous ses contrôles", all(ok for _, ok, _ in controles(T0)))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
