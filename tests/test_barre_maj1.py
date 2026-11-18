# -*- coding: utf-8 -*-
"""Barres en acier — LA MISE À JOUR 1 (brief du 18/11/2026), sur son cas de contrôle « PRS-C4 » : PRS 300×30 / 1420×25 /
500×50, S235, deux appuis, L = 28,00 m, poids propre + Q = 20 kN/m sur la semelle haute, 4 maintiens de la semelle haute
à 5,60 / 11,20 / 16,80 / 22,40 m.

  Étape 1 (§ 1.5, tolérance 0,1 %) :
    1. les quatre réglages des DÉCISIONS BUREAU n° 11 (âme de classe 3) et n° 12 (M_cr retenu) : W_y, M_cr, λ̄_LT, χ_LT,
       M_b,Rd, taux — « élastique + solveur » redonne exactement le résultat d'avant ;
    2. W_pl,eff : ε_w, b, z_p, le trou, la flexion (49,0 %) ; chaque formule imprimée redonne le chiffre du moteur ;
    3. les deux lignes du M_cr, à l'écran et dans la note ;
    4. la même barre sans maintien : la formule prend le diagramme « réparti », z_g pondéré, et ne dépasse pas le solveur
       de plus de 2 % ;
    5. le garde-fou : une formule forcée au-dessus du solveur rend le solveur et sa ligne ;
    6. la méthode affichée : « ELS » sans la carte Eurocode 3, et inversement ; plus le bloc des grosses pastilles.

  Étape 2 (§ 2.6, les diagrammes des contraintes) :
    7. la vue « Contraintes » en ELS : trois volets — 102,4 et 66,3, σ_adm,d = 63,1 ; 10,0, σ_adm/√3 = 90,5 ; 17,4, σ_adm = 156,7 ;
    8. la note ELS : la garde et 3 pages ; tous les textes d'avant (seuls les rappels « (suite) » changent de place), les blocs dans
       le même ordre ; un dessin dans « ELS · déversement », un dans « ELS · contrainte équivalente » ;
    9. l'Eurocode 3 en variante « élastique » : le volet « Flexion » écrit 150,3 et 97,3, f_y/γ_M0 = 215,0 ; en classe 2 efficace
       (le défaut) : pas de dessin ;
   10. un IPE 300 et un RHS 200×100×8 : le dessin sort ; un tube rond : pas de vue.

    /usr/bin/python3.12 -B tests/test_barre_maj1.py
"""
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from collections import Counter  # noqa: E402

from acier.barre import ec3_deversement as DV  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from api_web import dessin_profil as DP  # noqa: E402
from api_web import note_profil as NP  # noqa: E402
from api_web import profil as P  # noqa: E402

OK, KO = [], []
MAINTIENS = (5.6, 11.2, 16.8, 22.4)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK  " if cond else "KO  ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def proche(a, b, tol=1e-3):
    return a is not None and abs(a - b) <= tol * abs(b)


def barre(maintiens=MAINTIENS, params=None):
    b = P.barre_neuve("B1")
    b.update(profil="PRS", prs={"b1": 300.0, "t1": 30.0, "hw": 1420.0, "tw": 25.0, "b2": 500.0, "t2": 50.0, "a": 10.0, "bords": "oxycoupes",
                                "nom": None},
             nuance="S235", travees=[28.0], poids_propre=True, niveau="haut",
             maintiens=[{"x": x, "genre": "haut", "neglige": False} for x in maintiens])
    b["cas"] = [dict(b["cas"][1], nom="Q", charges=[{"type": "q", "v": 20.0, "a": 0.0, "b": 28.0, "niveau": None}])]
    if params:
        b["parametres"] = dict(params)
    return b


def ec3(b, impression="deux"):
    etat = P.normaliser(P.etat_de([b], impression=impression))
    return P.calculer(etat["elements"][0])["methodes"]["ec3"]


def verif(e, ident):
    return next((v for v in e["verifications"] if v["id"] == ident), None)


def blanc(t):
    return re.sub(r"[\s  ]+", " ", t)


def main():
    print("== 1. les quatre réglages (§ 1.5)")
    TABLE = (({}, 34736.0, 8704.5, 0.9263, 0.5062, 3780.1, 96.9),
             ({"mcr_retenu": "solveur"}, 34736.0, 10716.7, 0.8348, 0.5587, 4172.2, 87.8),
             ({"ame_classe_3": "elastique"}, 24367.7, 8704.5, 0.7758, 0.5947, 3115.6, 117.6),
             ({"ame_classe_3": "elastique", "mcr_retenu": "solveur"}, 24367.7, 10716.7, 0.6992, 0.6437, 3372.2, 108.6))
    for prm, W, Mcr, lb, chi, MbRd, taux in TABLE:
        e = ec3(barre(params=prm))
        D, v = e["grandeurs"]["deversement"], verif(e, "E-Mb")
        nom = " + ".join(f"{k} = {x}" for k, x in prm.items()) or "défauts (classe 2 efficace + formule)"
        chk(f"{nom} : W_y {W}, M_cr {Mcr}, λ̄_LT {lb}, χ_LT {chi}, M_b,Rd {MbRd}, taux {taux} %",
            proche(D["W_y"] / 1e3, W) and proche(D["M_cr"] / 1e6, Mcr) and proche(D["lambda_LT"], lb) and proche(D["chi_LT"], chi)
            and proche(D["M_b_Rd"] / 1e6, MbRd) and proche(v["taux"] * 100, taux) and proche(v["valeur"], 3662.7),
            (D["W_y"] / 1e3, D["M_cr"] / 1e6, D["lambda_LT"], D["chi_LT"], D["M_b_Rd"] / 1e6, v["taux"]))

    print("== 2. W_pl,eff (§ 5.5.2(11), § 6.2.2.4)")
    e = ec3(barre())
    G, v = e["grandeurs"], verif(e, "E-My")
    k = G["classe_2_efficace"] or {}
    chk("ε_w = 1,02198 ; b = 510,99 mm ; z_p = 1 038,02 mm ; trou de 16,04 mm",
        proche(k.get("eps_w"), 1.02198, 1e-5) and proche(k.get("b"), 510.99, 1e-5) and proche(k.get("z_p"), 1038.02, 1e-5)
        and proche(k.get("trou"), 16.04, 1e-3), k)
    chk("flexion : 3 662,7 / 7 468,2 = 49,0 % ; classe affichée « 3 → classe 2 efficace »",
        proche(v["valeur"], 3662.7) and proche(v["limite"], 7468.2) and proche(v["taux"] * 100, 49.0)
        and G["classe_affichee"] == "3 → classe 2 efficace" and any("classe 3 → classe 2 efficace" in h for h in e["hypotheses"]),
        (v["valeur"], v["limite"], G.get("classe_affichee")))
    ecarts = [(b["cle"], b["resultat"], FO.evaluer(b)) for b in e["calcul"] if abs(FO.evaluer(b) - b["resultat"]) > 1e-9 * max(1.0, abs(b["resultat"]))]
    chk("chaque formule imprimée (W_pl,eff, z_p, b, M_cr de la formule…) redonne le chiffre du moteur",
        not ecarts and {"E-My/W_pl_eff", "E-My/z_p_eff2", "E-My/b_eff2", "E-Mb/M_cr_E1"} <= {b["cle"] for b in e["calcul"]}, ecarts)
    hors = ec3(barre(params={"ame_classe_3": "elastique"}))
    chk("variante « élastique » : W_el, classe 3, aucune formule de W_pl,eff",
        hors["grandeurs"]["classe_2_efficace"] is None and hors["grandeurs"]["classe_affichee"] == "3"
        and not any("eff2" in b["cle"] for b in hors["calcul"]), hors["grandeurs"].get("classe_affichee"))

    print("== 3. les deux lignes du M_cr")
    m = [blanc(t) for t in verif(e, "E-Mb")["messages"]]
    chk("« M_cr retenu : formule, tronçon 11,20 à 16,80 m, moment constant = 8 704,5 kN·m » puis « Solveur (poutre entière) : "
        "10 716,7 kN·m (+23 %) »",
        m[:2] == ["M_cr retenu : formule, tronçon 11,20 à 16,80 m, moment constant = 8 704,5 kN·m.",
                  "Solveur (poutre entière) : 10 716,7 kN·m (+23 %)."], m[:3])
    s = [blanc(t) for t in verif(ec3(barre(params={"mcr_retenu": "solveur"})), "E-Mb")["messages"]]
    chk("réglage « solveur » : « M_cr retenu : solveur… » puis la formule et son écart",
        s[:2] == ["M_cr retenu : solveur (poutre entière) = 10 716,7 kN·m.",
                  "Formule, tronçon 11,20 à 16,80 m, moment constant : 8 704,5 kN·m (−19 %)."], s[:3])

    print("== 4. sans maintien")
    e0 = ec3(barre(maintiens=()))
    D0 = e0["grandeurs"]["deversement"]
    F = D0["mcr"]["formule"]
    t0 = F["troncons"][0] if F and F["troncons"] else {}
    chk("la formule : diagramme « réparti », z_g pondéré (entre le centre de gravité et la semelle haute), retenue",
        D0["mcr"]["retenu"] == "annexe_E" and t0.get("diagramme") == "reparti" and t0.get("C1") == 1.12 and t0.get("z_g") is not None, t0)
    chk("elle ne dépasse pas le solveur de plus de 2 %", t0 and t0["M_cr"] <= 1.02 * D0["mcr"]["solveur"], (t0.get("M_cr"), D0["mcr"]["solveur"]))
    cles0 = [b["cle"] for b in e0["calcul"]]
    chk("la formule retenue n'a qu'UN bloc « M_cr » de l'annexe E (l'ancien contrôle (E.1) du PRS ne le réécrit pas)",
        cles0.count("E-Mb/M_cr_E1") == 1, cles0)

    print("== 5. le garde-fou")
    garde = DV.GARDE_FORMULE
    DV.GARDE_FORMULE = 0.5                                    # (forcé : la formule passe « au-dessus » du solveur)
    try:
        P._RESULTATS.clear()                                  # (le calcul de l'écran est gardé en mémoire : le refaire)
        ef = ec3(barre())
    finally:
        DV.GARDE_FORMULE = garde
        P._RESULTATS.clear()
    Df, mf = ef["grandeurs"]["deversement"], [blanc(t) for t in verif(ef, "E-Mb")["messages"]]
    chk("une formule au-dessus du solveur : le solveur est retenu, avec sa ligne",
        Df["mcr"]["retenu"] == "solveur" and proche(Df["M_cr"] / 1e6, 10716.7) and mf[0] == "Formule au-dessus du solveur : solveur retenu.", mf[:2])

    print("== 6. la méthode affichée (§ 1.4)")
    for imp, avec, sans in (("historique", "ELS", "Eurocode 3"), ("ec3", "Eurocode 3", "ELS"), ("deux", "ELS", None)):
        etat = P.normaliser(P.etat_de([barre()], impression=imp))
        html = P.resultat(etat)["resultat"]
        titres = re.findall(r'class="et-bloc[^"]*".*?<h3[^>]*>(.*?)</h3>', html)
        chk(f"méthode affichée « {imp} » : la carte {avec}" + (f", jamais celle {'de l' if sans.startswith('E') else 'des'} {sans}" if sans else
                                                                ", les deux cartes") + " ; plus de bloc des grosses pastilles",
            avec in html and (sans is None or sans not in html) and (sans is not None or "Eurocode 3" in html) and "pf-verdicts" not in html,
            titres[:4])
    etat = P.normaliser(P.etat_de([barre()], impression="ec3"))
    chk("méthode affichée « Eurocode 3 » : le taux de la barre est celui de l'Eurocode 3 (97 %), pas celui de l'ELS",
        str(P.resultat(etat)["resume"]["taux"]) == "97", P.resultat(etat)["resume"])

    etape_2()
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


def textes_svg(svg):
    """Ce qu'un dessin ÉCRIT : chaque texte, ses indices recollés (« σ_adm,d = 63,1 » se lit « σadm,d = 63,1 »)."""
    return [blanc(re.sub(r"<[^>]+>", "", m)).replace("\u200b", "").strip() for m in re.findall(r"<text[^>]*>(.*?)</text>", svg, re.S)]


def vue(b, impression):
    etat = P.normaliser(P.etat_de([b], impression=impression))
    return {v["cle"]: v for v in P.resultat(etat)["vues"]}


def etape_2():
    print("== 7. la vue « Contraintes » en ELS (§ 2.4, § 2.6)")
    vs = vue(barre(), "historique")
    c = vs.get("contraintes")
    ts = textes_svg(c["svg"]) if c else []
    chk("après « Section », la vue « Contraintes dans la section » (« Contraintes ») : trois volets — Flexion 102,4 et 66,3, σ_adm,d = 63,1 ; "
        "Cisaillement 10,0, σ_adm/√3 = 90,5 ; Contrainte équivalente 17,4, σ_adm = 156,7",
        c is not None and list(vs).index("contraintes") == list(vs).index("section") + 1 and c["titre"] == "Contraintes dans la section"
        and c["court"] == "Contraintes" and all(x in ts for x in ("Flexion", "Cisaillement", "Contrainte équivalente", "102,4", "66,3", "σadm,d = 63,1",
                                                                  "10,0", "σadm/√3 = 90,5", "17,4", "σadm = 156,7")), ts)
    print("== 8. la note ELS (§ 2.5, § 2.6)")
    etat = P.normaliser(P.etat_de([barre()], impression="historique"))
    el = etat["elements"][0]
    r = P.calculer(el)
    sec = NP.planche(el, r, NP.Planche(), "historique")
    dessins = {b_["titre"]: [it_ for it_ in b_["items"] if it_[0] == "d"] for b_ in sec["verifs"]}
    avec = {k_: len(v_) for k_, v_ in dessins.items() if v_}
    blocs = {b_["titre"]: b_ for b_ in sec["verifs"]}
    rang = {k_: [it_[0] for it_ in blocs[k_]["items"]] for k_ in avec}
    chk("un dessin dans « ELS · déversement », un dans « ELS · contrainte équivalente », aucun ailleurs ; chacun SOUS la dernière formule, "
        "avant les remarques et le verdict",
        avec == {"ELS · déversement": 1, "ELS · contrainte équivalente": 1}
        and all(r_.index("d") == max(i_ for i_, k_ in enumerate(r_) if k_ == "e") + 1 for r_ in rang.values()), (avec, rang))
    ts_eq = textes_svg(dessins["ELS · contrainte équivalente"][0][1]) if dessins.get("ELS · contrainte équivalente") else []
    ts_dv = textes_svg(dessins["ELS · déversement"][0][1]) if dessins.get("ELS · déversement") else []
    chk("… dans la note, un volet n'écrit pas son abscisse (« x = … ») ; la contrainte équivalente : « Cisaillement » et « Équivalente », "
        "côte à côte", not any(x_.startswith("x = ") for x_ in ts_eq + ts_dv) and "Cisaillement" in ts_eq and "Équivalente" in ts_eq
        and "Flexion" in ts_dv and "102,4" in ts_dv, (ts_dv, ts_eq))
    from importlib import util as U_
    spec = U_.spec_from_file_location("tn_", os.path.join(RACINE, "tests", "test_note_profil.py"))
    tn = U_.module_from_spec(spec)
    spec.loader.exec_module(tn)
    octets, pages, _f = tn.faire(etat)
    vrai = DP.volets_du_bloc
    DP.volets_du_bloc = lambda *a_, **k_: []
    try:
        P._RESULTATS.clear()
        o0, p0, _f0 = tn.faire(etat)
    finally:
        DP.volets_du_bloc = vrai
        P._RESULTATS.clear()

    def lus(oc):
        """Les mots de la note, hors rappels « (suite) » en tête de colonne (ils peuvent changer de place)."""
        S_ = NP.Planche()
        zx = S_.margin + S_.coupe_w + S_.gap
        lignes = {}
        for page, w, h, ws in tn.mots(oc):
            for x0, y0, _x1, _y1, mot in ws:
                lignes.setdefault((page, x0 >= zx - 2 and w > h, round(y0)), []).append((x0, mot))
        out = []
        for k_, ws in lignes.items():
            mots = [m_ for _x, m_ in sorted(ws)]
            if not (k_[1] and mots and mots[-1] == "(suite)"):
                out += mots
        return Counter(out)
    ca, cb = lus(o0), lus(octets)
    perdus = {w: n for w, n in ca.items() if cb[w] < n}

    def ordre(ps):
        t_ = " ".join(ps[1:])
        vus = []
        for m_ in re.finditer(r"(?<![\d,])(\d{1,2}) (La barre et ses|Écarts avec|Méthode ELS|ELS · [a-zé]+|Contrôles complémentaires|Appui [AB] : [a-zé]+)", t_):
            if not vus or vus[-1] != m_.groups():
                vus.append(m_.groups())
        return vus
    chk("la note ELS de PRS-C4 : la garde et 3 pages, comme sans les dessins ; aucun texte d'avant perdu (seuls les rappels « (suite) » "
        "changent de place) ; les blocs dans le même ordre ; aucun mot ne sort de sa colonne ni de la page",
        len(pages) == len(p0) == 4 and not perdus and ordre(pages) == ordre(p0) and len(ordre(pages)) >= 8
        and not tn.debordements(octets) and not tn.entetes_chevauches(octets), (len(pages), len(p0), perdus, ordre(pages)[:12]))
    print("== 9. l'Eurocode 3 : la flexion en élastique seulement")
    e3 = vue(barre(params={"ame_classe_3": "elastique"}), "ec3").get("contraintes")
    t3 = textes_svg(e3["svg"]) if e3 else []
    chk("variante « élastique » : le volet « Flexion » de l'Eurocode 3 écrit 150,3 et 97,3, f_y/γ_M0 = 215,0",
        e3 is not None and t3.count("Flexion") == 1 and all(x in t3 for x in ("150,3", "97,3", "fy/γM0 = 215,0")), t3)
    chk("classe 2 efficace (le défaut) : pas de dessin de l'Eurocode 3 — ni vue, ni dessin dans la note",
        "contraintes" not in vue(barre(), "ec3")
        and not any(it_[0] == "d" for b_ in NP.planche(el, r, NP.Planche(), "ec3")["verifs"] for it_ in b_["items"]))
    e3n = P.normaliser(P.etat_de([barre(params={"ame_classe_3": "elastique"})], impression="ec3"))["elements"][0]
    s3 = NP.planche(e3n, P.calculer(e3n), NP.Planche(), "ec3")
    chk("… et dans la note en élastique : un dessin, sous la flexion de l'Eurocode 3",
        [b_["titre"] for b_ in s3["verifs"] if any(it_[0] == "d" for it_ in b_["items"])] == ["Eurocode 3 · flexion, axe fort"],
        [b_["titre"] for b_ in s3["verifs"] if any(it_[0] == "d" for it_ in b_["items"])])
    print("== 10. les formes")
    for prof in ("IPE 300", "RHS 200x100x8", "CHS 168.3x6.3"):
        b_ = P.barre_neuve("B1")
        b_["profil"] = prof
        v_ = vue(b_, "deux").get("contraintes")
        rond = prof.startswith("CHS")
        chk(f"{prof} : {'pas de vue (un tube rond)' if rond else 'le dessin sort (une coupe peinte, ses diagrammes)'}",
            (v_ is None) if rond else (v_ is not None and "<polygon" in v_["svg"] and "Flexion" in textes_svg(v_["svg"])), v_ and v_["titre"])
    revue_dessin()


def barre_efforts(prof, N, My, Vz, Lc=0.0, prs=None):
    """Une barre en efforts imposés (§ 2.3 : « les mêmes volets, dans la section donnée ») : N (kN, > 0 compression), M_y (kN·m),
    V_z (kN) de service ; ×1,4 à l'ELU ; aucun flambement."""
    b = P.barre_neuve("B1")
    b.update(profil=prof, mode="efforts", flambement={"y": {"L": 0.0, "u": 1.0}, "z": {"L": 0.0, "u": 1.0}}, Lc=Lc,
             efforts={"service": {"N": N, "My": My, "Mz": 0.0, "Vz": Vz, "Vy": 0.0},
                      "elu": {"N": 1.4 * N, "My": 1.4 * My, "Mz": 0.0, "Vz": 1.4 * Vz, "Vy": 0.0}})
    if prs:
        b["prs"] = prs
    return b


def volets_de(b, k="historique"):
    etat = P.normaliser(P.etat_de([b], impression=k))
    r = P.calculer(etat["elements"][0])
    s = MO.section_de(r["barre"])
    return s, DP.volets(k, r["methodes"][k], s)


def contours(svg):
    """Les contours des diagrammes (un par volet, dans l'ordre) : la liste de leurs points (x, y), la base comprise."""
    return [[tuple(map(float, p.split(","))) for p in d.split(" L")]
            for d in re.findall(r'<path d="M([^"]+)" fill="none" stroke="[^"]+" stroke-width="0.6" stroke-linejoin="round"/>', svg)]


def hors_cadre(svg):
    """Les textes qui sortent du dessin, à gauche, à droite, en haut ou en bas (leur largeur estimée comme le dessin l'estime ;
    du haut des lettres à leurs jambages)."""
    W, H = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    out = []
    for x, y, ancre, fs, corps in re.findall(r'<text x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"[^>]*>(.*?)</text>', svg, re.S):
        t = blanc(re.sub(r"<[^>]+>", "", corps)).replace("​", "").strip()
        lw, f = DP.largeur_gras(t, float(fs)), float(fs)
        a = float(x) - {"start": 0.0, "middle": lw / 2.0, "end": lw}[ancre]
        if a < -0.5 or a + lw > W + 0.5 or float(y) - ASC * f < -0.5 or float(y) + DESC * f > H + 0.5:
            out.append((t, round(a, 1), round(a + lw, 1), float(y), W, H))
    return out


def textes_y(svg):
    """(texte, ligne de base) de chaque texte d'un dessin."""
    return [(blanc(re.sub(r"<[^>]+>", "", c_)).replace("​", "").strip(), float(y_))
            for y_, c_ in re.findall(r'<text x="[-\d.]+" y="([-\d.]+)"[^>]*>(.*?)</text>', svg, re.S)]


ASC, DESC = 0.93, 0.24      # la boîte d'un texte, comme l'écran la mesure (getBoundingClientRect) : du haut des lettres à leurs jambages


def chevauches(svg):
    """Les paires de textes qui se recouvrent (une boîte par texte : sa largeur estimée, du haut des lettres à leurs jambages —
    la boîte de l'écran ; à 0,72 au-dessus de la ligne de base seulement, deux lignes à 7 l'une de l'autre passaient ici et
    se touchaient au harnais)."""
    boites = []
    for x, y, ancre, fs, corps in re.findall(r'<text x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"[^>]*>(.*?)</text>', svg, re.S):
        t = blanc(re.sub(r"<[^>]+>", "", corps)).replace("​", "").strip()
        lw, f = DP.largeur_gras(t, float(fs)), float(fs)
        a = float(x) - {"start": 0.0, "middle": lw / 2.0, "end": lw}[ancre]
        boites.append((t, a, a + lw, float(y) - ASC * f, float(y) + DESC * f))
    return [(p[0], q[0]) for i, p in enumerate(boites) for q in boites[i + 1:]
            if p[1] < q[2] and q[1] < p[2] and p[3] < q[4] and q[3] < p[4]]


def revue_dessin():
    """Les défauts du dessin trouvés à la relecture de l'étape 2, chacun figé : le test rougit si on retire le correctif."""
    print("== 11. la relecture du dessin")
    # (a) σ_eq(z) = √(σ(z)² + 3τ²), σ(z) = N/A + M_y·(z − z_g)/I_y : sous une traction seule, σ_eq est LE MÊME sur toute la hauteur
    #     — 188,9 au centre de gravité comme aux fibres (le dessin perdait N/A : √3·τ = 33,7 au centre de gravité)
    s, vs = volets_de(barre_efforts("IPE 300", -1000.0, 0.0, 50.0))
    svg = DP.contraintes("t_eq", s, [vs["eq"]], 150.0) if "eq" in vs else ""
    c_ = contours(svg)
    xs = [x for x, _y in c_[0][1:-1]] if c_ else []
    chk("IPE 300, N = −1 000 kN, V_z = 50 kN : le volet « Contrainte équivalente » est droit (σ_eq = 188,9 partout, au centre de "
        "gravité aussi)", bool(xs) and max(xs) - min(xs) < 0.05 and "188,9" in textes_svg(svg), (xs and (min(xs), max(xs)), textes_svg(svg)))
    # (b) la valeur de l'autre fibre, du même signe et presque aussi grande, ne sort pas du volet (« 185,8 »)
    svg_s = DP.contraintes("t_n", s, [vs["sigma"]], 150.0) if "sigma" in vs else ""
    chk("… le volet « Flexion » (traction seule, 185,8 aux deux fibres) : aucun texte ne sort du volet",
        bool(svg_s) and textes_svg(svg_s).count("185,8") == 2 and not hors_cadre(svg_s), hors_cadre(svg_s))
    # (c) H-DEV dessine σ_M, la flexion seule — sans N : sous N = 300 kN et M_y = 30 kN·m, la fibre basse est COMPRIMÉE (55,8 − 53,9
    #     > 0) ; le volet « Flexion » est alors celui de la contrainte normale (H-SIGMA, qui porte N) : les deux fibres comprimées,
    #     peintes et dites comme telles
    s, vs = volets_de(barre_efforts("IPE 300", 300.0, 30.0, 0.0, Lc=3.0))
    svg_d = DP.contraintes("t_dev", s, [vs["sigma"]], 150.0) if "sigma" in vs else ""
    td = textes_svg(svg_d)
    chk("IPE 300, N = 300 kN, M_y = 30 kN·m, L_c = 3 m : le volet « Flexion » est celui de la contrainte normale (H-SIGMA, il porte N) ; "
        "les deux fibres « comprimée », aucune « tendue », aucun chiffre en bleu de traction",
        vs.get("sigma", {}).get("id") == "H-SIGMA" and td.count("comprimée") == 2 and "tendue" not in td
        and f'fill="{DP.TRACTION}"' not in "".join(re.findall(r"<text[^>]*>", svg_d)), (vs.get("sigma", {}).get("id"), td))
    # (c') la flexion SEULE (σ_M sans N, σ de l'Eurocode 3) : les mots disent l'état RÉEL de la fibre (N_reel) — une fibre « tendue »
    #     par la flexion mais comprimée sous N n'a pas de mot
    vo = {"genre": "sigma", "titre": "Flexion", "lieu": None, "limite": ("f_y/γ_M0", 235.0), "N": 0.0, "N_reel": 300e3, "My": 30e6,
          "sigma": 30e6 * (s["h"] / 2.0) / s["Iy"] if s.get("Iy") else None}
    try:
        svg_c = DP.contraintes("t_reel", s, [vo], 150.0)
    except Exception as ex:                                       # (un volet mal formé : le contrôle le dit)
        svg_c = f"erreur {ex}"
    tc = textes_svg(svg_c)
    chk("… un volet de flexion seule (N_reel = 300 kN, M_y = 30 kN·m) : « comprimée » en haut, la fibre basse sans mot",
        tc.count("comprimée") == 1 and "tendue" not in tc, tc)
    # (d) une coupe élancée : les mots « comprimée » et « tendue » ne sortent pas du volet, à gauche
    prs = {"b1": 200.0, "t1": 20.0, "hw": 3000.0, "tw": 12.0, "b2": 200.0, "t2": 20.0, "a": 6.0, "bords": "oxycoupes", "nom": None}
    s, vs = volets_de(barre_efforts("PRS", 0.0, 2000.0, 0.0, prs=prs))
    svg_p = DP.contraintes("t_prs", s, [vs["sigma"]], 150.0) if "sigma" in vs else ""
    chk("PRS 200×20 / 3000×12, M_y = 2 000 kN·m : « comprimée » et « tendue » sont écrits, et aucun texte ne sort du volet",
        bool(svg_p) and {"comprimée", "tendue"} <= set(textes_svg(svg_p)) and not hors_cadre(svg_p), hors_cadre(svg_p))
    # (e) le harnais : une poutre SYMÉTRIQUE — le nom de la limite (« σ_adm,d = … », sous la fibre basse) recouvrait la valeur de la
    #     fibre tendue, écrite à gauche de la base à la même hauteur ; et à valeurs égales, c'est la fibre COMPRIMÉE qui gouverne
    s, vs = volets_de(barre_efforts("IPE 300", 0.0, 50.0, 0.0, Lc=3.0))
    svg_e = DP.contraintes("t_sym", s, [vs["sigma"]], 150.0) if "sigma" in vs else ""
    te = textes_svg(svg_e)
    chk("IPE 300, M_y = 50 kN·m, L_c = 3 m (symétrique) : aucun texte n'en recouvre un autre, la valeur en gras est celle de la fibre "
        "comprimée (en haut, en rouge)", bool(svg_e) and not chevauches(svg_e) and not hors_cadre(svg_e)
        and re.search(rf'<text [^>]*font-size="8" fill="{DP.COMPRESSION}"', svg_e) is not None, (chevauches(svg_e), te))
    # (f) un profil REMIS À L'ÉCHELLE de la formule (k ≠ 1 : ici |N|/A + M/W_el,min, qui somme les valeurs absolues) : la valeur de
    #     l'autre fibre ne serait celle d'aucune formule (« 43,1 » sur une fibre à 32,8) — seule la plus grande (90,3) est écrite
    prs_c4 = {"b1": 300.0, "t1": 30.0, "hw": 1420.0, "tw": 25.0, "b2": 500.0, "t2": 50.0, "a": 10.0, "bords": "oxycoupes", "nom": None}
    s, vs = volets_de(barre_efforts("PRS", -2000.0, 1500.0, 0.0, prs=prs_c4))
    svg_f = DP.contraintes("t_k", s, [vs["sigma"]], 150.0) if "sigma" in vs else ""
    tf_ = textes_svg(svg_f)
    nombres = [t_ for t_ in tf_ if re.fullmatch(r"\d+,\d", t_)]
    chk("PRS-C4, N = −2 000 kN, M_y = 1 500 kN·m : la valeur de la formule (90,3) en gras, et aucune autre valeur de fibre",
        nombres == ["90,3"], tf_)
    # (g) V_y ≠ 0 n'écarte que le cisaillement et l'équivalente : la flexion se dessine
    b_g = barre_efforts("IPE 300", 0.0, 60.0, 20.0)
    for cas in ("service", "elu"):
        b_g["efforts"][cas]["Vy"] = 5.0 * (1.4 if cas == "elu" else 1.0)
    s, vs = volets_de(b_g)
    chk("IPE 300, M_y = 60 kN·m, V_z = 20 kN, V_y = 5 kN : le volet « Flexion » est dessiné, ni cisaillement ni équivalente",
        set(vs) == {"sigma"}, sorted(vs))
    # (h) la valeur en gras de la contrainte équivalente est à la fibre de celle de la flexion — N et M compriment le bas ; et à
    #     valeurs égales (bisymétrique, M_y < 0), à la fibre comprimée
    def fibre_gras(svg_):
        ys = [float(y_) for y_ in re.findall(r'<text x="[-\d.]+" y="([-\d.]+)" text-anchor="start" font-size="8" ', svg_)]
        return ys[0] if len(ys) == 1 else None
    for prof, N_, My_, Vz_ in (("RHS 200x100x8", 300.0, -30.0, 20.0), ("IPE 300", 0.0, -50.0, 20.0)):
        s, vs = volets_de(barre_efforts(prof, N_, My_, Vz_))
        ok_ = {"sigma", "eq"} <= set(vs)
        y_s = fibre_gras(DP.contraintes("t_hs", s, [vs["sigma"]], 150.0)) if ok_ else None
        y_e = fibre_gras(DP.contraintes("t_he", s, [vs["eq"]], 150.0)) if ok_ else None
        chk(f"{prof}, N = {N_:g} kN, M_y = {My_:g} kN·m : les valeurs en gras de « Flexion » et de « Contrainte équivalente » à la même "
            "fibre, la basse", ok_ and y_s is not None and y_s == y_e and y_s > 100.0, (sorted(vs), y_s, y_e))
    # (i) le harnais, encore : trois volets côte à côte à l'écran (450), une poutre au-delà de sa limite — le diagramme prend la place,
    #     le nom de la limite ne tient plus à droite de la base et recouvrait la valeur de l'autre fibre ; il passe à la ligne
    rangs = []
    for My_ in (62.0, -62.0, 120.0):
        s, vs = volets_de(barre_efforts("IPE 300", 0.0, My_, 20.0, Lc=3.0))
        # (l'abscisse en tête, comme à l'écran : un nom passé AU-DESSUS ne la touche pas, ni le filet de la tête)
        svg_i = DP.contraintes("t_i", s, [dict(vs[g_], lieu="x = 14,00 m") for g_ in ("sigma", "tau", "eq") if g_ in vs], 450.0, 110.0)
        tetes = [(t_, y_) for t_, y_ in textes_y(svg_i) if y_ > 8.0 and y_ - ASC * 8.0 < 11.2 + 0.5]
        rangs.append((My_, len(vs), chevauches(svg_i), hors_cadre(svg_i), "σadm,d = 95,3" in textes_svg(svg_i), tetes))
    chk("IPE 300, M_y = ±62 et 120 kN·m, V_z = 20 kN, L_c = 3 m, trois volets sur 450 : aucun texte sur un autre, aucun hors du "
        "dessin ni sur le filet de la tête, le nom de la limite écrit",
        all(n_ == 3 and not c_ and not h_ and e_ and not t_ for _m, n_, c_, h_, e_, t_ in rangs), rangs)
    # (i') la règle, nue : un écart ESTIMÉ de 4 entre le nom et la valeur de l'autre fibre ne suffit pas (le harnais : la police de
    #     l'écran est plus large que l'estimation, « σ_adm,d = 60,5 » sur « 107,7 ») — le nom passe à la ligne ; à 8, il reste
    K_ = DP.Coupe(s, 3.0, DP.H_TETE + DP.Y_CARTE, 110.0, 52.0)
    nom_ = "σ_adm,d = 60,5"
    demi_ = DP.largeur_gras(nom_, 6.2) / 2.0
    xb_ = K_.x1 + 30.0

    def passe(ecart):
        x_fin_ = xb_ + 1.0 + 2.0 * demi_ + 1.0                    # (le nom calé contre la base : x_nom − demi = xb + 1)
        bas_ = DP._habillage(K_, xb_, 0.1, ("σ_adm,d", 60.5), x_fin_, 107.7, True, DP.COMPRESSION, (xb_ - 30.0, xb_ + 1.0 - ecart))[1]
        haut_ = DP._habillage(K_, xb_, 0.1, ("σ_adm,d", 60.5), x_fin_, 107.7, False, DP.COMPRESSION, (xb_ - 30.0, xb_ + 1.0 - ecart))[1]
        return bas_, haut_
    chk("la règle du nom de la limite : à 4 d'écart estimé il passe à la ligne (sous le volet ou au-dessus), à 8 il reste sur la sienne",
        passe(4.0) == ((DP.LIGNE_LIMITE, 0.0), (0.0, DP.LIGNE_LIMITE)) and passe(8.0) == ((0.0, 0.0), (0.0, 0.0)), (passe(4.0), passe(8.0)))
    tous = [x_ for x_ in (svg, svg_s, svg_d, svg_p, svg_e, svg_c, svg_f) if x_]
    chk("… et dans tous les dessins de cette relecture, aucun texte n'en recouvre un autre", not any(chevauches(x_) for x_ in tous),
        [chevauches(x_) for x_ in tous])


if __name__ == "__main__":
    sys.exit(main())
