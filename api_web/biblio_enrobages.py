# -*- coding: utf-8 -*-
"""La rubrique « ENROBAGES » de la Bibliothèque Béton armé (docs/site/LOT_Q_QUATRE_MODULES.md § 4.3,
docs/site/BIBLIOTHEQUES.md § 2.2) : DEUX ONGLETS qui partagent les mêmes paramètres, portés par l'adresse
(``?rubrique=enrobages&onglet=calcul&element=poteau&feu=R90&mu_fi=0.7…``) :

- « Calcul » (l'onglet ouvert par défaut) : le calcul complet de l'enrobage d'UN élément — à gauche le DESSIN du détail
  à l'échelle (api_web/dessin_biblio_enrobage.py), à droite la FEUILLE DE CALCUL : la chaîne terme par terme, chaque
  terme avec son calcul, sa valeur et sa règle (les ``termes`` du moteur, aucun ajouté ici), la ligne de c_nom allumée ;
  sous le dessin, les conditions d'emploi du tableau et son état de vérification ;
- « Tableaux » : les tableaux de référence, la case de la situation allumée et chaque case cliquable — « Durabilité »
  (tableau 4.4N) et « Feu » (le tableau ENTIER de l'élément et de sa variante : 5.2a, 5.3/5.4, 5.5/5.6, 5.8/5.9) ; sous le
  tableau du feu, ses notes, ses conditions, sa source et son état de vérification.

L'ancien paramètre ``vue=durabilite|feu`` reste lu : sans ``onglet``, il ouvre « Tableaux » sur ce tableau. Les
valeurs viennent de ``beton/bibliotheque/enrobage.py`` et ``feu.py`` (R3) ; l'écran choisit, compose et formate. Un
paramètre illisible reprend son défaut, et ``message`` le dit ; une saisie que la formule (5.7) refuse ramène le poteau
au tableau 5.2a, et ``message`` dit pourquoi.
"""
from typing import Dict, List, Optional

from api_web import biblio_beton as B
from api_web import commun as C
from api_web import dessin_biblio_enrobage as D
from beton.bibliotheque import enrobage as EN
from beton.bibliotheque import feu as FEU

ONGLETS = (("calcul", "Calcul"), ("tableaux", "Tableaux"))
VUES = (("durabilite", "Durabilité · 4.4N"), ("feu", "Feu · EN 1992-1-2"))
FEUX = (("", "sans exigence"),) + tuple((r, FEU.libelle_duree(r)) for r in EN.R_DUREES)
ELEMENTS = tuple(EN.ELEMENTS)
NOMS = dict(ELEMENTS)
# la dimension de chaque élément : (paramètre de l'adresse, paramètre du moteur, libellé, aide, défaut en mm)
DIMENSIONS = {
    "poutre": ("b", "b", "Largeur b", "largeur de la poutre : elle choisit la combinaison b_min / a du tableau", 300),
    "poteau": ("b", "b", "Côté b", "plus petit côté du poteau rectangulaire, ou son diamètre", 300),
    "dalle": ("h", "h", "Épaisseur h", "épaisseur de la dalle, revêtement de sol incombustible compris (h_s)", 200),
    "voile": ("e", "e", "Épaisseur e", "épaisseur du voile", 200),
}
MU_POTEAU = tuple(sorted(FEU.MU_POTEAU))                 # (0,2 ; 0,5 ; 0,7) : les colonnes du tableau 5.2a
BARRES = (("4", "4 barres"), ("plus", "plus de 4"))
NBSP = C.NBSP


def _fr(x) -> str:
    """Un nombre de la feuille : au dixième au plus, sans zéro inutile (30 ; 29,5 ; 24,9)."""
    return C.nb(float(x), 1)


def _q(x, unite: str) -> str:
    """Une quantité : le nombre et son unité, jamais coupés (« 29,5 mm »)."""
    return f"{_fr(x)}{NBSP}{unite}" if unite else _fr(x)


def _m(texte: str) -> str:
    """Une expression du moteur écrite comme dans la note (indices ; Ø_étrier et Ø_ext indicés aussi)."""
    return C.symb_math(texte).replace("Ø_étrier", "Ø<sub>étrier</sub>").replace("Ø_ext", "Ø<sub>ext</sub>")


# ================================================================================================ les paramètres
def lire(params: Optional[Dict]) -> tuple:
    """(lecture, paramètres du moteur, onglet, vue) : chaque paramètre relu (les absents prennent leur défaut, un
    illisible aussi — et le message le dit). Seuls les paramètres de l'élément choisi sont relus (l'adresse ne garde
    pas la largeur d'une poutre quand on regarde un voile)."""
    L = B.Lecture(params)
    brut = L.brut
    if "onglet" in brut:
        onglet = L.choix("onglet", tuple(k for k, _ in ONGLETS), "calcul")
    else:                                       # l'ancienne adresse (vue=…) ouvre les tableaux
        onglet = "tableaux" if "vue" in brut else "calcul"
        L.out["onglet"] = onglet
    element = L.choix("element", tuple(NOMS), "poutre")
    expo = L.choix("exposition", EN.EXPO_CLES, "XC1")
    classe = L.choix("classe", B.CLASSES_BETON, "C30/37")
    duree = L.choix("duree", EN.DUREES, 50)
    qualite = L.case("qualite", False)
    cs = L.choix("classe_structurale", ("",) + EN.CLASSES_STRUCTURALES, "")
    sol = L.choix("contre_sol", tuple(k for k, _, _ in EN.CONTRE_SOL), "non")
    phi = L.nombre("phi", 20, 6, 60, entier=True)
    if element in EN.ELEMENTS_A_ETRIER:
        phi_e = L.nombre("phi_etrier", 8, 0, 40, entier=True)
        phi_x = 0
    else:
        phi_x = L.nombre("phi_ext", 0, 0, 40, entier=True)
        phi_e = 8
    dg = L.nombre("dg", 20, 4, 63, entier=True)
    ddev = L.choix("delta_dev", tuple(d for d, _ in EN.DELTA_DEV), 10)
    cle_dim, cle_moteur, _, _, defaut_dim = DIMENSIONS[element]
    dim = L.nombre(cle_dim, defaut_dim, 20, 5000, entier=True)
    feu = L.choix("feu", tuple(k for k, _ in FEUX), "")
    vue = L.choix("vue", tuple(k for k, _ in VUES), "feu" if feu else "durabilite")
    f = {cle_moteur: dim}
    if element == "poutre":
        f["appuis"] = L.choix("appuis", tuple(k for k, _, _ in FEU.APPUIS), "simples")
    elif element == "dalle":
        f["portee"] = L.choix("portee", tuple(k for k, _ in FEU.PORTEES), "une")
    elif element == "voile":
        f["porteur"] = L.case("porteur", True)
        if f["porteur"]:
            f["mu_fi"] = L.choix("mu_voile", FEU.MU_VOILE, 0.7)
            f["faces"] = L.choix("faces", (1, 2), 1)
    else:
        f["methode"] = L.choix("methode", tuple(k for k, _ in FEU.METHODES_POTEAU), FEU.METHODE_DEFAUT)
        f["exposition"] = L.choix("expo_feu", tuple(k for k, _ in FEU.EXPOSITIONS_POTEAU), "plusieurs")
        mu = L.nombre("mu_fi", 0.7, 0.01, 0.7)
        if f["methode"] == "tableau":
            # le tableau n'a que trois colonnes : un μ_fi entre deux se lit dans la colonne du DESSUS (côté sûr), dit
            col = next(m for m in MU_POTEAU if m >= mu - 1e-9)
            if abs(col - mu) > 1e-9:
                L.messages.append(f"μ_fi = {C.nb(mu, 2)} : le tableau 5.2a n'a que les colonnes 0,2, 0,5 et 0,7 — celle de {C.nb(col, 1)} "
                                  "est lue (côté sûr)")
                L.out["mu_fi"] = col
            f["mu_fi"] = col
        else:
            f["mu_fi"] = mu
            f["h"] = L.nombre("h_grand", None, 20, 5000, entier=True, vide=None)
            f["l0_fi"] = L.nombre("l0_fi", FEU.L0_FI_DEFAUT, 0.1, 50)
            f["barres"] = L.choix("barres", tuple(k for k, _ in BARRES), "4")
    moteur = {"element": element, "exposition": expo, "classe": classe, "duree": duree, "qualite": qualite, "phi": phi,
              "phi_etrier": phi_e, "phi_ext": phi_x, "dg": dg, "delta_dev": ddev, "contre_sol": sol, "classe_structurale": cs or None,
              "feu": {"r": feu, **f} if feu else None, "_selection": {k: v for k, v in f.items() if k in ("appuis", "portee", "porteur")}}
    return L, moteur, onglet, vue


def calculer(L, moteur: Dict) -> Dict:
    """Le calcul du moteur ; une saisie que la formule (5.7) refuse (hors de son domaine) ramène le poteau au tableau
    5.2a — c'est DIT dans le message, et l'adresse suit (methode=tableau)."""
    m = {k: v for k, v in moteur.items() if k != "_selection"}
    try:
        return EN.calculer(m)
    except ValueError as e:
        f = m.get("feu") or {}
        if m["element"] != "poteau" or f.get("methode") != "formule":
            raise
        L.messages.append(f"formule (5.7) refusée : {e} — le tableau 5.2a est lu")
        col = next(x for x in MU_POTEAU if x >= f["mu_fi"] - 1e-9)
        f2 = {"r": f["r"], "methode": "tableau", "exposition": f["exposition"], "b": f["b"], "mu_fi": col}
        for k in ("h_grand", "l0_fi", "barres"):
            L.out.pop(k, None)
        L.out["methode"], L.out["mu_fi"] = "tableau", col
        moteur["feu"] = f2
        return EN.calculer({**m, "feu": f2})


# ================================================================================================ la barre
def _seg(cle: str, valeur, options, inactifs=()) -> str:
    """Des segments (B._segments), certains grisés avec leur raison au survol : {valeur: raison}."""
    if not inactifs:
        return B._segments(cle, valeur, options)
    bt = ""
    for k, t in options:
        on = ' class="on"' if str(k) == str(valeur) else ""
        dis = f' disabled title="{C.esc(inactifs[k])}"' if k in inactifs else ""
        bt += f'<button type="button"{on}{dis} data-valeur="{C.esc(str(k))}">{t}</button>'
    return f'<div class="seg" data-param="{C.esc(cle)}">{bt}</div>'


def _groupe(titre: str, contenu: str, cls: str = "") -> str:
    """Un groupe de paramètres de la barre, nommé (Élément, Feu, Environnement, Armatures)."""
    return f'<div class="bib-gr{(" " + cls) if cls else ""}" role="group" aria-label="{C.esc(titre)}"><span class="bib-gr-t">{titre}</span>{contenu}</div>'


def _barre(p: Dict, moteur: Dict, res: Dict, onglet: str, vue: str) -> str:
    element = p["element"]
    s = res["durabilite"]["classe_structurale"]
    expos = [(e[0], f"{e[0]} · {e[2]}" if e[2] != "—" else e[0]) for e in EN.EXPOSITIONS]
    cle_dim, _, lib_dim, aide_dim, _ = DIMENSIONS[element]
    onglets = B._champ("onglet", "", B._segments("onglet", onglet, ONGLETS)).replace('class="bib-p bib-p-case"', 'class="bib-p bib-onglets"')
    if onglet == "tableaux":
        onglets += B._champ("vue", "Tableau", B._segments("vue", vue, VUES))
    el = (B._champ("element", "Type", B._segments("element", element, ELEMENTS))
          + B._champ(cle_dim, lib_dim, B._num(cle_dim, p[cle_dim], "mm"), aide_dim))
    feu = B._champ("feu", "Durée", B._select("feu", p["feu"], FEUX), "résistance au feu normalisé (EN 1992-1-2, méthode tabulée) ; « sans exigence » : la durabilité seule")
    if p["feu"] or (onglet == "tableaux" and vue == "feu"):
        feu += _barre_feu(p, element)
    env = (B._champ("exposition", "Exposition", B._select("exposition", p["exposition"], expos), dict((e[0], e[4]) for e in EN.EXPOSITIONS)[p["exposition"]])
           + B._champ("classe", "Béton", B._select("classe", p["classe"], [(c, c) for c in B.CLASSES_BETON]))
           + B._champ("duree", "Durée d'utilisation", B._segments("duree", p["duree"], ((50, "50 ans"), (100, "100 ans"))))
           + B._champ("classe_structurale", "Classe S", B._select("classe_structurale", p["classe_structurale"],
                                                                   [("", f"calculée · {s['calculee']}")] + [(k, k) for k in EN.CLASSES_STRUCTURALES]),
                      "S4 pour 50 ans, puis les modifications du tableau 4.3N (NIT 260, tableau 21) ; une classe imposée remplace le calcul")
           + B._champ("qualite", "", B._case("qualite", p["qualite"], "Qualité maîtrisée", "maîtrise particulière de la qualité (préfabrication sous les conditions de la NBN EN 13369) : − 1 classe structurale"))
           + B._champ("contre_sol", "Contre le sol", B._select("contre_sol", p["contre_sol"], [(k, t if not v else f"{t.split(' (')[0]} · {v}{NBSP}mm") for k, t, v in EN.CONTRE_SOL]),
                      " ; ".join(f"{t} : c_nom ≥ {v} mm" for _, t, v in EN.CONTRE_SOL if v) + " (§4.4.1.3(4))"))
    if element in EN.ELEMENTS_A_ETRIER:
        ext = B._champ("phi_etrier", "Ø étrier", B._num("phi_etrier", p["phi_etrier"], "mm"), "l'étrier (le cadre) qui entoure les barres principales : il se retire de a")
    else:
        ext = B._champ("phi_ext", "Ø nappe ext.", B._num("phi_ext", p["phi_ext"], "mm"),
                       "Ø de la nappe extérieure, s'il y en a une entre la barre et le parement (0 sinon) — une dalle ou un voile n'a pas d'étrier")
    arm = (B._champ("phi", "Ø barre", B._num("phi", p["phi"], "mm"), "Ø des armatures principales (c_min,b = Ø, tableau 4.2)") + ext
           + B._champ("dg", "Granulat", B._num("dg", p["dg"], "mm"), "au-delà de 32 mm, c_min,b + 5 mm (tableau 4.2)")
           + B._champ("delta_dev", "Δc<sub>dev</sub>", B._select("delta_dev", p["delta_dev"], [(k, f"{k} mm") for k, _ in EN.DELTA_DEV]), " ; ".join(t for _, t in EN.DELTA_DEV)))
    # (l'ordre tient la barre en trois lignes au PC : onglets, élément et armatures ; l'environnement ; le feu)
    return B._barre(f'<div class="bib-onglets-l">{onglets}</div>' + _groupe("Élément", el) + _groupe("Armatures", arm)
                    + _groupe("Environnement", env) + _groupe("Feu", feu, "bib-gr-feu"))


def _barre_feu(p: Dict, element: str) -> str:
    if element == "poutre":
        return B._champ("appuis", "Appuis", B._segments("appuis", p["appuis"], (("simples", "simples · 5.5"), ("continue", "continue · 5.6"))),
                        "poutre continue : redistribution des moments à froid ≤ 15 % (sinon : sur appuis simples)")
    if element == "dalle":
        return B._champ("portee", "Portée", B._select("portee", p["portee"], [(k, t) for k, t in FEU.PORTEES]),
                        "une ou deux directions (tableau 5.8, l_y/l_x), continue (§5.7.3), plancher-dalle (tableau 5.9)")
    if element == "voile":
        out = B._champ("porteur", "", B._case("porteur", p["porteur"], "Porteur", "voile porteur : tableau 5.4 ; non porteur (cloison) : tableau 5.3, l'épaisseur seule"))
        if p["porteur"]:
            out += (B._champ("mu_voile", "μ<sub>fi</sub>", B._segments("mu_voile", p["mu_voile"], [(m, C.nb(m, 2)) for m in FEU.MU_VOILE]),
                             "μ_fi = N_Ed,fi / N_Rd : les colonnes 0,35 et 0,7 du tableau 5.4 (0,7 sans calcul)")
                    + B._champ("faces", "Faces", B._segments("faces", p["faces"], ((1, "une"), (2, "deux"))), "faces exposées au feu"))
        return out
    un = p["expo_feu"] == "un"
    out = B._champ("methode", "Méthode", _seg("methode", p["methode"], (("tableau", "Tableau 5.2a"), ("formule", "Formule (5.7)")),
                                              {"formule": "la formule (5.7) vaut pour un poteau exposé sur plus d'un côté"} if un else ()),
                   "le tableau 5.2a de l'EN est la lecture par défaut (la plus sévère) ; la formule (5.7) reproduit les tableaux de l'annexe belge — à contrôler sur l'ANB")
    out += B._champ("expo_feu", "Côtés exposés", B._segments("expo_feu", p["expo_feu"], (("plusieurs", "plusieurs"), ("un", "un seul"))))
    if p["methode"] == "tableau":
        out += B._champ("mu_fi", "μ<sub>fi</sub>", _seg("mu_fi", p["mu_fi"], [(m, C.nb(m, 1)) for m in MU_POTEAU],
                                                         {m: "exposé sur un seul côté : le tableau 5.2a n'a que la colonne μ_fi = 0,7" for m in MU_POTEAU if m != 0.7} if un else ()),
                        "μ_fi = N_Ed,fi / N_Rd (éq. 5.6) ; 0,7 sans calcul ; " + ("un seul côté : le tableau ne donne que μ_fi = 0,7" if un else "colonnes 0,2, 0,5, 0,7"))
    else:
        out += (B._champ("mu_fi", "μ<sub>fi</sub>", B._num("mu_fi", p["mu_fi"]), "μ_fi = N_Ed,fi / N_Rd, de 0,2 à 0,7 (sous 0,2 : 0,2, côté sûr)")
                + B._champ("h_grand", "Grand côté h", B._num("h_grand", p.get("h_grand"), "mm", "= b"), "h ≤ 1,5·b (vide : poteau carré)")
                + B._champ("l0_fi", "l<sub>0,fi</sub>", B._num("l0_fi", p["l0_fi"], "m"), "longueur efficace au feu, de 2 à 6 m (sous 2 m : 2 m, côté sûr)")
                + B._champ("barres", "Barres", B._segments("barres", p["barres"], BARRES), "R_n = 0 pour 4 barres (aux angles), 12 pour plus de 4"))
    return out


# ================================================================================================ l'onglet « Calcul »
# les avertissements du moteur qui se lisent sous le dessin (et sous le tableau, pour les conditions) plutôt que dans
# la bande : ce que la lecture a choisi ou ramené (« mentions »), les conditions d'emploi ; tous restent dans
# « Pourquoi ? » et dans la fiche
A_SOUS_DESSIN = ("feu_mention", "feu_conditions")
GROUPES = (("durabilite", "Durabilité", "EN 1992-1-1 §4.4.1"), ("feu", "Feu", ""), ("retenu", "", ""))


def _valeur(t: Dict) -> str:
    v = t["valeur"]
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, (int, float)):
        return _q(v, t.get("unite", ""))
    return C.esc(str(v)) + (f"{NBSP}{C.esc(t['unite'])}" if t.get("unite") and str(v)[:1].isdigit() else "")


def _comparaison(t: Dict) -> str:
    cmp_ = t.get("comparaison")
    etat = t.get("etat")
    if cmp_ and cmp_.get("valeur") is not None:
        ok = cmp_["ok"]
        u = t.get("unite") if t.get("unite") in ("mm", "min") else ""
        txt = f"{cmp_['sens']} {_m(cmp_['symbole'])} = {_q(cmp_['valeur'], u)}"
        return f'<span class="bib-cmp {"ok" if ok else ("ko" if ok is False else "nv")}">{txt}{(" · " + C.esc(etat)) if etat else ""}</span>'
    if etat:
        return f'<span class="bib-cmp {"ok" if etat == "atteinte" else ("ko" if etat == "non atteinte" else "nv")}">{C.esc(etat)}</span>'
    return ""


def _feuille(res: Dict) -> str:
    """La feuille de calcul : un groupe par étape (durabilité, feu, puis l'enrobage retenu), une ligne par terme du
    moteur — symbole, nom, calcul (l'expression, puis les valeurs), valeur (et sa comparaison), référence (la règle
    complète au survol de la ligne et dans « Pourquoi ? ») ; la ligne de c_nom allumée."""
    f = res["feu"]
    corps = ""
    for g, titre, sous in GROUPES:
        termes = [t for t in res["termes"] if t["groupe"] == g]
        if not termes:
            continue
        if g == "feu" and f:
            sous = f"{FEU.NORME}, {'formule (5.7)' if f['methode'] == 'formule' else 'tableau ' + f['tableau']} — {f['duree']}, {f['colonne_titre']}"
        if titre:                                   # (l'enrobage retenu suit sans titre : sa ligne est allumée)
            corps += f'<tr class="g"><th colspan="5"><b>{titre}</b>{(" <span>" + _m(sous) + "</span>") if sous else ""}</th></tr>'
        for t in termes:
            cle = t["cle"]
            calc = ""
            if t.get("formule"):
                calc = f'<span class="fo">{_m(t["formule"])}</span>'
            if t.get("valeurs"):
                calc += f'<span class="va">= {_m(t["valeurs"])}</span>'
            cur = cle == "cnom"
            nom = t["nom"].capitalize() if t["nom"].isupper() else t["nom"]
            corps += (f'<tr data-terme="{C.esc(cle)}"{" class=" + chr(34) + "on" + chr(34) if cur else ""} title="{C.esc(nom + " — " + t["regle"])}">'
                      f'<th class="nm"><span class="s">{_m(t["symbole"])}</span></th><td class="nom">{C.esc(nom)}</td>'
                      f'<td class="fc">{calc}</td>'
                      f'<td class="{"cur" if cur else "v"}"><b>{_valeur(t)}</b>{_comparaison(t)}</td>'
                      f'<td class="rg">{_m(t["reference"])}</td></tr>')
    cols = '<colgroup><col class="c-s"><col class="c-n"><col class="c-c"><col class="c-v"><col class="c-r"></colgroup>'
    return f'<table class="bib-tab bib-fc">{cols}<tbody>{corps}</tbody></table>'


def _notes_calcul(res: Dict) -> str:
    """Sous le dessin : la règle de lecture, les conditions d'emploi du tableau, son état de vérification, ce qui
    n'est pas couvert (replié)."""
    f = res["feu"]
    if not f:
        return ('<div class="bib-notes"><p>Aucune exigence au feu : l\'enrobage est celui de la durabilité. '
                'Choisir une durée R pour lire la méthode tabulée de l\'EN 1992-1-2.</p></div>')
    quoi = "formule (5.7)" if f["methode"] == "formule" else f"tableau {f['tableau']}"
    mentions = "".join(f'<div class="bib-av">{C.icon("info")}<span>{_m(a["message"])}</span></div>' for a in res["avertissements"] if a["code"] == "feu_mention")
    cond = "".join(f"<li>{_m(c)}</li>" for c in f["conditions"])
    nc = "".join(f"<li>{_m(c)}</li>" for c in f["non_couvert"])
    return (f'<div class="bib-notes">{mentions}'
            f'<p class="bib-verif">{C.icon("check")}<span>{C.esc(quoi[0].upper() + quoi[1:])} · état de vérification : {C.esc(f["verification"])}</span></p>'
            f'<details class="bib-cond"><summary>Conditions d\'emploi à vérifier ({len(f["conditions"])})</summary><ul>{cond}</ul></details>'
            f'<details><summary>Non couvert ({len(f["non_couvert"])})</summary><ul>{nc}</ul></details></div>')


def _calcul(res: Dict) -> str:
    el = res["entree"]["element"]
    return (f'<div class="bib-calc"><figure class="bib-calc-d"><div class="bib-calc-s">{D.detail("bib-enr-detail", res)}</div>'
            f'<figcaption>{C.esc(D.TITRES[el])} · cotes en mm</figcaption>{_notes_calcul(res)}</figure>'
            f'<div class="bib-calc-f">{_feuille(res)}</div></div>')


# ================================================================================================ l'onglet « Tableaux »
def _tab_durabilite(res: Dict, expo: str) -> tuple:
    s = res["durabilite"]["classe_structurale"]
    t = EN.tableau_c_min_dur()
    col = EN.colonne(expo)
    j = t["colonnes"].index(col)
    th = ('<tr><th class="nm">Classe structurale</th>' + "".join(
        f'<th class="{"on" if c == col else ""}"><span class="s">{C.esc(c)}</span><span class="u">{C.esc(t["belges"][i])}</span></th>' for i, c in enumerate(t["colonnes"])) + "</tr>")
    corps = ""
    for srow in t["lignes"]:
        on = ' class="on"' if srow == s["retenue"] else ""
        cases = "".join(B._td(str(v), {"classe_structurale": srow, "exposition": [e[0] for e in EN.EXPOSITIONS if e[1] == c][0] if c != col else expo},
                              "cur" if (srow == s["retenue"] and c == col) else ("col" if c == col else ""))
                        for c, v in zip(t["colonnes"], t["cellules"][srow]))
        corps += f'<tr{on}><th class="nm">{srow}{" <small>calculée</small>" if (srow == s["calculee"] and not s["imposee"]) else ""}</th>{cases}</tr>'
    extrait = B._extrait(["S"] + t["colonnes"], [[srow] + [str(v) for v in t["cellules"][srow]] for srow in t["lignes"]],
                         (t["lignes"].index(s["retenue"]), j + 1), "EN 1992-1-1, tableau 4.4N — c_min,dur (mm)", EN.SOURCE_DUR)
    return B._tableau(th, corps, "bib-enr"), extrait


def _sel_case(tf: Dict, colonne: Dict, r: str, comb: Dict, element: str) -> Optional[Dict]:
    """Les paramètres qu'un clic sur une case du tableau du feu règle : la durée, la dimension (la dimension minimale
    de la combinaison : elle l'allume), et ce qui choisit la colonne. Une colonne d'épaisseur seule à côté de colonnes
    de distance (5.8, 5.9) n'est pas cliquable : la distance porte la case."""
    if colonne["sorte"] == "dimension" and any(c["sorte"] != "dimension" for c in tf["colonnes"]):
        return None
    cle_dim = DIMENSIONS[element][0]
    sel = {"feu": r}
    pa = colonne["parametres"]
    if element == "poteau":
        sel.update({"methode": "tableau", "expo_feu": pa["exposition"], "mu_fi": pa["mu_fi"], cle_dim: comb["dimension"]})
    elif element == "voile":
        sel.update({"porteur": pa["porteur"], cle_dim: comb["dimension"]})
        if pa["porteur"]:
            sel.update({"mu_voile": pa["mu_fi"], "faces": pa["faces"]})
    elif element == "poutre":
        sel.update({"appuis": pa["appuis"], cle_dim: comb["dimension"]})
    else:
        hs = tf["lignes"][r]["hs"][0]["dimension"]
        sel.update({"portee": pa.get("portee", "plancher_dalle"), cle_dim: hs})
    return sel


def _texte_comb(comb: Dict, sorte: str) -> tuple:
    """(HTML de la case, texte de l'extrait) : « 53 » en gras et « b ≥ 350 » dessous ; « ** » : au moins 8 barres."""
    rep = comb["repere"]
    marque = f'<i class="mk" title="{C.esc(FEU.NOTE_8_BARRES if rep == "**" else FEU.NOTE_ETOILE)}">{rep}</i>' if rep else ""
    if sorte == "dimension":
        return f"<b>{comb['dimension']}</b>", str(comb["dimension"])
    if sorte == "a":
        return f"<b>{comb['a']}{marque}</b>", f"{comb['a']}{rep}"
    return (f"<b>{comb['a']}{marque}</b><small>≥{NBSP}{comb['dimension']}</small>", f"{comb['dimension']}/{comb['a']}{rep}")


def _tab_feu(res: Dict, moteur: Dict) -> tuple:
    element = moteur["element"]
    tf = EN.tableau_feu(element, **moteur["_selection"])
    f = res["feu"]
    sd = C.symb_math(tf["symbole_dimension"] + "_min" if tf["symbole_dimension"] != "h_s" else "h_s")
    # chaque colonne du tableau porte autant de cases que sa plus longue liste de combinaisons (5.2a : deux)
    larg = {c["cle"]: max(1, max(len(tf["lignes"][r][c["cle"]]) for r in tf["durees"])) for c in tf["colonnes"]}
    multi = any(larg[c["cle"]] > 1 for c in tf["colonnes"])
    allume = None
    compare = None
    if f and f["methode"] == "tableau" and f["tableau"] == tf["numero"] and f["indice"] is not None:
        col = f["colonne"] if tf["numero"] not in ("5.5", "5.6") else f"c{f['indice'] + 1}"
        allume = (f["r"], col, 0 if tf["numero"] in ("5.5", "5.6") else f["indice"])
    elif f and f["methode"] == "formule":
        cp = f["comparaison"]
        tab = FEU.exigence("poteau", f["r"], methode="tableau", exposition="plusieurs", b=f["formule"]["b"], mu_fi=cp["mu_fi"])
        if tab["indice"] is not None:
            compare = (f["r"], tab["colonne"], tab["indice"])
    # l'en-tête : le titre de chaque colonne (et ce qu'elle porte), puis la dimension / a
    unite = {"combinaison": f"{sd} / a (mm)", "dimension": f"{sd} (mm)", "a": "a (mm)"}
    h1 = '<tr class="r1"><th class="nm" rowspan="2">Durée</th>' if multi else '<tr><th class="nm">Durée</th>'
    h2 = ""
    poutre = tf["numero"] in ("5.5", "5.6")
    for c in tf["colonnes"]:
        on = " on" if (f and f["methode"] == "tableau" and not poutre and f.get("colonne") == c["cle"]) else ""
        if poutre:
            titre = f'<b>{c["titre"]}</b><small>{unite["combinaison"]}</small>'
        else:
            titre = f'<b>{C.symb_math(c["titre"])}</b><small>{C.esc(c["sous"])}</small>'
        if multi:
            h1 += f'<th class="k{on}" colspan="{larg[c["cle"]]}">{titre}</th>'
            h2 += "".join(f'<th class="k"><small>{unite[c["sorte"]]}</small></th>' for _ in range(larg[c["cle"]]))
        else:
            h1 += f'<th class="k{on}">{titre}{"" if poutre else "<small>" + unite[c["sorte"]] + "</small>"}</th>'
    thead = h1 + "</tr>" + (f'<tr class="r2">{h2}</tr>' if multi else "")
    corps = ""
    lignes_ex, case_ex = [], None
    for i, r in enumerate(tf["durees"]):
        on = ' class="on"' if (f and f["r"] == r) else ""
        cases = ""
        ligne_ex = [tf["libelles"][r]]
        for c in tf["colonnes"]:
            combs = tf["lignes"][r][c["cle"]]
            textes = []
            for k in range(larg[c["cle"]]):
                if k >= len(combs):
                    vide = "aucune valeur : méthode B (tableau 5.2b), annexe C ou calcul" if not combs else ""
                    cases += f'<td class="vide"{(" title=" + chr(34) + C.esc(vide) + chr(34)) if vide else ""}>{"—" if not combs and k == 0 else ""}</td>'
                    continue
                comb = combs[k]
                html_, tx = _texte_comb(comb, c["sorte"])
                textes.append(tx)
                cls = ""
                if allume == (r, c["cle"], k):
                    cls = "cur" + (" ko" if f["atteinte"] is False else "")
                elif compare == (r, c["cle"], k):
                    cls = "cmp"
                elif c["sorte"] == "dimension" and f and f["r"] == r and len(tf["colonnes"]) > 1:
                    cls = "col"
                sel = _sel_case(tf, c, r, comb, moteur["element"])
                cases += B._td(html_, sel, cls)
            ligne_ex.append(" · ".join(textes) or "—")
            if allume and allume[0] == r and allume[1] == c["cle"]:
                case_ex = (i, len(ligne_ex) - 1)
        corps += f'<tr{on}><th class="nm">{tf["libelles"][r].replace(" ", NBSP)}</th>{cases}</tr>'
    tableau = B._tableau(thead, corps, "bib-feu" + (" bib-petit" if len(tf["colonnes"]) <= 2 else ""))
    entetes = ["Durée"] + [f"{c['titre']} — {c['sous']}" for c in tf["colonnes"]]
    extrait = B._extrait(entetes, lignes_ex_from(tf, larg), case_ex, f"{FEU.NORME}, tableau {tf['numero']} — {tf['titre']} (mm)", tf["source"])
    notes = "".join(f"<li>{_m(n)}</li>" for n in tf["notes"])
    cond = "".join(f"<li>{_m(n)}</li>" for n in tf["conditions"])
    pied = (f'<div class="bib-notes bib-notes-t"><p class="bib-verif">{C.icon("check")}<span>Tableau {tf["numero"]} · {C.esc(tf["titre"])} · '
            f'état de vérification : {C.esc(tf["verification"])}</span></p>'
            f'<details><summary>Notes du tableau ({len(tf["notes"])})</summary><ul>{notes}</ul></details>'
            f'<details class="bib-cond"><summary>Conditions d\'emploi à vérifier ({len(tf["conditions"])})</summary><ul>{cond}</ul></details></div>')
    pied = "".join(f'<div class="bib-av">{C.icon("info")}<span>{_m(a["message"])}</span></div>' for a in res["avertissements"]
                   if a["code"] == "feu_mention" and f and f["tableau"] in (tf["numero"], "5.7")) + pied
    if f and f["methode"] == "formule":
        pied = (f'<div class="bib-av">{C.icon("info")}<span>Méthode retenue : formule (5.7) — a ≥ {f["formule"]["a"]}{NBSP}mm. '
                f'La case marquée en pointillé est la lecture du tableau 5.2a pour comparaison : {C.esc(f["comparaison"]["phrase"])}.</span></div>') + pied
    return tableau + pied, extrait


def lignes_ex_from(tf: Dict, larg: Dict) -> List[List[str]]:
    """Les lignes de l'extrait du tableau du feu (une case par colonne de la norme, ses combinaisons jointes)."""
    out = []
    for r in tf["durees"]:
        ligne = [tf["libelles"][r]]
        for c in tf["colonnes"]:
            ligne.append(" · ".join(_texte_comb(x, c["sorte"])[1] for x in tf["lignes"][r][c["cle"]]) or "—")
        out.append(ligne)
    return out


# ================================================================================================ la rubrique
def rendre(params: Optional[Dict], hote: str) -> Dict:
    L, moteur, onglet, vue = lire(params)
    res = calculer(L, moteur)
    p = L.out
    element = p["element"]
    f = res["feu"]
    barre = _barre(p, moteur, res, onglet, vue)
    t_dur, ex_dur = _tab_durabilite(res, p["exposition"])
    if p["feu"]:
        t_feu, ex_feu = _tab_feu(res, moteur)
    else:
        t_feu, ex_feu = None, None
    if onglet == "calcul":
        tableau = _calcul(res)
    elif vue == "durabilite":
        tableau = t_dur
    else:
        tableau = t_feu if t_feu is not None else _tab_feu(res, moteur)[0]
    extrait = ex_feu if ex_feu is not None else ex_dur
    avert_tous = [a["message"] for a in res["avertissements"]]
    # la bande : ce qui se lit à côté de la valeur ; les conditions d'emploi sont sous le dessin et sous le tableau
    avert = [C.esc(a["message"]) for a in res["avertissements"] if a["code"] not in A_SOUS_DESSIN]
    sources = [EN.SOURCE_DUR] + ([f["source"]] if f else []) + [EN.NOTE_XA]
    gouv = EN.GOUVERNANTS[res["gouvernant"]]
    s = res["durabilite"]["classe_structurale"]
    ext = res["phi_ext"]
    sym_ext = "étrier" if element in EN.ELEMENTS_A_ETRIER else "nappe ext."
    cle_dim = DIMENSIONS[element][0]
    phrase = (f"{NOMS[element]} {C.esc(cle_dim)}{NBSP}={NBSP}{p[cle_dim]}{NBSP}mm · {p['exposition']} ({EN.LIBELLES_BELGES[EN.colonne(p['exposition'])]}) · "
              f"{p['classe']} · {p['duree']}{NBSP}ans · {s['retenue']} · Ø{NBSP}{p['phi']}" + (f" / {sym_ext} {ext}" if ext else "")
              + f" · Δc<sub>dev</sub>{NBSP}{p['delta_dev']}" + (f" · {C.esc(f['duree'])}" if f else "") + f" — {gouv}")
    nom = NOMS[element].lower()
    return {"titre": "Enrobages et feu", "barre": barre, "tableau": tableau,
            "bande": B._bande(_fr(res["c_nom"]), "mm", "c_nom", "enrobage nominal", phrase, hote, avert),
            "pourquoi": B._pourquoi(f"Enrobage nominal · {NOMS[element]} {p['exposition']} {p['classe']}",
                                    "EN 1992-1-1 §4.4.1 ; EN 1992-1-2 §5 (feu)", res["termes"], sources, [C.esc(a) for a in avert_tous], nd=1),
            "sources": B._sources(*sources),
            "fiche": B._fiche(f"Enrobage {nom} {p['exposition']} {p['classe']}" + (f" {f['duree']}" if f else ""), "c_nom", res["c_nom"], "mm", p,
                              res["termes"], sources, extrait, avert_tous, texte=_fr(res["c_nom"]), phrase=phrase, nd=1),
            "params": p, "message": L.message()}

