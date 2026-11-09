# -*- coding: utf-8 -*-
"""La BIBLIOTHÈQUE ACIER du site (docs/site/BIBLIOTHEQUES.md § 2.3, lot BIB5) : le gabarit de la Bibliothèque Béton
armé — le menu des rubriques à gauche ; la barre de paramètres, LE TABLEAU, la bande de la valeur retenue et les
sources à droite ; la fiche (« Pourquoi ? ») dans un volet ; l'adresse ``?rubrique=…&…`` qui garde l'état — et trois
rubriques :

- « Profilés » : le CATALOGUE DES PROFILÉS (familles, recherche, séries courantes, filtres numériques, tri, colonnes,
  lignes ; le profilé ouvert : sa vignette et ses propriétés dans la bande, sa section dessinée à l'échelle, toutes
  ses propriétés et leurs formules dans la fiche). RIEN n'y est refait : la liste, les colonnes visibles, les valeurs
  écrites, le plus léger, les raisons d'un tableau vide, les formules sont celles du module « Catalogue des profilés »
  (api_web/catalogue.py, qui les tient de acier/profiles/consultation.py, extrait mot pour mot de l'écran Streamlit) ;
  les nombres viennent de acier/donnees/profiles.json ; le dessin est celui du catalogue (api_web/dessin_catalogue.py).
- « Nuances » : les aciers de construction tels que les moteurs d'assemblage, de soudure et de garde-corps les lisent
  (acier/donnees/bibliotheques.json : f_y, f_u, β_w) ; f_y au-delà de 40 mm, ε et E par le moteur du flambement
  (acier/flambement/normes.py, tableau 3.1) ; f_vw,d par acier/resistances.py.
- « Boulons » : les boulons et leurs classes tels que les assemblages les lisent (bibliotheques.json : d, d_0, A, A_s,
  d_m, d_w ; f_yb, f_ub, α_v) ; F_t,Rd et F_v,Rd par acier/resistances.py (les fonctions des moteurs d'assemblage).

Aucune valeur n'est écrite ici (R2, R3) : ce fichier CHOISIT parmi ce que les moteurs nomment et MET EN FORME. Les
briques HTML et la relecture des paramètres sont celles de la Bibliothèque Béton armé (api_web/biblio_beton.py : une
source). Comme elle, ce fichier n'est pas un module du registre (ni MODULE : une consultation n'est pas une note) ; le
routeur l'appelle par l'action « consulter » (api_web/routeur.py, BIBLIOTHEQUES), la page bibliotheque-acier
(site/web/construire.py, page_bibliotheque) le pose par js/bibliotheque.js et js/biblio.js.

    consulter(rubrique, params, hote="page") → {rubrique, titre, barre, tableau, bande, pourquoi, sources, fiche,
                                                  params, message, bibliotheque}

Les paramètres illisibles reprennent leur défaut et ``message`` le dit ; tout texte venu des paramètres (la recherche,
un nom de profilé) est échappé.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from acier import bibliotheques as BQ
from acier import resistances as RS
from acier.flambement import normes as NO
from acier.profiles import catalogue as CAT
from acier.profiles import consultation as CO
from acier.profiles import familles as F
from acier.soudures import criteres as CR

from . import biblio_beton as BB
from . import catalogue as CW
from . import commun as C
from . import dessin_catalogue as D

BIBLIOTHEQUE = "acier"
TITRE = "Bibliothèque Acier"
PAGE = "bibliotheque-acier"
VERSION = "2026-11-10"       # la date des données de la bibliothèque : une fiche figée dit quel moteur l'a rendue
# (clé, titre du menu, sous-titre du menu, icône)
RUBRIQUES = (
    ("profils", "Profilés", "Le catalogue des profilés", "list-filter"),
    ("nuances", "Nuances", "Aciers de construction", "sliders-horizontal"),
    ("boulons", "Boulons", "Dimensions et classes", "settings"),
)
CLES = tuple(r[0] for r in RUBRIQUES)
NOMS = {r[0]: r[1] for r in RUBRIQUES}
RUBRIQUE_DEFAUT = "profils"
HOTES = BB.HOTES
NBSP = C.NBSP
Lecture = BB.Lecture


def _fiche(titre: str, symbole: str, valeur, unite: str, params: Dict, termes: Sequence[Dict], sources: Sequence[str],
           avertissements: Sequence[str] = (), texte: str = "", phrase: str = "", nd: int = 2) -> Dict:
    """La fiche figée de la rubrique (le même format que celle de la Bibliothèque Béton armé, api_web/fiches.py la
    relirait), signée par CETTE bibliothèque."""
    f = BB._fiche(titre, symbole, valeur, unite, params, termes, sources, None, avertissements, texte=texte, phrase=phrase, nd=nd)
    f["moteur"] = f"{TITRE} {VERSION}"
    return f


def _bande(valeur: str, unite: str, symbole_html: str, nom: str, phrase: str, avert: Sequence[str] = (), devant: str = "",
           apres: str = "", bouton: Tuple[str, str] = ("lightbulb", "Pourquoi ?")) -> str:
    """La bande sous le tableau (la forme de celle du béton) : la valeur retenue en grand, son symbole et son nom, la
    phrase, les avertissements, le bouton qui ouvre la fiche. ``devant`` : ce qui précède la valeur (la vignette d'un
    profilé) ; ``apres`` : une ligne sous la valeur (le plus léger qui satisfait les filtres). Une consultation de
    l'acier n'écrit rien dans une note : ni « Inclure », ni « Utiliser cette valeur »."""
    av = "".join(f'<div class="bib-av">{C.icon("triangle-alert")}<span>{a}</span></div>' for a in avert)
    ic, lib = bouton
    return (f'<div class="bib-bande-v">{devant}<b class="bib-val">{valeur}<small>{unite}</small></b>'
            f'<div class="bib-bande-t"><span class="bib-sym">{symbole_html}</span><span class="bib-nom">{nom}</span>'
            f'<span class="bib-phr">{phrase}</span></div></div>'
            f'<div class="bib-bande-b"><button type="button" class="btn" data-pourquoi="1">{C.icon(ic)}<span>{lib}</span></button></div>{apres}{av}')


def _bloc(libelle: str, controle: str, aide: str = "") -> str:
    """Un paramètre fait de PLUSIEURS boutons (pastilles, segments) : le paramètre de la barre du béton
    (``BB._champ``), dans un ``<div>`` — un ``<label>`` renverrait un clic sur son libellé (ou entre deux pastilles) au
    PREMIER bouton qu'il contient, qui changerait la valeur sans qu'on l'ait visé."""
    h = BB._champ("", libelle, controle, aide)
    assert h.startswith("<label") and h.endswith("</label>")
    return "<div" + h[len("<label"):-len("</label>")] + "</div>"


def _tete_fiche(titre: str, sous: str) -> str:
    return (f'<div class="bib-pq-h"><div><b>{titre}</b><span>{sous}</span></div>'
            f'<button type="button" class="btn icon ghost" data-fermer-pourquoi="1" aria-label="Fermer">{C.icon("x")}</button></div>')


# ================================================================================================ 1 · profilés
RECHERCHE_MAX = 60
AUCUNE = "aucune"                      # « familles = aucune » : aucune famille cochée (un vide reprendrait le défaut)
BORNE = 1e12
ORDRES = (("croissant", "↑ croissant"), ("decroissant", "↓ décroissant"))
GROUPES = CW.GROUPES_FAMILLES          # les noms des groupes de familles : ceux du catalogue
# les propriétés que la bande résume (celles du sélecteur du catalogue, CW._prop)
BANDE_PROPS = ("A", "Iy", "Wpl_y", "Iz")


def _liste_txt(v) -> Optional[List[str]]:
    """Une liste écrite comme l'adresse la porte (« IPE,HEA ») ; une vraie liste passe ; None si illisible."""
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return None


def _joint(xs: Sequence[str]) -> str:
    return ",".join(xs)


def _lire_profils(params: Optional[Dict]) -> Tuple[Lecture, Dict]:
    """Les paramètres de la rubrique « Profilés » relus — la recherche du catalogue (``r``, la forme que
    api_web/catalogue.py attend) et ``L.out`` (ce que l'adresse garde)."""
    L = Lecture(params)
    b = L.brut
    defaut = CO.reglages_initiaux()
    toutes = CO.toutes_familles()
    # les familles
    v = b.get("familles")
    if v is None or (isinstance(v, str) and not v.strip()):
        fams = list(defaut["familles"])
    elif v == AUCUNE:
        fams = []
    else:
        xs = _liste_txt(v)
        if xs is None:
            L._dit("familles", "ignoré : liste illisible")
            xs = list(defaut["familles"])
        inconnues = [x for x in xs if x not in toutes]
        if inconnues:
            L._dit("familles", f"ignoré en partie : « {', '.join(inconnues)[:40]} » n'est pas une famille du catalogue")
        fams = CO.familles_ordonnees([x for x in xs if x in toutes])
    L.out["familles"] = _joint(fams) if fams else AUCUNE
    # la recherche : un TEXTE (échappé partout où il est écrit), borné
    rech = b.get("recherche", "")
    rech = "" if rech is None else str(rech)
    if len(rech) > RECHERCHE_MAX:
        L._dit("recherche", f"coupée à {RECHERCHE_MAX} caractères")
        rech = rech[:RECHERCHE_MAX]
    rech = "".join(ch for ch in rech if ch.isprintable())
    L.out["recherche"] = rech
    courants = L.case("courants", defaut["courants"])
    tri = L.choix("tri", [""] + CO.cles_colonnes(), defaut["tri"])
    ordre = L.choix("ordre", tuple(k for k, _ in ORDRES), "decroissant" if defaut["decroissant"] else "croissant")
    # les colonnes
    v = b.get("colonnes")
    if v is None or (isinstance(v, str) and not v.strip()):
        cols = list(defaut["colonnes"])
    elif v == AUCUNE:
        cols = []
    else:
        xs = _liste_txt(v)
        if xs is None:
            L._dit("colonnes", "ignoré : liste illisible")
            xs = list(defaut["colonnes"])
        inconnues = [x for x in xs if x not in CAT.LIBELLE]
        if inconnues:
            L._dit("colonnes", f"ignoré en partie : « {', '.join(inconnues)[:40]} » n'est pas une propriété du catalogue")
        cols = CO.colonnes_ordonnees([x for x in xs if x in CAT.LIBELLE])
    L.out["colonnes"] = _joint(cols) if cols else AUCUNE
    lignes = L.choix("lignes", CW.LIGNES, CW.LIGNES_DEFAUT)
    # les filtres numériques : f1…f3 (la propriété), min1…, max1… ; « filtres = effacer » les vide tous. Une propriété
    # ne se filtre qu'une fois (le catalogue refuse la seconde) ; les filtres posés passent devant, sans trou.
    effacer = b.get("filtres") == "effacer"
    poses: List[Dict] = []
    if not effacer:
        for i in range(1, CO.N_FILTRES + 1):
            cle = L.choix(f"f{i}", [""] + CO.cles_colonnes(), "")
            mini = L.nombre(f"min{i}", None, -BORNE, BORNE, vide=None)
            maxi = L.nombre(f"max{i}", None, -BORNE, BORNE, vide=None)
            if not cle:
                continue
            if any(f["cle"] == cle for f in poses):
                L._dit(f"f{i}", f"ignoré : {CO.entete(cle)} est déjà filtrée (une propriété ne se filtre qu'une fois)")
                continue
            poses.append({"cle": cle, "min": mini, "max": maxi})
    filtres = poses + [{"cle": "", "min": None, "max": None} for _ in range(CO.N_FILTRES - len(poses))]
    for i in range(1, CO.N_FILTRES + 1):
        for k in ("f", "min", "max"):
            L.out.pop(f"{k}{i}", None)
    for i, f in enumerate(filtres, 1):
        L.out[f"f{i}"] = f["cle"]
        L.out[f"min{i}"] = f["min"]
        L.out[f"max{i}"] = f["max"]
    L.out["filtres"] = ""
    # le profilé ouvert : un nom du catalogue (un nom inconnu est dit ; un profilé que les filtres écartent laisse la
    # place au premier de la liste, sans message — la règle du catalogue)
    nom = b.get("profil")
    profil = None
    if isinstance(nom, str) and nom.strip():
        if nom in CAT.par_nom():
            profil = nom
        else:
            L._dit("profil", f"ignoré : « {nom[:40]} » n'est pas un profilé du catalogue")
    r = {"familles": fams, "recherche": rech, "filtres": filtres, "courants": courants, "persos": [], "tri": tri,
         "decroissant": ordre == "decroissant", "colonnes": cols, "fiche": profil, "compare": [], "lignes": lignes}
    return L, r


def _puce(texte: str, valeur: str, on: bool, titre: str = "", petit: str = "") -> str:
    """Une pastille : ``texte`` (du HTML : un symbole et ses indices tiennent dans UN élément — la pastille aligne ses
    enfants avec un écart) et ``petit`` (le nombre de profilés, l'unité) ; ``valeur`` : ce qu'un clic règle."""
    t = f' title="{C.esc(titre)}"' if titre else ""
    p = f"<small>{petit}</small>" if petit else ""
    return (f'<button type="button" class="fchip{" on" if on else ""}" data-valeur="{C.esc(valeur)}" aria-pressed="{"true" if on else "false"}"{t}>'
            f'<span class="bib-pt">{texte}</span>{p}</button>')


def _puces_familles(fams: Sequence[str]) -> str:
    """Les familles en pastilles, groupe par groupe (celles du catalogue, avec leur nombre de profilés) : chacune porte
    la liste qu'un clic donne (la sienne ajoutée ou retirée) ; « Toutes », « Aucune »."""
    comptes = dict(CAT.familles_presentes())
    sel = set(fams)
    out = ""
    for g, membres in CAT.groupes_presents():
        bt = ""
        for f in membres:
            apres = CO.familles_ordonnees(sel ^ {f})
            bt += _puce(C.esc(f), _joint(apres) or AUCUNE, f in sel, F.FAMILLES[f]["libelle"], str(comptes.get(f, 0)))
        out += f'<span class="bib-grp"><span class="bib-grp-t">{C.esc(GROUPES.get(g, g))}</span>{bt}</span>'
    out += (f'<span class="bib-grp bib-grp-a">{_puce("Toutes", _joint(CO.toutes_familles()), False)}'
            f'{_puce("Aucune", AUCUNE, False)}</span>')
    return f'<div class="bib-puces" data-param="familles">{out}</div>'


def _texte(cle: str, valeur: str, vide: str = "") -> str:
    return (f'<div class="field text bib-txt">{C.icon("search")}<input type="search" data-param="{C.esc(cle)}" value="{C.esc(valeur)}" '
            f'placeholder="{C.esc(vide)}" maxlength="{RECHERCHE_MAX}" spellcheck="false" autocomplete="off" aria-label="Rechercher un profilé"></div>')


def _borne(cle: str, valeur: Optional[float], unite: str) -> str:
    v = "" if valeur is None else C.nb(valeur, 6)
    u = f'<span class="u">{C.esc(unite)}</span>' if unite else ""
    return f'<div class="field bib-borne"><input data-param="{C.esc(cle)}" inputmode="decimal" value="{C.esc(v)}" placeholder="—">{u}</div>'


def _colonnes(cols: Sequence[str]) -> str:
    """« Colonnes » : un menu déroulant (il reste ouvert d'une réponse à l'autre : js/biblio.js, data-garde), les
    propriétés en pastilles groupe par groupe (celles du catalogue), chacune avec la liste qu'un clic donne."""
    sel = set(cols)
    corps = ""
    for g in CAT.GROUPES_COLONNES:
        bt = "".join(_puce(CW.symbole(c), _joint(CO.colonnes_ordonnees(sel ^ {c})) or AUCUNE, c in sel,
                           CO.entete(c) + (" — " + CAT.AIDE_COLONNES[c] if c in CAT.AIDE_COLONNES else ""), C.esc(CAT.UNITE.get(c, "")))
                     for c in CAT.colonnes_du_groupe(g))
        corps += f'<div class="bib-plus-g"><span class="bib-grp-t">{C.esc(g)}</span><div class="bib-plus-p">{bt}</div></div>'
    livrees = CO.reglages_livres()["colonnes"]
    corps += (f'<div class="bib-plus-g bib-plus-a">{_puce("Colonnes livrées", _joint(livrees), False, "les colonnes par défaut du catalogue")}'
              f'{_puce("Aucune", AUCUNE, False)}<span class="bib-plus-n">Les colonnes vides pour les familles affichées sont masquées d\'elles-mêmes.</span></div>')
    return (f'<details class="bib-plus" data-garde="colonnes"><summary class="btn sm">{C.icon("layout-grid")}<span>Colonnes · {len(cols)}/{len(CO.cles_colonnes())}</span>'
            f'{C.icon("chevron-down")}</summary><div class="bib-plus-c" data-param="colonnes">{corps}</div></details>')


def _barre_profils(r: Dict, n: int, montres: int) -> str:
    cles = CO.cles_colonnes()
    opts_tri = [("", "ordre du catalogue")] + [(c, CO.entete(c)) for c in cles]
    l2 = (BB._champ("recherche", "Rechercher", _texte("recherche", r["recherche"], "IPE 200 · 100x100 · HEB…"),
                    "La recherche ignore la casse et les espaces : « ipe200 », « 100x100 », « HEB 3 ».")
          + BB._champ("courants", "", BB._case("courants", r["courants"], "Séries courantes", CO.AIDE_COURANTS))
          + BB._champ("tri", "Trier par", BB._select("tri", r["tri"], opts_tri))
          + (_bloc("Ordre", BB._segments("ordre", "decroissant" if r["decroissant"] else "croissant", ORDRES)) if r["tri"] else "")
          + _bloc(f"Lignes · {montres} sur {n}", BB._segments("lignes", r["lignes"], [(x, str(x) if x else "Toutes") for x in CW.LIGNES]),
                  "le profilé ouvert reste toujours montré")
          + f'<span class="bib-p bib-p-case">{_colonnes(r["colonnes"])}</span>')
    # les filtres : ceux qui sont posés, et une ligne libre tant qu'il en reste
    k = sum(1 for f in r["filtres"] if f["cle"])
    opts = [("", "— propriété —")] + [(c, CO.entete(c)) for c in cles]
    l3 = ""
    for i, f in enumerate(r["filtres"][:min(CO.N_FILTRES, k + 1)], 1):
        u = CAT.UNITE.get(f["cle"], "") if f["cle"] else ""
        l3 += f'<span class="bib-filtre">{BB._champ(f"f{i}", f"Filtre {i}", BB._select(f"f{i}", f["cle"], opts))}'
        if f["cle"]:
            l3 += (BB._champ(f"min{i}", "min ≥", _borne(f"min{i}", f["min"], u))
                   + BB._champ(f"max{i}", "max ≤", _borne(f"max{i}", f["max"], u)))
        l3 += "</span>"
    if k:
        l3 += _bloc("", f'<div class="seg" data-param="filtres"><button type="button" data-valeur="effacer">{C.icon("x")}Effacer les filtres</button></div>')
    # (les réglages et les filtres sur UNE ligne qui passe à la suivante quand la place manque : la barre reste basse)
    return BB._barre(_bloc("Familles", _puces_familles(r["familles"])), l2 + l3)


def _entetes(cols: Sequence[str], tri: str, desc: bool) -> str:
    """Les en-têtes : un clic trie par la colonne (un second clic renverse l'ordre) ; « Profilé » revient à l'ordre du
    catalogue. La définition au survol (celle du catalogue)."""
    th = (f'<th class="nm"{BB._sel({"tri": ""})} title="Profilé — un clic : l\'ordre du catalogue">Profilé</th>'
          f'<th class="fa">Famille</th>')
    for c in cols:
        on = c == tri
        fl = f'<i>{"↓" if desc else "↑"}</i>' if on else ""
        ordre = ("croissant" if desc else "decroissant") if on else "croissant"
        aide = CAT.AIDE_COLONNES.get(c, "")
        titre = CO.entete(c) + (" — " + aide if aide else "") + " — un clic trie par cette colonne"
        th += (f'<th class="n{" on" if on else ""}"{BB._sel({"tri": c, "ordre": ordre})} title="{C.esc(titre)}">'
               f'<span class="s">{CW.symbole(c)}{fl}</span><span class="u">{C.esc(CAT.UNITE.get(c, "")) or "&nbsp;"}</span></th>')
    return f"<thead><tr>{th}</tr></thead>"


def _ligne(p: Dict, ligne: Dict, cols: Sequence[str], tri: str, ouvert: Optional[str], leger: Optional[str]) -> str:
    on = p["nom"] == ouvert
    mk = '<i class="mk" title="le plus léger qui satisfait les filtres">★</i>' if p["nom"] == leger else ""
    tds = "".join(f'<td{" class=" + chr(34) + "tri" + chr(34) if c == tri else ""}>{CW._valeur(ligne[CO.entete(c)], c)}</td>' for c in cols)
    return (f'<tr{" class=" + chr(34) + "on" + chr(34) if on else ""}{BB._sel({"profil": p["nom"]})}>'
            f'<th class="nm{" cur" if on else ""}">{C.esc(p["nom"])}{mk}</th><td class="fa">{C.esc(p["famille"])}</td>{tds}</tr>')


def _tableau_profils(r: Dict, v: Dict) -> Tuple[str, int]:
    """Le tableau des profilés (la liste du catalogue, ses colonnes, la colonne de tri en tête) : au plus ``lignes``
    lignes, le profilé ouvert toujours montré (après « … n lignes … » s'il est plus loin). Rend (html, lignes montrées)."""
    lst = v["liste"]
    if not lst:
        return (f'<div class="bib-vide bib-vide-p">{C.icon("search")}<b>{C.esc(CO.MESSAGE_VIDE)}</b>'
                f'<span>{C.esc(CW._raisons_vide(r))}</span></div>'), 0
    cols = CW.colonnes_affichees(r, v)
    n = len(lst)
    montres = min(r["lignes"] or n, n)
    ouvert = v["p"]["nom"] if v["p"] else None
    i_o = v["noms"].index(ouvert) if ouvert in v["noms"] else -1
    leger = CW._plus_leger(lst)["nom"] if CW._bornes_actives(r) else None
    lignes = CO.lignes_tableau(lst, cols)
    corps = "".join(_ligne(lst[i], lignes[i], cols, r["tri"], ouvert, leger) for i in range(montres))
    if i_o >= montres:
        saut = i_o - montres
        corps += (f'<tr class="saut"><td colspan="{len(cols) + 2}">… {saut} ligne{"s" if saut > 1 else ""} …</td></tr>'
                  + _ligne(lst[i_o], lignes[i_o], cols, r["tri"], ouvert, leger))
    return f'<table class="bib-tab bib-profils">{_entetes(cols, r["tri"], r["decroissant"])}<tbody>{corps}</tbody></table>', montres


def _props_txt(p: Dict) -> str:
    """Les propriétés que la bande résume, écrites comme le catalogue (symbole, valeur, unité), chacune d'un tenant (un
    grand nombre ne se coupe pas à son séparateur de milliers)."""
    return " · ".join(f'<span class="bib-nw">{CW.symbole(c)} = {CAT.formater(p[c], c)}{NBSP}{C.esc(CAT.UNITE.get(c, ""))}</span>'
                      for c in BANDE_PROPS if p.get(c) is not None)


def _fiche_profil(p: Dict, r: Dict) -> str:
    """La fiche du profilé ouvert (le volet) : la section à l'échelle, puis TOUTES ses propriétés groupe par groupe
    (``CO.fiche``, la fiche de l'écran), chacune avec la formule qui la donne (``CW.formules``, les valeurs du profilé
    substituées) ou sa définition ; la méthode de I_t ; les sources."""
    fam = F.FAMILLES[p["famille"]]
    fx = {li["cle"]: li for _t, lignes in CW.formules(p) for li in lignes}
    fil = {f["cle"]: f for f in r["filtres"] if f["cle"] and (f["min"] is not None or f["max"] is not None)}
    lignes = ""
    for g, props in CO.fiche(p):
        lignes += f'<tr class="g"><td colspan="3">{C.esc(g)}</td></tr>'
        for c, _lib, val, unite, aide in props:
            f = fx.get(c)
            if f and len(f["parts"]) > 1:
                regle = " = ".join(C.symb_math(x) for x in f["parts"][1:])
            else:
                regle = C.esc(aide)
            borne = f' <span class="bib-fil">filtre : {CW._resume_filtres([fil[c]])}</span>' if c in fil else ""
            u = f' <span class="u">{C.esc(unite)}</span>' if unite else ""
            lignes += (f'<tr{" class=" + chr(34) + "f" + chr(34) if borne else ""}><td class="s">{CW.symbole(c)}</td>'
                       f'<td class="n">{C.esc(val)}{u}</td><td class="rg">{regle}{borne}</td></tr>')
    notes = ""
    if CO.methode_it(p):
        notes += f'<div class="bib-src">{C.icon("info")}<span>I<sub>t</sub> : {C.esc(CW._methode_txt(CO.methode_it(p)))}.</span></div>'
    sources = _sources_profils(p, complet=True)
    dessin = CW._dessin(lambda: D.section("bib_sec", p), "")
    return (_tete_fiche(f'{C.esc(p["nom"])} · {CW._masse(p["masse"])}{NBSP}kg/m', f'{C.esc(fam["libelle"])} · {C.esc(fam["norme"])}')
            + f'<div class="bib-pq-b"><div class="bib-pq-des">{dessin}</div><div class="bib-pq-leg">{C.esc(D.legende(p))}</div>'
            f'<table class="bib-pq-t bib-fiche"><tbody>{lignes}</tbody></table>{notes}'
            f'<div class="bib-pq-s">{BB._sources(*sources)}</div></div>')


def _sources_profils(p: Optional[Dict], complet: bool = False) -> List[str]:
    """Les sources : sous le tableau, une ligne (le fichier, sa version, la norme des dimensions du profilé ouvert) ;
    dans la fiche (``complet``), le calcul des propriétés et leur contrôle croisé en plus (méta du catalogue)."""
    m = CO.methode_et_sources()
    tete = (f"Catalogue du bureau : {C.esc(m['fichier'])}, version {C.esc(str(m['version']))} du {C.esc(CW._date(m['date']))}, "
            f"{C.fr(m['nombre'], 0)} profilés")
    if p is not None:
        tete += f" · dimensions {C.esc(p['famille'])} : {C.esc(F.FAMILLES[p['famille']]['norme'])}"
    out = [tete]
    if complet:
        out += [f"Propriétés : {C.esc(m['calcul'])}", f"Contrôle : {C.esc(m['controle'])}"]
    return out


def _r_profils(params: Dict, hote: str) -> Dict:
    L, r = _lire_profils(params)
    v = CW._vue(r)
    p, lst = v["p"], v["liste"]
    tableau, montres = _tableau_profils(r, v)
    barre = _barre_profils(r, len(lst), montres)
    if p is not None:
        L.out["profil"] = p["nom"]
    else:
        L.out.pop("profil", None)
    sources = _sources_profils(p)
    if p is None:
        bande = _bande("—", "", "Aucun profilé", C.esc(CO.MESSAGE_VIDE), C.esc(CW._raisons_vide(r)), bouton=("file-text", "Fiche du profilé"))
        bande = bande.replace('data-pourquoi="1"', 'data-pourquoi="1" disabled', 1)
        pourquoi = _tete_fiche("Aucun profilé", C.esc(CO.MESSAGE_VIDE)) + '<div class="bib-pq-b"></div>'
        fiche = _fiche("Aucun profilé", "m", None, "kg/m", L.out, [{"symbole": "—", "nom": CO.MESSAGE_VIDE, "valeur": "—", "unite": "", "regle": CW._raisons_vide(r)}],
                       [BB._texte_brut(s) for s in sources], texte="—")
        return {"titre": "Profilés", "barre": barre, "tableau": tableau, "bande": bande, "pourquoi": pourquoi,
                "sources": BB._sources(*sources), "fiche": fiche, "params": L.out, "message": L.message()}
    fam = F.FAMILLES[p["famille"]]
    serie = "série courante" if p.get("courant") else "série peu courante"
    vig = f'<span class="bib-vig" aria-hidden="true">{CW._dessin(lambda: D.vignette("bib_vig", p), "")}</span>'
    # le plus léger de la liste qui satisfait les filtres (★), quand une borne est posée : on CHOISIT parmi les masses
    apres = ""
    if CW._bornes_actives(r):
        leger = CW._plus_leger(lst)
        if leger["nom"] == p["nom"]:
            apres = f'<div class="bib-leger"><i class="mk">★</i><span>{C.esc(p["nom"])} est le plus léger qui satisfait les filtres.</span></div>'
        else:
            apres = (f'<div class="bib-leger"><i class="mk">★</i><span>Le plus léger qui satisfait les filtres : <b>{C.esc(leger["nom"])}</b>'
                     f'{NBSP}·{NBSP}{CW._masse(leger["masse"])}{NBSP}kg/m</span>'
                     f'<button type="button" class="btn sm"{BB._sel({"profil": leger["nom"]})}>Ouvrir {C.esc(leger["nom"])}</button></div>')
    phrase = f"{C.esc(D.legende(p))} — {_props_txt(p)}"
    bande = _bande(C.esc(p["nom"]), f'{CW._masse(p["masse"])}{NBSP}kg/m', C.esc(fam["libelle"]), f'{C.esc(fam["norme"])} · {serie}', phrase,
                   devant=vig, apres=apres, bouton=("file-text", "Fiche du profilé"))
    termes = [{"symbole": BB._texte_brut(CW.symbole(c)), "nom": CO.entete(c), "valeur": p[c], "unite": u, "regle": "catalogue du bureau (acier/donnees/profiles.json)"}
              for _g, props in CO.fiche(p) for c, _lib, _val, u, _a in props]
    fiche = _fiche(f"Profilé {p['nom']}", "m", p["masse"], "kg/m", L.out, termes, [BB._texte_brut(s) for s in sources],
                   texte=CW._masse(p["masse"]), phrase=phrase)
    return {"titre": "Profilés", "barre": barre, "tableau": tableau, "bande": bande, "pourquoi": _fiche_profil(p, r),
            "sources": BB._sources(*sources), "fiche": fiche, "params": L.out, "message": L.message()}


# ================================================================================================ 2 · nuances
EPAISSEURS = (("40", "t ≤ 40 mm"), ("80", "40 < t ≤ 80 mm"))
SOURCE_BQ = ("acier/donnees/bibliotheques.json — la bibliothèque des moteurs d'assemblage, de soudure et de garde-corps "
             "(extraite du corrigé de référence, acier/bibliotheques.py)")
SOURCE_NO = "EN 1993-1-1 tableau 3.1 et §3.2.6 — acier/flambement/normes.py (f_y au-delà de 40 mm, E, ε)"
SOURCE_SOUDURE = "EN 1993-1-8 §4.5.3.3 — f_vw,d = f_u / (√3 · β_w · γ_M2), acier/resistances.py ; γ_M2 de acier/soudures/criteres.py"


def _symboles(sources: Sequence[str]) -> List[str]:
    """Les lignes de source (du texte : « f_vw,d », « γ_M2 ») écrites avec leurs indices."""
    return [C.symb_math(s) for s in sources]


def _base(nom: str) -> str:
    """La nuance de base d'un acier de la bibliothèque (« S355 N/NL » → « S355 ») : la clé du tableau 3.1 du moteur du
    flambement."""
    return nom.split()[0]


def _fy_epais(nom: str) -> Optional[float]:
    """f_y pour 40 < t ≤ 80 mm, par le moteur du flambement (None si la nuance n'y est pas)."""
    try:
        return NO.limite_elastique(_base(nom), 80.0)
    except NO.HorsDomaine:
        return None


def _r_nuances(params: Dict, hote: str) -> Dict:
    L = Lecture(params)
    aciers = BQ.DB["aciers"]
    noms = [a["n"] for a in aciers]
    nuance = L.choix("nuance", noms, "S355")
    ep = L.choix("epaisseur", tuple(k for k, _ in EPAISSEURS), "40")
    p = L.out
    a = BQ.find(aciers, nuance)
    fy = float(a["fy"]) if ep == "40" else _fy_epais(nuance)
    barre = BB._barre(BB._champ("nuance", "Nuance", BB._select("nuance", nuance, [(n, n) for n in noms]))
                      + _bloc("Épaisseur", BB._segments("epaisseur", ep, EPAISSEURS),
                              "f_y selon l'épaisseur nominale de la paroi (EN 1993-1-1, tableau 3.1)"))
    th = ('<tr><th class="nm">Nuance</th>' + BB._th("f_y", "MPa · t ≤ 40", "t ≤ 40 mm — bibliotheques.json")
          + BB._th("f_y", "MPa · 40 < t ≤ 80", "40 < t ≤ 80 mm — tableau 3.1, acier/flambement/normes.py")
          + BB._th("f_u", "MPa", "t ≤ 40 mm — bibliotheques.json") + BB._th("β_w", "", "facteur de corrélation des soudures d'angle — bibliotheques.json")
          + BB._th("ε", "", "√(235 / f_y), t ≤ 40 mm — acier/flambement/normes.py") + "</tr>")
    corps = ""
    for x in aciers:
        n = x["n"]
        on = ' class="on"' if n == nuance else ""
        f80 = _fy_epais(n)
        corps += (f'<tr{on}><th class="nm">{C.esc(n)}</th>'
                  f'{BB._td(C.fr(x["fy"], 0), {"nuance": n, "epaisseur": "40"}, "cur" if (n == nuance and ep == "40") else ("col" if ep == "40" else ""))}'
                  + (BB._td(C.fr(f80, 0), {"nuance": n, "epaisseur": "80"}, "cur" if (n == nuance and ep == "80") else ("col" if ep == "80" else ""))
                     if f80 is not None else BB._td("—", {"nuance": n}, "vide", "nuance absente du tableau 3.1 du moteur"))
                  + f'{BB._td(C.fr(x["fu"], 0), {"nuance": n})}{BB._td(C.fr(x["bw"], 2), {"nuance": n})}'
                  f'{BB._td(C.fr(NO.epsilon(float(x["fy"])), 2), {"nuance": n})}</tr>')
    tableau = BB._tableau(th, corps, "bib-nuances")
    avert = []
    if ep == "80":
        avert.append("f<sub>u</sub> n'est connu du dépôt que pour t ≤ 40 mm (bibliotheques.json) : au-delà de 40 mm, le tableau 3.1 "
                     "de l'EN 1993-1-1 donne pour certaines nuances une valeur plus faible — à lire dans la norme.")
    if fy is None:
        avert.append(f"{C.esc(nuance)} : f<sub>y</sub> au-delà de 40 mm n'est pas dans le tableau 3.1 du moteur.")
    termes = [
        {"symbole": "f_y", "nom": "LIMITE D'ÉLASTICITÉ", "valeur": fy if fy is not None else "—", "unite": "MPa",
         "regle": ("t ≤ 40 mm — bibliotheques.json" if ep == "40" else "40 < t ≤ 80 mm — EN 1993-1-1 tableau 3.1, acier/flambement/normes.limite_elastique")},
        {"symbole": "f_u", "nom": "résistance ultime à la traction (t ≤ 40 mm)", "valeur": float(a["fu"]), "unite": "MPa", "regle": "bibliotheques.json"},
        {"symbole": "β_w", "nom": "facteur de corrélation des soudures d'angle", "valeur": float(a["bw"]), "unite": "", "regle": "bibliotheques.json"},
        {"symbole": "E", "nom": "module d'élasticité", "valeur": NO.E_ACIER, "unite": "MPa", "regle": "EN 1993-1-1 §3.2.6 — acier/flambement/normes.py"},
    ]
    if fy is not None:
        termes.append({"symbole": "ε", "nom": "coefficient des classes de section", "valeur": NO.epsilon(fy), "unite": "", "regle": f"√(235 / f_y) = √(235 / {C.fr(fy, 0)}) — tableau 5.2"})
    termes.append({"symbole": "γ_M2", "nom": "coefficient partiel des assemblages", "valeur": CR.GAMMA_M2, "unite": "", "regle": "ANB — acier/soudures/criteres.py"})
    fvw = RS.fvw_d(float(a["fu"]), float(a["bw"]), CR.GAMMA_M2)
    termes.append({"symbole": "f_vw,d", "nom": "résistance de calcul au cisaillement d'un cordon", "valeur": fvw, "unite": "MPa",
                   "regle": f"f_u / (√3 · β_w · γ_M2) = {C.fr(a['fu'], 0)} / (√3 · {C.nb(a['bw'], 2)} · {C.nb(CR.GAMMA_M2, 2)}) — EN 1993-1-8 §4.5.3.3"})
    sources = [SOURCE_BQ, SOURCE_NO, SOURCE_SOUDURE]
    phrase = (f"{C.esc(nuance)} · {dict(EPAISSEURS)[ep]} · f<sub>u</sub> = {C.fr(a['fu'], 0)}{NBSP}MPa · β<sub>w</sub> = {C.fr(a['bw'], 2)}"
              + (f" · ε = {C.fr(NO.epsilon(fy), 2)}" if fy is not None else ""))
    val = C.fr(fy, 0) if fy is not None else "—"
    return {"titre": "Nuances d'acier", "barre": barre, "tableau": tableau,
            "bande": _bande(val, "MPa", "f<sub>y</sub>", "limite d'élasticité", phrase, avert),
            "pourquoi": BB._pourquoi(f"Acier {C.esc(nuance)}", "EN 1993-1-1 tableau 3.1 ; EN 1993-1-8 §4.5.3", termes, _symboles(sources), avert),
            "sources": BB._sources(*_symboles(sources)),
            "fiche": _fiche(f"Acier {nuance}", "f_y", fy, "MPa", p, termes, [BB._texte_brut(s) for s in sources], avert, texte=val, phrase=phrase),
            "params": p, "message": L.message()}


# ================================================================================================ 3 · boulons
VUES_BOULONS = (("boulons", "Boulons"), ("classes", "Classes"))
SOURCE_RS = ("EN 1993-1-8 tableau 3.4 — F_t,Rd = 0,9 · f_ub · A_s / γ_M2, F_v,Rd = α_v · f_ub · A_s / γ_M2 par plan dans le filetage : "
             "acier/resistances.py (les fonctions des moteurs d'assemblage) ; γ_M2 de acier/soudures/criteres.py")


def _r_boulons(params: Dict, hote: str) -> Dict:
    L = Lecture(params)
    bl, cl = BQ.DB["boulons"], BQ.DB["classes"]
    boulon = L.choix("boulon", [b["n"] for b in bl], "M16")
    classe = L.choix("classe", [c["n"] for c in cl], "8.8")
    vue = L.choix("vue", tuple(k for k, _ in VUES_BOULONS), "boulons")
    p = L.out
    b, c = BQ.find(bl, boulon), BQ.find(cl, classe)
    g = CR.GAMMA_M2

    def ft(x):
        return RS.Ft_Rd(float(c["fub"]), float(x["As"]), g)

    def fv(x):
        return RS.Fv_Rd(float(c["av"]), float(c["fub"]), float(x["As"]), g)
    barre = BB._barre(BB._champ("boulon", "Boulon", BB._select("boulon", boulon, [(x["n"], x["n"]) for x in bl]))
                      + _bloc("Classe", BB._segments("classe", classe, [(x["n"], x["n"]) for x in cl]))
                      + _bloc("Tableau", BB._segments("vue", vue, VUES_BOULONS)))
    if vue == "boulons":
        th = ('<tr><th class="nm">Boulon</th>' + BB._th("d", "mm", "diamètre nominal") + BB._th("d_0", "mm", "diamètre du trou")
              + BB._th("A", "mm²", "section de la tige lisse") + BB._th("A_s", "mm²", "section résistante dans le filetage")
              + BB._th("d_m", "mm", "diamètre moyen de la tête ou de l'écrou (poinçonnement)") + BB._th("d_w", "mm", "diamètre de la rondelle")
              + BB._th("F_t,Rd", f"kN · {classe}", f"traction, classe {classe}, γ_M2 = {C.nb(g, 2)}")
              + BB._th("F_v,Rd", f"kN · {classe}", f"cisaillement par plan dans le filetage, classe {classe}, γ_M2 = {C.nb(g, 2)}") + "</tr>")
        corps = ""
        for x in bl:
            n = x["n"]
            on = ' class="on"' if n == boulon else ""
            sel = {"boulon": n}
            corps += (f'<tr{on}><th class="nm">{C.esc(n)}</th>{BB._td(C.fr(x["d"], 0), sel)}{BB._td(C.fr(x["d0"], 0), sel)}'
                      f'{BB._td(C.fr(x["A"], 1), sel)}{BB._td(C.fr(x["As"], 1), sel)}{BB._td(C.nb(x["dm"], 1), sel)}{BB._td(C.fr(x["dw"], 0), sel)}'
                      f'{BB._td(C.fr(ft(x), 1), sel)}{BB._td(C.fr(fv(x), 1), sel, "cur" if n == boulon else "")}</tr>')
        tableau = BB._tableau(th, corps, "bib-boulons")
        valeur, unite, sym, nom = C.fr(fv(b), 1), "kN", "F<sub>v,Rd</sub>", "résistance au cisaillement par plan (filetage)"
    else:
        th = ('<tr><th class="nm">Classe</th>' + BB._th("f_yb", "MPa", "limite d'élasticité du boulon") + BB._th("f_ub", "MPa", "résistance ultime du boulon")
              + BB._th("α_v", "", "coefficient du cisaillement dans le filetage (tableau 3.4)") + "</tr>")
        corps = ""
        for x in cl:
            n = x["n"]
            on = ' class="on"' if n == classe else ""
            sel = {"classe": n}
            corps += (f'<tr{on}><th class="nm">{C.esc(n)}</th>{BB._td(C.fr(x["fyb"], 0), sel)}{BB._td(C.fr(x["fub"], 0), sel, "cur" if n == classe else "")}'
                      f'{BB._td(C.nb(x["av"], 2), sel)}</tr>')
        tableau = BB._tableau(th, corps, "bib-classes bib-petit")
        valeur, unite, sym, nom = C.fr(c["fub"], 0), "MPa", "f<sub>ub</sub>", f"résistance ultime, classe {C.esc(classe)}"
    termes = [
        {"symbole": "d", "nom": "diamètre nominal", "valeur": float(b["d"]), "unite": "mm", "regle": "bibliotheques.json"},
        {"symbole": "d_0", "nom": "diamètre du trou", "valeur": float(b["d0"]), "unite": "mm", "regle": "bibliotheques.json"},
        {"symbole": "A_s", "nom": "section résistante dans le filetage", "valeur": float(b["As"]), "unite": "mm²", "regle": "bibliotheques.json"},
        {"symbole": "f_ub", "nom": f"résistance ultime, classe {classe}", "valeur": float(c["fub"]), "unite": "MPa", "regle": "bibliotheques.json"},
        {"symbole": "α_v", "nom": "coefficient du cisaillement dans le filetage", "valeur": float(c["av"]), "unite": "", "regle": "bibliotheques.json (tableau 3.4)"},
        {"symbole": "γ_M2", "nom": "coefficient partiel des assemblages", "valeur": g, "unite": "", "regle": "ANB — acier/soudures/criteres.py"},
        {"symbole": "F_t,Rd", "nom": "RÉSISTANCE À LA TRACTION", "valeur": ft(b), "unite": "kN",
         "regle": f"0,9 · f_ub · A_s / γ_M2 = 0,9 · {C.fr(c['fub'], 0)} · {C.nb(b['As'], 1)} / {C.nb(g, 2)} — tableau 3.4, acier/resistances.Ft_Rd"},
        {"symbole": "F_v,Rd", "nom": "RÉSISTANCE AU CISAILLEMENT PAR PLAN", "valeur": fv(b), "unite": "kN",
         "regle": f"α_v · f_ub · A_s / γ_M2 = {C.nb(c['av'], 2)} · {C.fr(c['fub'], 0)} · {C.nb(b['As'], 1)} / {C.nb(g, 2)} — tableau 3.4, acier/resistances.Fv_Rd"},
    ]
    sources = [SOURCE_BQ, SOURCE_RS]
    phrase = (f"{C.esc(boulon)} classe {C.esc(classe)} · F<sub>t,Rd</sub> = {C.fr(ft(b), 1)}{NBSP}kN · F<sub>v,Rd</sub> = {C.fr(fv(b), 1)}{NBSP}kN "
              f"(plan dans le filetage) · A<sub>s</sub> = {C.fr(b['As'], 1)}{NBSP}mm² · d<sub>0</sub> = {C.fr(b['d0'], 0)}{NBSP}mm")
    return {"titre": "Boulons", "barre": barre, "tableau": tableau,
            "bande": _bande(valeur, unite, sym, nom, phrase),
            "pourquoi": BB._pourquoi(f"Boulon {C.esc(boulon)} {C.esc(classe)}", "EN 1993-1-8 tableau 3.4", termes, _symboles(sources), nd=2),
            "sources": BB._sources(*_symboles(sources)),
            "fiche": _fiche(f"Boulon {boulon} {classe}", "F_v,Rd", fv(b), "kN", p, termes, [BB._texte_brut(s) for s in sources], texte=C.fr(fv(b), 1), phrase=phrase),
            "params": p, "message": L.message()}


# ================================================================================================ la porte
RENDUS = {"profils": _r_profils, "nuances": _r_nuances, "boulons": _r_boulons}


def consulter(rubrique: Optional[str], params: Optional[Dict], hote: str = "page") -> Dict:
    """La rubrique rendue pour ces paramètres (les absents prennent leur défaut, les illisibles aussi — ``message`` le
    dit) ; une rubrique inconnue ouvre la première. ``hote`` : « page » ou « note » (une note ne reçoit rien de
    l'acier : la bande est la même)."""
    if hote not in HOTES:
        raise ValueError(f"hôte inconnu : {hote!r} (page ou note)")
    r = rubrique if rubrique in RENDUS else RUBRIQUE_DEFAUT
    out = RENDUS[r](params if isinstance(params, dict) else {}, hote)
    out["rubrique"] = r
    out["bibliotheque"] = BIBLIOTHEQUE
    if rubrique is not None and rubrique != r:
        out["message"] = " ; ".join(m for m in (f"rubrique « {str(rubrique)[:40]} » inconnue : « {NOMS[r]} » ouverte", out.get("message")) if m)
    return out


def menu(actif: str = RUBRIQUE_DEFAUT) -> str:
    """Le menu des rubriques (la page le porte en dur, js/biblio.js allume l'entrée ouverte)."""
    return "".join(f'<button type="button" class="bib-m{" on" if k == actif else ""}" data-rubrique="{k}" aria-pressed="{"true" if k == actif else "false"}">'
                   f'{C.icon(ic)}<span class="bib-mt">{t}</span><span class="bib-ms">{st}</span></button>' for k, t, st, ic in RUBRIQUES)

