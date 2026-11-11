# -*- coding: utf-8 -*-
"""Le module « Garde-corps » du site : l'état d'une note (celui du moteur, acier/garde_corps/etat), les champs de
saisie, le résultat, la grande fenêtre de calcul et le sélecteur de profilés — rendus en fragments HTML.

Le site reprend l'écran Streamlit (modules/garde_corps.py) : mêmes champs, mêmes effets de bord à la saisie (le type
pose le système et le remplissage, la catégorie pose q_h), même moteur. Rien n'est recalculé ici : les taux, les
formules et les dessins viennent du moteur ; ce fichier les MET EN FORME.

L'écran est l'établi COMPACT (api_web/etabli_compact.py, la mise en forme des poutres — 11/11/2026) :
- à gauche, la ligne du garde-corps ouvert (son nom, la liste, copier, supprimer) et six rubriques en accordéon
  (les aides au survol de ⓘ) ; la dernière, « Hypothèses », est secondaire ;
- au centre, les vues du moteur, chacune sous son en-tête technique ;
- à droite, la tête, le verdict, puis deux blocs — « Profilés » (les pièces qui se choisissent) et « Fixation » — ;
  chaque vérification tient en deux lignes : ce qu'elle compare, en clair, puis sa jauge ; elle ouvre son calcul.
"""
import copy
from typing import Dict, List, Optional, Sequence

from acier.garde_corps import bibliotheque as BI
from acier.garde_corps import charges as CH
from acier.garde_corps import etat as ET
from acier.garde_corps import fixation as FX
from acier.garde_corps import formules as FO
from acier.garde_corps import geometrie as GE
from acier.garde_corps import membres as ME
from acier.garde_corps import schemas as SC
from acier.garde_corps import types as TY

from . import commun as C
from . import etabli_compact as E
from . import note_cas as NC

MODULE = "garde_corps"
PAGE = "garde-corps"
MODES = {ME.ELS: "ELS", ME.EUROCODE: "Eurocode"}
FIXATIONS_COURTES = {"platine": "Platine", "laterale": "Latérale", "soudee": "Soudé", "scellee": "Scellé"}
JONCTIONS = {"angle": "Angle", "partielle": "Partielle", "pleine": "Pleine"}
VUES_COURTES = {"elevation": "Élévation", "coupe": "Coupe A-A", "plan": "Plan", "platine": "Platine", "pied": "Fixation", "section": "Section"}
# les familles proposées par défaut dans le sélecteur de chaque rôle (celles de l'écran Streamlit)
FAMILLES = {"montant": ("RHS", "SHS", "CHS"), "lisse": ("RHS", "SHS", "CHS"), "barreau": ("PLAT", "ROND"), "lisse_inter": ("RHS", "SHS", "CHS"),
            "console": ("PLAT",)}
FAMILLES_DEFAUT = {"montant": ("RHS",), "lisse": ("RHS",), "barreau": ("PLAT", "ROND"), "lisse_inter": ("RHS",), "console": ("PLAT",)}
LIB_FAMILLES = {"RHS": "RHS", "SHS": "SHS", "CHS": "CHS", "PLAT": "Plats", "ROND": "Ronds"}
GARDE_CORPS = E.Genre("Garde-corps", "le garde-corps", "ce garde-corps", "un garde-corps")
# les deux blocs du résultat : les pièces dont la section se choisit (ET.ROLES), puis la fixation
BLOCS = ("Profilés", "Fixation")
CLASSES_BLOC = {"p": "ok", "f": "warn", "r": "nok"}
CALCULS = {"montant": "montant", "lisse": "lisse", "barreau": "barreau", "lisse_inter": "barreau", "console": "console", "soudure": "soudure",
           "ancrage": "platine", "platine": "platine"}


# ------------------------------------------------------------------ l'état
NOM_DEFAUT = "G1"               # comme C1, F1, P1 : un nom court, qui tient entier dans la ligne de l'élément (tablettes)


def nouvel_etat() -> Dict:
    etat = ET.nouvel_etat()
    etat["cas"][0]["nom"] = NOM_DEFAUT
    return etat


def _nom_libre(cas: List[Dict]) -> str:
    """Le premier nom « G<n> » que la note ne porte pas encore."""
    pris = {c.get("nom") for c in cas}
    k = len(cas) + 1
    while f"G{k}" in pris:
        k += 1
    return f"G{k}"


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur : clés manquantes complétées, cas actif valide."""
    return ET.depuis_json(ET.vers_json(etat))


def _cas(etat: Dict) -> Dict:
    return ET.cas_actif(etat)


def _aller(d, chemin: Sequence):
    for c in chemin[:-1]:
        d = d[int(c)] if isinstance(d, list) else d[c]
    k = chemin[-1]
    return d, (int(k) if isinstance(d, list) else k)


# les champs ENTIERS ; les autres nombres sont des réels
ENTIERS = {("remplissage", "n_lisses"), ("fixation", "ancrage", "n_par_rangee"), ("fixation", "soudure", "troncons", "n")}


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans le cas actif, avec les mêmes effets de bord que l'écran Streamlit. Rend un message si la
    saisie est refusée (nombre illisible), None sinon. ``chemin`` : la liste des clés (« geometrie », « H »)."""
    cas = _cas(etat)
    chemin = [int(c) if isinstance(c, str) and c.isdigit() else c for c in chemin]
    if chemin == ["type"]:
        TY.appliquer_type(cas, valeur)
        return None
    if chemin == ["charges", "mode_P"]:
        cas["charges"]["mode"] = "P" if valeur else "q"
        return None
    if chemin[-1] == "entraxes":
        d, k = _aller(cas, chemin)
        vals = []
        for mot in str(valeur or "").replace(";", " ").replace(",", " ").split():
            x = C.lire_nombre(mot)
            if x is None or x <= 0:
                return f"Entraxes illisibles : « {valeur} »"
            vals.append(x)
        d[k] = vals
        return None
    d, k = _aller(cas, chemin)
    ancien = d.get(k) if isinstance(d, dict) else d[k]
    if isinstance(valeur, bool) or isinstance(ancien, bool):
        d[k] = bool(valeur)
    elif isinstance(ancien, (int, float)) or tuple(chemin) in ENTIERS:
        x = C.lire_nombre(valeur)
        if x is None:
            return f"Nombre illisible : « {valeur} »"
        d[k] = int(round(x)) if tuple(chemin) in ENTIERS else x
    else:
        d[k] = valeur
    if chemin == ["charges", "categorie"]:
        cas["charges"].update(CH.normaliser(cas["charges"]))
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Retient une section standard pour un rôle (le clic dans le sélecteur)."""
    ET.appliquer_section(_cas(etat), BI.section({"nom": nom}), role)


def _indice(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les garde-corps de la note : en ouvrir un, en ajouter, en copier un (``index`` ; sans : l'ouvert — la copie
    s'insère juste après), en supprimer un (jamais le dernier)."""
    cas = etat["cas"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(cas) - 1)
    elif action == "ajouter":
        cas.append(ET.nouveau_cas(_nom_libre(cas)))
        etat["actif"] = len(cas) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(cas) - 1)
        c = copy.deepcopy(cas[i])
        c["nom"] = f"{c.get('nom') or 'Garde-corps'} (copie)"
        cas.insert(i + 1, c)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(cas) > 1:
        i = min(max(0, _indice(index)), len(cas) - 1)
        del cas[i]
        if etat["actif"] > i or etat["actif"] >= len(cas):
            etat["actif"] = max(0, etat["actif"] - 1)


def _taux(r: Dict) -> Optional[float]:
    return r["taux_max"] if r.get("statut") != ET.REFUS and r.get("gouvernant") else None


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : son élément le plus chargé, tous garde-corps confondus."""
    vals = [(t, i) for i, r in enumerate(resultats) if (t := _taux(r)) is not None]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    niv = C.niveau(t)
    nom = etat["cas"][i].get("nom") or f"G{i + 1}"
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1], "gouvernant": f"{nom} · {resultats[i]['gouvernant']['libelle']}"}


def elements(etat: Dict, resultats: Optional[List[Dict]] = None) -> str:
    """Rien : les garde-corps de la note sont la LIGNE de l'élément, en tête de la colonne de gauche (``arbre``)."""
    return ""


# ------------------------------------------------------------------ la saisie
def _noms_sections(groupes: Sequence[str]) -> List[str]:
    return [s["nom"] for s in BI.standards().values() if BI.groupe_de(s) in groupes]


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Les six rubriques du garde-corps ouvert : 1 Garde-corps, 2 Tronçons, 3 Remplissage, 4 Charges, 5 Fixation,
    6 Hypothèses (secondaire). Chaque champ est lié à un chemin de l'état ; les aides sont au survol de ⓘ. Deux champs
    par ligne seulement là où ils tiennent (libellés très courts, petites valeurs) : la colonne fait 340 px, et une
    valeur doit s'y lire en entier (« 3 600 mm »)."""
    cas = _cas(etat)
    res = res if res is not None else ET.verifier(cas)
    g, ch, rp, fx, m = cas["geometrie"], cas["charges"], cas["remplissage"], cas["fixation"], cas["methode"]
    L, L2 = C.ligne, C.ligne2
    geo = res.get("geometrie") or {}
    # 1 · le garde-corps (son nom se modifie sur sa ligne, en tête de la colonne)
    s1 = (L("Type", C.choix(["type"], cas.get("type"), [(k, v["libelle"]) for k, v in TY.TYPES.items()]), aide=TY.TYPES[TY.type_de(cas)]["description"],
            full=True)
          + L("Forme", C.segments(["geometrie", "forme"], g.get("forme"), [(k, v[:1].upper() + v[1:]) for k, v in GE.FORMES.items()]), full=True)
          + L("hauteur", C.num(["geometrie", "H"], g.get("H"), "mm"), "H",
              "Hauteur de l'axe de la main courante au-dessus de la platine ; la charge s'applique au plus à 1,20 m (EN 1991-1-1 §6.4).")
          + L("pente", C.num(["geometrie", "pente"], g.get("pente"), "°", mini=0, maxi=60), "α", "Garde-corps d'escalier ; 0 : horizontal."))
    # 2 · les tronçons
    n = GE.N_TRONCONS.get(g.get("forme", "droit"), 1)
    s2 = ""
    for i in range(n):
        tr = g["troncons"][i]
        s2 += ((f'<div class="grp">Tronçon {i + 1}</div>' if n > 1 else "")
               + L("longueur", C.num(["geometrie", "troncons", i, "L"], tr.get("L"), "mm"), "L")
               + L("entraxe maxi", C.num(["geometrie", "troncons", i, "s_max"], tr.get("s_max"), "mm"), "s",
                   "La longueur est répartie en travées égales ne dépassant pas cet entraxe, sauf si des entraxes sont saisis.")
               + L("Entraxes imposés", C.texte(["geometrie", "troncons", i, "entraxes"], " ".join(f"{e:g}" for e in tr.get("entraxes") or []),
                                                "réguliers"), full=True))
    s2 += (L("Début", C.choix(["geometrie", "debut"], g.get("debut"), list(GE.EXTREMITES.items())), full=True)
           + L("Fin", C.choix(["geometrie", "fin"], g.get("fin"), list(GE.EXTREMITES.items())), full=True))
    if geo.get("sans_montant"):
        s2 += C.aide("Sans montant : la main courante porte de mur à mur (les deux bouts ancrés, une seule travée).")
    # 3 · le remplissage
    rpt = rp.get("type")
    if rpt == "barreaux":
        basses = _noms_sections(["creux", "froid", "plats", "ronds"])
        cur = (rp.get("section_basse") or {}).get("nom")
        s3 = (L("entraxe des barreaux", C.num(["remplissage", "entraxe"], rp.get("entraxe"), "mm"), "e<sub>b</sub>")
              + L("lisse basse", C.num(["remplissage", "h_bas"], rp.get("h_bas"), "mm"), "h<sub>bas</sub>", "Axe de la lisse basse au-dessus de la platine.")
              + L("Lisse basse", C.choix(["remplissage", "section_basse", "nom"], cur, [(x, x) for x in ([cur] if cur not in basses else []) + basses]),
                  full=True, aide="Elle fixe la portée des barreaux ; elle n'est pas vérifiée. La section des barreaux se choisit dans le résultat."))
        r = res.get("remplissage") or {}
        resume3 = f"barreaux · e {C.nb(rp.get('entraxe'))} mm" + (f" · portée {C.fr(r['L'], 0)} mm" if r.get("L") else "")
    elif rpt == "lisses":
        s3 = (L("nombre de lisses", C.num(["remplissage", "n_lisses"], rp.get("n_lisses"), "", entier=True, mini=1), "n")
              + L("entraxe des lisses", C.num(["remplissage", "entraxe_lisses"], rp.get("entraxe_lisses"), "mm"), "e<sub>l</sub>"))
        resume3 = f"{rp.get('n_lisses')} lisses · e {C.nb(rp.get('entraxe_lisses'))} mm"
    elif rpt == "panneau":
        s3 = (L("lisse basse", C.num(["remplissage", "h_bas"], rp.get("h_bas"), "mm"), "h<sub>bas</sub>")
              + C.aide("La résistance du panneau sous q<sub>r</sub> et Q<sub>r</sub> est à justifier par le fournisseur ; la note le dit."))
        resume3 = "panneau"
    else:
        s3 = C.aide("Main courante seule, sans remplissage (rampe, lisse de protection).")
        resume3 = "aucun"
    # 4 · les charges
    cats = [(c[0], c[1] + (f" — {c[2]:g} kN/m".replace(".", ",") if c[2] is not None else "")) for c in CH.CATEGORIES]
    s4 = (L("Catégorie", C.choix(["charges", "categorie"], ch.get("categorie"), cats), full=True)
          + '<div class="grp">Sur la main courante</div>'
          + L("horizontale", C.num(["charges", "qh"], ch.get("qh"), "kN/m", desactive=ch.get("categorie") != "perso"), "q<sub>h</sub>", CH.SOURCE_QH)
          + L("verticale", C.num(["charges", "qv"], ch.get("qv"), "kN/m"), "q<sub>v</sub>", CH.SOURCE_QV)
          + '<div class="grp">Sur le remplissage</div>'
          + L("ponctuelle", C.num(["charges", "Qr"], ch.get("Qr"), "kN"), "Q<sub>r</sub>", CH.SOURCE_QR_PONCT)
          + L("pression", C.num(["charges", "qr"], ch.get("qr"), "kN/m²"), "q<sub>r</sub>", CH.SOURCE_QR)
          + C.case(["charges", "cumul"], bool(ch.get("cumul")), "Cumuler q<sub>h</sub> et q<sub>v</sub> (cas 1 + 2, côté sûr)")
          + C.case(["charges", "mode_P"], ch.get("mode") == "P", "Saisir directement la charge P par montant"))
    if ch.get("mode") == "P":
        s4 += L("par montant", C.num(["charges", "P"], ch.get("P"), "kN"), "P")
    s4 += (L("sur la main courante", C.num(["charges", "Q"], ch.get("Q"), "kN"), "Q", "Charge ponctuelle horizontale ; 0 : sans.")
           + C.aide("Quatre cas distincts : <b>chaque charge agit seule</b>, sauf cumul demandé."))
    # 5 · la fixation
    typ = fx.get("type")
    s5 = L("Fixation", C.segments(["fixation", "type"], typ, list(FIXATIONS_COURTES.items())), full=True)
    if g.get("systeme") == "mur":
        s5 += L("console : mur → axe", C.num(["fixation", "console", "e"], fx["console"].get("e"), "mm"), "e")
    if geo.get("sans_montant"):
        s5 += C.aide("Sans montant : rien à fixer, la main courante est ancrée dans les murs.")
    elif typ == "scellee":
        s5 += C.aide("Montant scellé : la fixation n'est pas vérifiée ici.")
    else:
        lat = typ == FX.LATERALE and g.get("systeme") == "montants"
        sd = fx["soudure"]
        s5 += '<div class="grp">Soudure</div>' + L("Jonction", C.segments(["fixation", "soudure", "jonction"], sd.get("jonction"), list(JONCTIONS.items())), full=True)
        tp = ("platine t<sub>p</sub>", C.num(["fixation", "platine", "tp"], fx["platine"].get("tp"), "mm"), "Épaisseur de la platine.")
        if sd.get("jonction") == "angle":
            s5 += L2(("gorge a", C.num(["fixation", "soudure", "a"], sd.get("a"), "mm"), ""), tp)
        elif sd.get("jonction") == "partielle":
            s5 += (L("pénétration", C.num(["fixation", "soudure", "p"], sd.get("p"), "mm"), "p")
                   + L("renfort", C.num(["fixation", "soudure", "renfort"], sd.get("renfort"), "mm"), "a<sub>r</sub>", "0 : sans renfort.")
                   + L("épaisseur de platine", C.num(["fixation", "platine", "tp"], fx["platine"].get("tp"), "mm"), "t<sub>p</sub>"))
        else:
            s5 += L("épaisseur de platine", C.num(["fixation", "platine", "tp"], fx["platine"].get("tp"), "mm"), "t<sub>p</sub>")
        if typ in ("platine", FX.LATERALE):
            pl, an = fx["platine"], fx["ancrage"]
            s5 += '<div class="grp">Platine et ancrage</div>'
            if lat:
                s5 += L("descente", C.num(["fixation", "platine", "d_h"], pl.get("d_h"), "mm"), "d<sub>h</sub>", "Du dessus de dalle au haut de la platine.")
            s5 += (L2((("hauteur" if lat else "côté") + " b<sub>p</sub>", C.num(["fixation", "platine", "bp"], pl.get("bp"), "mm"),
                       "Hauteur de la platine." if lat else "Dimension de la platine dans le sens de l'effort."),
                      ("largeur l<sub>p</sub>", C.num(["fixation", "platine", "lp"], pl.get("lp"), "mm"), ""))
                   + L("entraxe des rangées", C.num(["fixation", "ancrage", "ex"], an.get("ex"), "mm"), "e<sub>x</sub>")
                   + L("par rangée", C.num(["fixation", "ancrage", "n_par_rangee"], an.get("n_par_rangee"), "", entier=True, mini=1, maxi=4), "n")
                   + L("entraxe dans la rangée", C.num(["fixation", "ancrage", "ey"], an.get("ey"), "mm"), "e<sub>y</sub>")
                   + L("Ancrage", C.segments(["fixation", "ancrage", "type"], an.get("type"), [("cheville", "Chevilles"), ("boulon", "Boulons")]), full=True))
            if an.get("type") == "boulon":
                s5 += L2(("boulon", C.choix(["fixation", "ancrage", "boulon"], an.get("boulon"), [(b, b) for b in FX.boulons()]), ""),
                         ("classe", C.choix(["fixation", "ancrage", "classe"], an.get("classe"), [(c, c) for c in FX.classes()]), ""))
            else:
                s5 += (L("Cheville", C.texte(["fixation", "ancrage", "nom"], an.get("nom"), "désignation"), full=True)
                       + L("traction, par cheville", C.num(["fixation", "ancrage", "NRd"], an.get("NRd"), "kN"), "N<sub>Rd</sub>",
                           "Valeur de calcul en traction de l'ETA, pour le béton et la profondeur du projet.")
                       + L("cisaillement, par cheville", C.num(["fixation", "ancrage", "VRd"], an.get("VRd"), "kN"), "V<sub>Rd</sub>"))
    # 6 · les hypothèses (secondaire : la méthode et ses coefficients)
    s6 = (L("Méthode", C.segments(["methode", "mode"], m.get("mode"), list(MODES.items())), full=True)
          + L("Nuance", C.choix(["methode", "nuance"], m.get("nuance"), [(k, k) for k in ET.nuances()]), full=True)
          + L("module d'élasticité", C.num(["methode", "E"], m.get("E"), "N/mm²"), "E")
          + L("Flèche", C.segments(["methode", "fleche_mode"], m.get("fleche_mode"), [("L/x", "L / x"), ("mm", "en mm")]), full=True)
          + L("x, ou la valeur en mm", C.num(["methode", "fleche_val"], m.get("fleche_val"), ""), "", "L/100 par défaut."))
    if m.get("mode") == ME.ELS:
        s6 += (L("σ<sub>adm</sub> = f<sub>y</sub> /", C.num(["methode", "gamma_adm"], m.get("gamma_adm"), ""), "γ<sub>adm</sub>",
                 "σ_adm = f_y/1,5 (156,7 N/mm² en S235), τ_adm = σ_adm/√3, σ_eq = √(σ² + 3τ²) ≤ σ_adm.")
               + L("cordons", C.num(["methode", "k"], m.get("k"), "kg/(mm·cm)"), "k")
               + L("résistances de calcul → service", C.num(["methode", "gamma_F"], m.get("gamma_F"), ""), "γ<sub>F</sub>"))
    else:
        s6 += (L2(("γ<sub>Q</sub>", C.num(["methode", "gamma_Q"], m.get("gamma_Q"), ""), "Coefficient des charges."),
                  ("γ<sub>M0</sub>", C.num(["methode", "gamma_M0"], m.get("gamma_M0"), ""), "Coefficient des sections."))
               + L("boulons, cordons", C.num(["methode", "gamma_M2"], m.get("gamma_M2"), ""), "γ<sub>M2</sub>")
               + L("Critère des cordons", C.choix(["methode", "ec"], m.get("ec"), [("directionnelle", "directionnelle"), ("simplifiee", "simplifiée")]),
                   full=True))
    trs = GE.longueurs_troncons(g)
    resume2 = f"{C.fr(sum(trs) / 1000, 2)} m · {geo.get('phrase', '')}" if trs else ""
    return (C.section(1, "Garde-corps", f"{TY.libelle(cas)} · H {C.nb(g.get('H'))}", s1, True, "gc")
            + C.section(2, "Tronçons", resume2, s2, False, "tr")
            + C.section(3, "Remplissage", resume3, s3, False, "rp")
            + C.section(4, "Charges", f"{CH.libelle_categorie(ch.get('categorie'))} · q<sub>h</sub> {C.nb(ch.get('qh'))} kN/m", s4, False, "ch")
            + C.section(5, "Fixation", FIXATIONS_COURTES.get(typ, ""), s5, False, "fx")
            + C.section(6, "Hypothèses", f"{MODES.get(m.get('mode'))} · {m.get('nuance')} · flèche L/{C.nb(m.get('fleche_val'))}", s6, False, "me",
                        secondaire=True))


def _dims(cas: Dict, res: Dict) -> str:
    """« · Barreaudage vertical · 3,60 m » : ce qu'on lit d'un garde-corps dans la liste de la note (HTML)."""
    lg = (res.get("geometrie") or {}).get("L_total")
    return f"· {C.esc(TY.libelle(cas))}" + (f" · {C.fr(lg / 1000, 2)}{C.NBSP}m" if lg else "")


def arbre(etat: Dict, res: Optional[Dict] = None, resultats: Optional[List[Dict]] = None) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact, la même que la poutre) : la ligne du
    garde-corps ouvert (son nom, qu'un clic modifie ; la liste des garde-corps de la note ; copier, supprimer), que
    l'écran pose dans l'en-tête du panneau, puis ses six rubriques."""
    resultats = resultats or [ET.verifier(c) for c in etat["cas"]]
    noms = [(c.get("nom") or f"G{i + 1}", _dims(c, r), False) for i, (c, r) in enumerate(zip(etat["cas"], resultats))]
    return E.arbre(GARDE_CORPS, noms, etat["actif"], formulaire(etat, res), "gardecorps", compact=True)


# ------------------------------------------------------------------ le résultat
def _blocs(res: Dict) -> Dict:
    """Les blocs de formules du moteur, par élément ; la soudure (groupe, point, critère) au même format."""
    sd = FO.soudure(res)
    soud = {"cas": "", "autres": "", "groupes": [(t, sd.get(k, [])) for k, t in (("groupe", "Groupe de cordons"), ("point", "Point le plus sollicité"),
                                                                                    ("critere", "Critère"))]}
    soud["subs"] = [x for _, xs in soud["groupes"] for x in xs]
    return {"montant": FO.montant(res), "console": FO.console(res), "lisse": FO.lisse(res), "barreau": FO.barre(res), "platine": FO.platine(res),
            "soudure": soud}


NOMS_TAUX = {"η_eq": "Contrainte équivalente", "η_τ": "Cisaillement", "η_f": "Flèche", "η_σ": "Contrainte", "η_t": "Ancrages · traction",
             "η_v": "Ancrages · cisaillement", "η_inter": "Ancrages · traction et cisaillement", "η_p": "Platine · flexion"}
UNITES = {"σ": "N/mm²", "τ": "N/mm²", "f": "mm", "F": "kN", "N": "kN", "V": "kN", "|": "N/mm"}


def _tableau(cle: str, bloc: Dict) -> List[Dict]:
    """Les lignes du tableau de la fenêtre de calcul : chaque taux η = valeur / limite du moteur, avec ses deux termes."""
    import re
    subs = bloc.get("subs") or [x for _, xs in bloc.get("groupes", []) for x in xs]
    defs = {s.nom: s for s in subs}
    lignes = []
    for s in subs:
        nom = NOMS_TAUX.get(s.nom) or ("Cordons de soudure" if cle == "soudure" and s.nom == "η" else None)
        if not nom or not s.vals:
            continue
        vals = list(s.vals.values())
        if len(vals) < 2:
            continue
        (sv, vv), (sl, vl) = vals[:2]
        d = defs.get(sv)
        u = (d.unit if d is not None and d.unit else UNITES.get(sv[0], ""))
        simple = re.fullmatch(r"\w+/\w+", s.template or "") is not None
        if simple:
            formule = FO.morceaux(d)[0][1] if d is not None and len(FO.morceaux(d)[0]) > 1 else ""
            f_html = f"{C.symb_math(sv)} = {C.symb_math(formule)}" if formule else C.symb_math(sv)
            valeur = f"{C.fr(vv, 2 if abs(vv) < 10 else 1)} {u}".strip()
            limite = f"{C.symb_math(sl)} = {C.fr(vl, 1)} {u}".strip()
        else:
            f_html = f"{C.symb_math(s.nom)} = {C.symb_math(FO.morceaux(s)[0][1])}"
            valeur, limite = C.fr(s.res, 2), "1,00"
        lignes.append({"nom": nom, "formule": f_html, "valeur": valeur, "limite": limite, "taux": C.pct(s.res, 1), "niveau": C.niveau(s.res), "t": s.res})
    lignes.sort(key=lambda li: -li["t"])
    if lignes:
        lignes[0]["gov"] = True
    return lignes


def _detail(bloc: Dict) -> str:
    out = ""
    for titre, subs in bloc.get("groupes", []):
        lignes = []
        for s in subs:
            parts, r = FO.morceaux(s)
            lignes.append({"parts": parts, "res": r, "niveau": C.niveau(s.res) if s.nom.startswith("η") else ""})
        if lignes:
            out += f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>{C.lignes_fx(lignes)}</div>'
    return out


def _hypotheses(res: Dict) -> List[str]:
    m = res["methode"]
    fy = res["acier"]["fy"]
    fl = f"L/{C.nb(m.get('fleche_val'))}" if m.get("fleche_mode", "L/x") == "L/x" else f"{C.nb(m.get('fleche_val'))} mm"
    if m.get("mode") == ME.ELS:
        lim = ME.limites(m, fy)
        meth = f"Méthode <b>ELS</b> · σ<sub>adm</sub> = f<sub>y</sub> / {C.nb(lim['coef'])} = {C.fr(lim['sigma'])} N/mm²"
    else:
        meth = f"Méthode <b>Eurocode</b> · γ<sub>Q</sub> = {C.nb(m.get('gamma_Q'))} · γ<sub>M0</sub> = {C.nb(m.get('gamma_M0'))}"
    return [f"Acier <b>{C.esc(res['acier']['nuance'])}</b> · f<sub>y</sub> = {C.nb(fy)} N/mm²", meth, f"Flèche ≤ <b>{fl}</b>"]


def _optimaux(cas: Dict) -> Dict[str, Dict]:
    """Le plus léger qui tient, pour chaque rôle, parmi les familles par défaut (l'étoile des lignes du résultat)."""
    out = {}
    for role in ET.roles_du_cas(cas):
        cands = [c for c in ET.candidats(cas, role, None, 15.0) if c["section"]["famille"].replace("-F", "") in FAMILLES_DEFAUT.get(role, ())]
        cands.sort(key=lambda c: c["masse"])
        b = ET.plus_leger(cands)
        if b is not None:
            out[role] = {"nom": b["section"]["nom"], "taux": C.pct(b["taux"]), "retenu": b["section"]["nom"] == (ET.spec_du_role(cas, role) or {}).get("nom")}
    return out


# le critère d'une ligne du résultat, en court (le tableau de la fenêtre de calcul garde le nom entier)
CRITERES_COURTS = {"Contrainte équivalente": "σ<sub>éq</sub>", "Cisaillement": "τ", "Flèche": "Flèche", "Contrainte": "σ",
                   "Ancrages · traction": "Traction", "Ancrages · cisaillement": "Cisaillement",
                   "Ancrages · traction et cisaillement": "Traction et cisaillement", "Platine · flexion": "Flexion", "Cordons de soudure": "Cordons"}
TITRES_CALCUL = {"montant": "Montant principal", "lisse": "Main courante", "barreau": "Remplissage", "console": "Console",
                 "platine": "Fixation · platine et ancrages", "soudure": "Soudure du montant sur la platine"}


def _lignes_calcul(e: Dict, blocs: Dict) -> List[Dict]:
    """Les lignes du tableau de calcul qui concernent un élément du résultat, la plus chargée d'abord : celles de sa
    fenêtre — l'ancrage et la platine partagent la même, chacun y a les siennes."""
    cal = CALCULS.get(e["cle"], "")
    lignes = _tableau(cal, blocs[cal]) if cal and (blocs.get(cal) or {}).get("groupes") else []
    if e["cle"] == "ancrage":
        lignes = [li for li in lignes if li["nom"].startswith("Ancrages")]
    elif e["cle"] == "platine":
        lignes = [li for li in lignes if li["nom"].startswith("Platine")]
    return lignes


def _comparaison(e: Dict, blocs: Dict) -> str:
    """Ce qu'une vérification compare, en clair (HTML) : le critère qui gouverne l'élément, sa valeur face à sa limite
    — la première ligne du tableau de sa fenêtre de calcul —, puis le cas de charge."""
    lignes = _lignes_calcul(e, blocs)
    d = e.get("detail") if isinstance(e.get("detail"), dict) else {}
    cas_ch = d.get("cas_charge")
    suite = f" · {C.esc(CH.CAS_LIBELLES.get(cas_ch, f'cas {cas_ch}').split(' · ')[0])}" if cas_ch else ""
    if not lignes:
        crit = ME.critere_texte(e["critere"]) if e["cle"] in ET.ROLES else e["critere"]
        return f"{C.esc(crit)}{suite}"
    li = lignes[0]
    # « Flèche 7,01 ≤ 11,0 mm · cas 1 » : le critère en court, la valeur sans son unité (la limite la porte), puis la limite
    valeur = li["valeur"].partition(" ")[0]
    return f'{CRITERES_COURTS.get(li["nom"], C.esc(li["nom"]))} {valeur} {"≤" if li["t"] <= 1 else "&gt;"} {li["limite"].split(" = ")[-1]}{suite}'


def _nom_et_piece(e: Dict) -> tuple:
    """Le nom d'un élément du résultat et sa pièce (la section choisie, la gorge, les ancrages…), tirés de son libellé.
    Le nom est dit COURT (« Montant M2 », « Lisses (2) ») : dans la colonne des résultats, le nom, la section et le taux
    tiennent sur une ligne sans être coupés."""
    parts = e["libelle"].split(" · ")
    nom = parts[0].replace(" principal", "").replace(" intermédiaires", "")
    if e["cle"] in ET.ROLES:
        return nom, (parts[1] if len(parts) > 1 else "")
    if nom.startswith("Platine "):
        return "Platine", nom[len("Platine "):]
    return nom, " · ".join(parts[1:]).replace("cordon d'angle ", "gorge ").replace(" chevilles", "")


def _charges_txt(res: Dict) -> str:
    """Les quatre charges du calcul, sur une ligne (HTML) : « q_h 0,5 kN/m · q_v 0,5 kN/m · Q_r 0,5 kN · q_r 0,5 kN/m² »."""
    return " · ".join(f"{C.symb_math(c['symbole'])} {C.nb(c['valeur'])}{C.NBSP}{c['unite']}" for c in res.get("charges_tableau") or [])


def _entete(res: Dict, titre: str) -> str:
    """L'en-tête technique d'une vue : le type, l'acier et la méthode ; la géométrie ; les charges (en texte)."""
    m = res["methode"]
    sous = f"{C.esc(res.get('type_libelle', ''))} · {C.esc(res['acier']['nuance'])} · {MODES.get(m.get('mode'), '')}"
    charges = " · ".join(f"{c['symbole'].replace('_', '')} = {C.nb(c['valeur'])} {c['unite']}" for c in res.get("charges_tableau") or [])
    return E.entete(titre.upper(), sous, [(res.get("geometrie") or {}).get("phrase", ""), charges], None)


def _fenetre_bloc(i: int, cles: Sequence[str], blocs: Dict, hyps: List[str]) -> str:
    """« Tout le calcul » d'un bloc (fenêtre ``s<i>``) : le tableau de toutes ses vérifications, chacune nommée par sa
    pièce, la plus chargée en tête ; puis le détail des formules, pièce par pièce."""
    lignes, detail = [], ""
    for k in dict.fromkeys(cles):
        b = blocs.get(k) or {}
        if not b.get("groupes"):
            continue
        titre = TITRES_CALCUL.get(k, k)
        for li in _tableau(k, b):
            li = dict(li, nom=f"{C.esc(titre)}<span>{li['nom']}</span>")
            li.pop("gov", None)
            lignes.append(li)
        detail += f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div></div>{_detail(b)}'
    if not lignes:
        return ""
    lignes.sort(key=lambda li: -li["t"])
    lignes[0]["gov"] = True
    return C.fenetre_calcul(f"s{i}", BLOCS[i], "toutes les vérifications du bloc · les formules de la note", hyps, lignes, detail)


def resultat(etat: Dict, avec_optimaux: bool = True) -> Dict:
    """Le calcul et sa mise en forme : la tête (le nom, le type et la géométrie), le verdict, les deux blocs (« Profilés »,
    « Fixation ») de vérifications sur deux lignes, les messages, la masse ; les fenêtres de calcul (une par pièce, une
    par bloc), les vues sous leur en-tête technique, la colonne de gauche, le résumé de la note."""
    cas = _cas(etat)
    res = ET.verifier(cas)
    stat = res["statut"]
    g = res.get("gouvernant")
    blocs = _blocs(res)
    refus = stat == ET.REFUS or not g
    if refus:
        niv = "x"
        verdict = f'<div class="verdict"><div class="gov"><b>Calcul impossible</b>{C.esc((res.get("messages") or ["données manquantes"])[0])}</div></div>'
    else:
        niv = C.niveau(res["taux_max"])
        el_g = next((e for e in res["elements"] if e["cle"] == g["cle"]), {})
        cal_g = CALCULS.get(g["cle"], "")
        ouvre = f' data-calcul="{cal_g}" title="Voir le calcul"' if cal_g and (blocs.get(cal_g) or {}).get("groupes") else ""
        verdict = (f'<div class="verdict"{ouvre}>{C.chip(C.pct(res["taux_max"]), niv, xl=True)}<div class="gov"><b>{C.esc(g["libelle"].split(" · ")[0])}</b>'
                   f'{_comparaison(el_g, blocs) if el_g else ""}</div></div>')
    resultats = [res if i == etat["actif"] else ET.verifier(c) for i, c in enumerate(etat["cas"])]
    opt = _optimaux(cas) if avec_optimaux and stat != ET.REFUS else {}
    roles = ET.roles_du_cas(cas)
    groupes: Dict[int, List] = {0: [], 1: []}
    for e in res.get("elements", []):
        nom, piece = _nom_et_piece(e)
        o = opt.get(e["cle"])
        leger = f"Plus léger qui tient : {o['nom']} · {o['taux']} %" if o and not o["retenu"] else ""
        cal = CALCULS.get(e["cle"], "")
        if cal and not (blocs.get(cal) or {}).get("groupes"):
            cal = ""
        row = C.ligne_verif(nom, _comparaison(e, blocs), C.pct(e["taux"]), C.niveau(e["taux"]), calcul=cal, pk=e["cle"] if e["cle"] in roles else "",
                            piece=C.esc(piece), gov=bool(g and e["cle"] == g["cle"]), leger=leger)
        groupes[0 if e["cle"] in ET.ROLES else 1].append((e, cal, row))
    hyps = _hypotheses(res) if stat != ET.REFUS else []
    html_blocs, volets = "", ""
    for i, titre in enumerate(BLOCS):
        if not groupes[i]:
            continue
        t = max(e["taux"] for e, _, _ in groupes[i])
        nb_ = C.niveau(t)
        efforts = _charges_txt(res) if i == 0 else C.esc((res.get("fixation") or {}).get("libelle", "").capitalize())
        html_blocs += E.bloc(i, titre, CLASSES_BLOC.get(nb_, "inc"), f"{C.STATUTS[nb_][1]} · {C.pc(C.pct(t))}{C.NBSP}%", efforts,
                             "".join(r for _, _, r in groupes[i]), theme=True)
        volets += _fenetre_bloc(i, [c for _, c, _ in groupes[i] if c], blocs, hyps)
    messages = "".join(C.message(C.esc(msg), msg.startswith("Refus") or "DÉPASSÉ" in msg) for msg in res.get("messages", []))
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse indicative de l\'acier</span><b>{C.fr(res.get("masse") or 0, 0)}{C.NBSP}kg</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    nom = C.esc(cas.get("nom") or f"G{etat['actif'] + 1}")
    dims = f'{C.esc(res.get("type_libelle", ""))} · {C.esc((res.get("geometrie") or {}).get("phrase", ""))}'
    html_res = (f'<div class="pbody">{E.tete_resultats(nom, dims)}{verdict}<div class="et-blocs">{html_blocs}</div>{messages}</div>{pied}')
    # les fenêtres de calcul des pièces (celles des blocs, « tout le calcul », sont déjà faites)
    for k, b in blocs.items():
        if not b.get("groupes"):
            continue
        lignes = _tableau(k, b)
        sous = f"{C.symb_math(b['cas'])} · les formules de la note" if b.get("cas") else "les formules de la note"
        pied_ck = f'<div class="ck-autres">Autres cas : {C.symb_math(b["autres"])}</div>' if b.get("autres") else ""
        volets += C.fenetre_calcul(k, TITRES_CALCUL.get(k, k), sous, hyps, lignes, _detail(b), pied_ck)
    # la toile : une feuille par vue du moteur, sous son en-tête technique
    vues = [] if stat == ET.REFUS else [{"cle": k, "titre": t, "court": VUES_COURTES.get(k, t), "svg": SC.svg(d, f"gc_{k}"), "entete": _entete(res, t)}
                                        for k, t, d in SC.vues(res)]
    # le statut de la NOTE (dès deux garde-corps) : le moins bon de ses éléments ; seul, le verdict le dit déjà
    nivs = [C.niveau(t) if (t := _taux(r)) is not None else "x" for r in resultats]
    statut = C.statut(next((n_ for n_ in ("x", "r", "f") if n_ in nivs), "p")) if len(resultats) > 1 else ""
    return {"resultat": html_res, "volets": volets, "vues": vues, "phrase": "", "statut_html": statut, "resume": resume_note(etat, resultats),
            "donnees": arbre(etat, res, resultats), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ la note PDF
# le générateur VALIDÉ de l'écran Streamlit (acier/garde_corps/note.py), deux pages par garde-corps de la note — trois
# quand une colonne ne tient pas (sa page de suite) — qu'il pose lui-même dans la note (``poser``, api_web/note_cas.py) ;
# l'identification imprimée est celle de la note du site
NOTE = NC.Generateur("acier.garde_corps.note", "Garde-corps", None,
                     ("acier/garde_corps/note.py", "acier/garde_corps/methode.py",
                      "acier/assemblages/poutre_poutre/doubles_cornieres/rapport.py"))


def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note (chaque garde-corps, deux pages, trois avec une page de
    suite), faite par le générateur validé de l'écran Streamlit ; ``infos`` : le nom, le chantier, l'indice et la date
    de la note du site."""
    if quoi != "note_pdf":
        return None
    return NC.fichier(NOTE, etat, infos)


# ------------------------------------------------------------------ le sélecteur
def selecteur(etat: Dict, role: str, familles: Optional[Sequence[str]] = None, froid: bool = True) -> str:
    """Le sélecteur de profilé d'un rôle : les sections du moteur TRIÉES PAR MASSE, le retenu et l'optimal marqués, une
    fenêtre autour d'eux ; les familles cochées filtrent (les puces du haut)."""
    cas = _cas(etat)
    fams = tuple(familles) if familles else FAMILLES_DEFAUT.get(role, ())
    groupes = sorted({"creux" for f in fams if f in ("RHS", "SHS", "CHS")} | ({"froid"} if froid and any(f in ("RHS", "SHS", "CHS") for f in fams) else set())
                     | ({"plats"} if "PLAT" in fams else set()) | ({"ronds"} if "ROND" in fams else set()))
    cands = sorted((c for c in ET.candidats(cas, role, groupes, 25.0) if c["section"]["famille"].replace("-F", "") in fams), key=lambda c: c["masse"])
    best = ET.plus_leger(cands)
    cur = (ET.spec_du_role(cas, role) or {}).get("nom")
    noms = [c["section"]["nom"] for c in cands]
    i_o = noms.index(best["section"]["nom"]) if best else 0
    i_c = noms.index(cur) if cur in noms else i_o

    def prop(sec):
        fam = sec["famille"]
        norme = ("EN 10219 · formé à froid" if fam.endswith("-F") else "EN 10210 · fini à chaud") if fam.replace("-F", "") in ("RHS", "SHS", "CHS") else "barre pleine"
        vals = [f"A = {C.fr(sec['A'], 2)} cm²"] + ([f"I<sub>y</sub> = {C.fr(sec['Iy'], 2)} cm⁴"] if sec.get("Iy") else []) \
            + ([f"W<sub>el,y</sub> = {C.fr(sec['Wel_y'], 2)} cm³"] if sec.get("Wel_y") else [])
        return " · ".join([norme] + vals)
    rows = [{"nom": c["section"]["nom"], "col": C.fr(c["masse"], 2), "taux": C.pct(min(c["taux"], 9.99)), "niveau": C.niveau(c["taux"]),
             "cur": c["section"]["nom"] == cur, "opt": best is not None and c["section"]["nom"] == best["section"]["nom"], "prop": prop(c["section"])}
            for c in cands]
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{LIB_FAMILLES[f]}</button>' for f in FAMILLES.get(role, ()))
    if any(f in ("RHS", "SHS", "CHS") for f in FAMILLES.get(role, ())):
        filtres += f'<button type="button" class="fchip{" on" if froid else ""}" data-froid="1">Formés à froid</button>'
    return C.selecteur(role, f"{ET.LIBELLES.get(role, role)} · profilé", "triés par masse · un clic retient la section", filtres, rows,
                       max(0, min(i_o, i_c) - 4), "kg/m")
