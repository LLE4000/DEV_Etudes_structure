# -*- coding: utf-8 -*-
"""Le module « Garde-corps » du site : l'état d'une note (celui du moteur, acier/garde_corps/etat), les champs de
saisie, le résultat, la grande fenêtre de calcul et le sélecteur de profilés — rendus en fragments HTML.

Le site reprend l'écran Streamlit (modules/garde_corps.py) : mêmes champs, mêmes effets de bord à la saisie (le type
pose le système et le remplissage, la catégorie pose q_h), même moteur. Rien n'est recalculé ici : les taux, les
formules et les dessins viennent du moteur ; ce fichier les MET EN FORME.
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
CALCULS = {"montant": "montant", "lisse": "lisse", "barreau": "barreau", "lisse_inter": "barreau", "console": "console", "soudure": "soudure",
           "ancrage": "platine", "platine": "platine"}


# ------------------------------------------------------------------ l'état
def nouvel_etat() -> Dict:
    return ET.nouvel_etat()


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


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les garde-corps de la note : en activer un, en ajouter, copier l'actif, en supprimer un (jamais le dernier)."""
    cas = etat["cas"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(cas) - 1)
    elif action == "ajouter":
        cas.append(ET.nouveau_cas(f"Garde-corps {len(cas) + 1}"))
        etat["actif"] = len(cas) - 1
    elif action == "dupliquer":
        c = copy.deepcopy(cas[etat["actif"]])
        c["nom"] = f"{c.get('nom') or 'Garde-corps'} (copie)"
        cas.append(c)
        etat["actif"] = len(cas) - 1
    elif action == "supprimer" and len(cas) > 1:
        i = min(max(0, int(index or 0)), len(cas) - 1)
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
    nom = etat["cas"][i].get("nom") or f"Garde-corps {i + 1}"
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1], "gouvernant": f"{nom} · {resultats[i]['gouvernant']['libelle']}"}


def elements(etat: Dict, resultats: Optional[List[Dict]] = None) -> str:
    """« Dans cette note » : chaque garde-corps avec son taux ; un clic l'ouvre."""
    items = ""
    resultats = resultats if resultats is not None else [ET.verifier(c) for c in etat["cas"]]
    for i, c in enumerate(etat["cas"]):
        t = _taux(resultats[i])
        txt = f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—"
        sup = (f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["cas"]) > 1 else "")
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(c.get("nom") or f"Garde-corps {i + 1}")} '
                  f'<span>· {C.esc(TY.libelle(c))}</span></span><span class="pct {C.niveau(t)}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["cas"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier le garde-corps ouvert">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter un garde-corps</div></div>')


# ------------------------------------------------------------------ la saisie
def _noms_sections(groupes: Sequence[str]) -> List[str]:
    return [s["nom"] for s in BI.standards().values() if BI.groupe_de(s) in groupes]


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Le cadre Données : les sections repliables, chaque champ lié à un chemin de l'état."""
    cas = _cas(etat)
    res = res if res is not None else ET.verifier(cas)
    g, ch, rp, fx, m = cas["geometrie"], cas["charges"], cas["remplissage"], cas["fixation"], cas["methode"]
    L = C.ligne
    # 1 · le garde-corps
    s1 = (L("Nom", C.texte(["nom"], cas.get("nom"), "ex. G1 Terrasse"), full=True)
          + L("Type", C.choix(["type"], cas.get("type"), [(k, v["libelle"]) for k, v in TY.TYPES.items()]), full=True)
          + C.aide(TY.TYPES[TY.type_de(cas)]["description"])
          + L("Forme", C.segments(["geometrie", "forme"], g.get("forme"), [(k, v[:1].upper() + v[1:]) for k, v in GE.FORMES.items()]), full=True)
          + L("hauteur", C.num(["geometrie", "H"], g.get("H"), "mm"), "H",
              "Hauteur de l'axe de la main courante au-dessus de la platine ; la charge s'applique au plus à 1,20 m (EN 1991-1-1 §6.4).")
          + L("pente", C.num(["geometrie", "pente"], g.get("pente"), "°", mini=0, maxi=60), "α", "Garde-corps d'escalier ; 0 : horizontal."))
    # 2 · les tronçons
    n = GE.N_TRONCONS.get(g.get("forme", "droit"), 1)
    s2 = ""
    for i in range(n):
        tr = g["troncons"][i]
        s2 += (f'<div class="grp">Tronçon {i + 1}</div>'
               + L("longueur", C.num(["geometrie", "troncons", i, "L"], tr.get("L"), "mm"), "L")
               + L("entraxe maxi", C.num(["geometrie", "troncons", i, "s_max"], tr.get("s_max"), "mm"), "s",
                   "La longueur est répartie en travées égales ne dépassant pas cet entraxe, sauf si des entraxes sont saisis.")
               + L("Entraxes imposés", C.texte(["geometrie", "troncons", i, "entraxes"], " ".join(f"{e:g}" for e in tr.get("entraxes") or []),
                                                "réguliers"), full=True))
    s2 += (L("Début", C.choix(["geometrie", "debut"], g.get("debut"), list(GE.EXTREMITES.items())), full=True)
           + L("Fin", C.choix(["geometrie", "fin"], g.get("fin"), list(GE.EXTREMITES.items())), full=True))
    geo = res.get("geometrie") or {}
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
                  full=True, aide="Elle fixe la portée des barreaux ; elle n'est pas vérifiée."))
        r = res.get("remplissage") or {}
        if r.get("L"):
            s3 += C.aide(f"Portée des barreaux : {C.fr(r['L'], 0)} mm ; leur section se choisit dans le résultat.")
        resume3 = f"barreaux · e {C.nb(rp.get('entraxe'))} mm"
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
        if sd.get("jonction") == "angle":
            s5 += L("gorge", C.num(["fixation", "soudure", "a"], sd.get("a"), "mm"), "a")
        elif sd.get("jonction") == "partielle":
            s5 += (L("pénétration", C.num(["fixation", "soudure", "p"], sd.get("p"), "mm"), "p")
                   + L("renfort", C.num(["fixation", "soudure", "renfort"], sd.get("renfort"), "mm"), "a<sub>r</sub>", "0 : sans renfort."))
        s5 += L("épaisseur de platine", C.num(["fixation", "platine", "tp"], fx["platine"].get("tp"), "mm"), "t<sub>p</sub>")
        if typ in ("platine", FX.LATERALE):
            pl, an = fx["platine"], fx["ancrage"]
            s5 += '<div class="grp">Platine et ancrage</div>'
            if lat:
                s5 += L("descente", C.num(["fixation", "platine", "d_h"], pl.get("d_h"), "mm"), "d<sub>h</sub>", "Du dessus de dalle au haut de la platine.")
            s5 += (L("dans le sens de l'effort" if not lat else "hauteur", C.num(["fixation", "platine", "bp"], pl.get("bp"), "mm"), "b<sub>p</sub>")
                   + L("largeur", C.num(["fixation", "platine", "lp"], pl.get("lp"), "mm"), "l<sub>p</sub>")
                   + L("entraxe des rangées", C.num(["fixation", "ancrage", "ex"], an.get("ex"), "mm"), "e<sub>x</sub>")
                   + L("par rangée", C.num(["fixation", "ancrage", "n_par_rangee"], an.get("n_par_rangee"), "", entier=True, mini=1, maxi=4), "n")
                   + L("entraxe dans la rangée", C.num(["fixation", "ancrage", "ey"], an.get("ey"), "mm"), "e<sub>y</sub>")
                   + L("Ancrage", C.segments(["fixation", "ancrage", "type"], an.get("type"), [("cheville", "Chevilles"), ("boulon", "Boulons")]), full=True))
            if an.get("type") == "boulon":
                s5 += (L("Boulon", C.choix(["fixation", "ancrage", "boulon"], an.get("boulon"), [(b, b) for b in FX.boulons()]), full=True)
                       + L("Classe", C.choix(["fixation", "ancrage", "classe"], an.get("classe"), [(c, c) for c in FX.classes()]), full=True))
            else:
                s5 += (L("Cheville", C.texte(["fixation", "ancrage", "nom"], an.get("nom"), "désignation"), full=True)
                       + L("traction, par cheville", C.num(["fixation", "ancrage", "NRd"], an.get("NRd"), "kN"), "N<sub>Rd</sub>",
                           "Valeur de calcul en traction de l'ETA, pour le béton et la profondeur du projet.")
                       + L("cisaillement, par cheville", C.num(["fixation", "ancrage", "VRd"], an.get("VRd"), "kN"), "V<sub>Rd</sub>"))
    # 6 · la méthode
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
        s6 += (L("charges", C.num(["methode", "gamma_Q"], m.get("gamma_Q"), ""), "γ<sub>Q</sub>")
               + L("sections", C.num(["methode", "gamma_M0"], m.get("gamma_M0"), ""), "γ<sub>M0</sub>")
               + L("boulons, cordons", C.num(["methode", "gamma_M2"], m.get("gamma_M2"), ""), "γ<sub>M2</sub>")
               + L("Critère des cordons", C.choix(["methode", "ec"], m.get("ec"), [("directionnelle", "directionnelle"), ("simplifiee", "simplifiée")]),
                   full=True))
    trs = GE.longueurs_troncons(g)
    resume2 = f"{C.fr(sum(trs) / 1000, 2)} m · {geo.get('phrase', '')}" if trs else ""
    return (C.section(1, "Garde-corps", f"{TY.libelle(cas)} · H {C.nb(g.get('H'))}", s1, True, "gc")
            + C.section(2, "Tronçons", resume2, s2, False, "tr")
            + C.section(3, "Remplissage", resume3, s3, False, "rp")
            + C.section(4, "Charges", f"{CH.libelle_categorie(ch.get('categorie'))} · q<sub>h</sub> {C.nb(ch.get('qh'))} kN/m", s4, True, "ch")
            + C.section(5, "Fixation", FIXATIONS_COURTES.get(typ, ""), s5, False, "fx")
            + C.section(6, "Méthode", f"{MODES.get(m.get('mode'))} · {m.get('nuance')} · flèche L/{C.nb(m.get('fleche_val'))}", s6, False, "me"))


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


def resultat(etat: Dict, avec_optimaux: bool = True) -> Dict:
    """Le calcul et sa mise en forme : le cadre Résultat, les fenêtres de calcul, la toile des dessins, le résumé de la
    note (taux, niveau, statut) ; et le cadre Données (ses champs dépendent du résultat : portée des barreaux…)."""
    cas = _cas(etat)
    res = ET.verifier(cas)
    stat = res["statut"]
    g = res.get("gouvernant")
    if stat == ET.REFUS or not g:
        niv = "x"
        verdict = f'<div class="verdict"><div class="gov"><b>Calcul impossible</b>{C.esc((res.get("messages") or ["données manquantes"])[0])}</div></div>'
    else:
        niv = C.niveau(res["taux_max"])
        el_g = next((e for e in res["elements"] if e["cle"] == g["cle"]), {})
        cas_g = (el_g.get("detail") or {}).get("cas_charge") if isinstance(el_g.get("detail"), dict) else None
        crit = ME.critere_texte(g["critere"]) if g["cle"] in ET.ROLES else g["critere"]
        verdict = (f'<div class="verdict">{C.chip(C.pct(res["taux_max"]), niv, xl=True)}<div class="gov"><b>{C.esc(g["libelle"].split(" · ")[0])}</b>'
                   f'{C.esc(crit)}{(" · " + C.symb_math(CH.CAS_LIBELLES.get(cas_g, cas_g))) if cas_g else ""}</div></div>')
    resultats = [res if i == etat["actif"] else ET.verifier(c) for i, c in enumerate(etat["cas"])]
    opt = _optimaux(cas) if avec_optimaux and stat != ET.REFUS else {}
    blocs = _blocs(res)
    roles = ET.roles_du_cas(cas)
    rows = ""
    for e in res.get("elements", []):
        parts = e["libelle"].split(" · ")
        nom = parts[0]
        dim = e["cle"] in ET.ROLES
        if dim:
            piece = parts[1] if len(parts) > 1 else ""
        elif nom.startswith("Platine "):
            nom, piece = "Platine", nom[len("Platine "):]
        else:
            piece = " · ".join(parts[1:]).replace("cordon d'angle ", "gorge ").replace(" chevilles", "")
        o = opt.get(e["cle"])
        leger = f"Plus léger qui tient : {o['nom']} · {o['taux']} %" if o and not o["retenu"] else ""
        cal = CALCULS.get(e["cle"], "")
        if cal and not (blocs.get(cal) or {}).get("groupes"):
            cal = ""
        rows += C.ligne_res(nom, C.esc(piece), C.pct(e["taux"]), C.niveau(e["taux"]), calcul=cal, pk=e["cle"] if e["cle"] in roles else "",
                            gov=bool(g and e["cle"] == g["cle"]), leger=leger)
    messages = "".join(C.message(C.esc(msg), msg.startswith("Refus") or "DÉPASSÉ" in msg) for msg in res.get("messages", []))
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse indicative de l\'acier</span><b>{C.fr(res.get("masse") or 0, 0)}{C.NBSP}kg</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    html_res = f'<div class="pbody">{verdict}<div class="els">{rows}</div>{messages}</div>{pied}'
    # les fenêtres de calcul
    hyps = _hypotheses(res) if stat != ET.REFUS else []
    titres = {"montant": "Montant principal", "lisse": "Main courante", "barreau": "Remplissage", "console": "Console", "platine": "Fixation · platine et ancrages",
              "soudure": "Soudure du montant sur la platine"}
    volets = ""
    for k, b in blocs.items():
        if not b.get("groupes"):
            continue
        lignes = _tableau(k, b)
        sous = f"{C.symb_math(b['cas'])} · les formules de la note" if b.get("cas") else "les formules de la note"
        pied_ck = f'<div class="ck-autres">Autres cas : {C.symb_math(b["autres"])}</div>' if b.get("autres") else ""
        volets += C.fenetre_calcul(k, titres.get(k, k), sous, hyps, lignes, _detail(b), pied_ck)
    # la toile : une feuille par vue du moteur
    vues = [(k, t, SC.svg(d, f"gc_{k}")) for k, t, d in SC.vues(res)] if stat != ET.REFUS else []
    return {"resultat": html_res, "volets": volets, "vues": [{"cle": k, "titre": t, "court": VUES_COURTES.get(k, t), "svg": s} for k, t, s in vues],
            "phrase": f'{C.esc(res.get("type_libelle", ""))} · {C.esc((res.get("geometrie") or {}).get("phrase", ""))}',
            "statut_html": C.statut(niv), "resume": resume_note(etat, resultats), "donnees": formulaire(etat, res),
            "elements": elements(etat, resultats)}


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
