# -*- coding: utf-8 -*-
"""La page « BASE DES PRIX » du module « Prix des grues » (cahier § 4) — trois colonnes :

- à GAUCHE : les loueurs (écusson du dossier, pourcentage de prix à jour, prix manquants) ; sous le loueur choisi, ses
  catalogues, « Accessoires et services », « Questions » ; les filtres du tableau (écusson, recherche) ; « Réglages » et
  « Historique » ; l'import d'une base (JSON) ;
- au CENTRE : le tableau choisi. Un catalogue : une ligne par grue, une colonne par champ de prix, un point d'écusson par
  case — on corrige dans la case (la case passe en « Corrigé », la valeur du loueur reste dessous) ; les accessoires :
  une ligne par poste, une colonne par catalogue ; les questions ; les réglages ; l'historique ;
- à DROITE : le dossier du loueur (écusson, années des tarifs, prix manquants, questions ouvertes) et la case choisie
  (valeur, unité, année, validité, source, historique de la case ; « Revenir au prix du loueur », « Marquer à
  vérifier », le motif de notre correction).

Rien n'est calculé ici : les écussons et les dossiers viennent de ``grues.prix`` ; les modifications passent par
``grues.prix.edition`` (chacune écrite dans l'historique)."""
import datetime as dt
from typing import Dict, List, Optional, Tuple

from grues.prix import base as B
from grues.prix import dossiers as D
from grues.prix import ecussons as E
from grues.prix import edition as ED
from grues.prix import valeurs as V

from . import commun as C
from . import etabli_compact as EC
from . import prix_grues_commun as PC

FILTRES = (("", "Tous les prix"), (E.MANQUANT, "Manquants"), (E.A_VERIFIER, "À vérifier"), (E.PAS_A_JOUR, "Pas à jour"),
           (E.CORRIGE, "Corrigés"))
TABLEAUX_FIXES = (("postes", "Accessoires et services"), ("questions", "Questions"))
HISTORIQUE_AFFICHE = 200
CHAMPS_PARC = ("nb_grues_chez_loueur", "nb_disponibles")


# ------------------------------------------------------------------ lecture d'une valeur de case
def texte_valeur(v) -> str:
    n = V.nature(v)
    if n == V.INCONNU:
        return ""
    if n == V.COMPRIS:
        return "compris"
    if n == V.NON_DISPONIBLE:
        return "non disponible"
    if n == V.SUR_DEMANDE:
        return "sur demande"
    if n == V.NOMBRE:
        return PC.nombre(float(v), 3)
    return str(v)


def _ref_txt(ref) -> str:
    return "|".join(str(x) for x in ref)


def _ref_de(base: Dict, sorte: str, ident: str, cle: str) -> Tuple:
    """La référence d'une case depuis l'écran : (« grue », id, champ) ou (« poste », code, nom du catalogue)."""
    if sorte == "grue":
        return ("grue", ident, cle)
    if sorte == "poste":
        cats = B.catalogues(base)
        try:
            return ("poste", ident, cats[int(cle)]["nom"])
        except (ValueError, IndexError):
            if B.catalogue(base, cle):
                return ("poste", ident, cle)
            raise ValueError("catalogue inconnu")
    raise ValueError("case inconnue")


def _rang_cas(base: Dict, cas: str) -> int:
    return next(i for i, c in enumerate(B.catalogues(base)) if c["nom"] == cas)


# ------------------------------------------------------------------ les saisies
def modifier(etat: Dict, base: Dict, ch: List[str], valeur, auj: dt.date) -> Optional[str]:
    vb = etat["vb"]
    ini = etat.get("initiales") or ""
    if ch[0] == "vb":
        k = ch[1] if len(ch) > 1 else ""
        if k == "loueur":
            if not B.loueur(base, valeur):
                raise ValueError("loueur inconnu")
            cats = B.catalogues_du_loueur(base, valeur)
            vb.update(loueur=valeur, tableau=cats[0]["nom"] if cats else "postes", case=None)
        elif k == "tableau":
            permis = [c["nom"] for c in B.catalogues(base)] + [t for t, _ in TABLEAUX_FIXES] + ["reglages", "historique"]
            if valeur not in permis:
                raise ValueError("tableau inconnu")
            vb.update(tableau=valeur, case=None)
        elif k == "filtre":
            if valeur not in dict(FILTRES):
                raise ValueError("filtre inconnu")
            vb["filtre"] = valeur
        elif k == "recherche":
            vb["recherche"] = " ".join(str(valeur or "").split())[:40]
        elif k == "tout":
            vb["tout"] = bool(C.lire_bool(valeur))
        elif k == "case":
            v = str(valeur or "")
            if v.count("|") != 2:
                vb["case"] = None
            else:
                sorte, ident, cle = v.split("|")
                vb["case"] = list(_ref_de(base, sorte, ident, cle))
        else:
            raise ValueError("champ inconnu")
        return None
    # ch[0] == "b" : une modification de la base
    k = ch[1] if len(ch) > 1 else ""
    if k in ("grue", "poste") and len(ch) == 4:
        ref = _ref_de(base, k, ch[2], ch[3])
        vb["case"] = list(ref)
        actuel = ED._prix(base, ref)
        if isinstance(valeur, str) and not valeur.strip():
            return ED.revenir(base, ref, ini, auj) if actuel.get("corrige") else None
        nv = ED.lire_valeur(valeur)
        if nv == actuel.get("valeur") or (V.est_nombre(nv) and V.est_nombre(actuel.get("valeur")) and abs(nv - actuel["valeur"]) < 1e-9):
            return None
        if not actuel.get("corrige") and (nv == actuel.get("valeur")):
            return None
        return ED.corriger(base, ref, valeur, "", ini, auj)
    if k in ("revenir", "verifier", "motif") and len(ch) == 5:
        ref = _ref_de(base, ch[2], ch[3], ch[4])
        if k == "revenir":
            return ED.revenir(base, ref, ini, auj)
        if k == "verifier":
            return ED.marquer(base, ref, bool(C.lire_bool(valeur)), ini, auj)
        o, cle, _ = ED._cible(base, ref)
        c = (o.get("corrections") or {}).get(cle)
        if not c:
            raise ValueError("ce prix n'est pas corrigé")
        c["motif"] = " ".join(str(valeur or "").split())[:200]
        return None
    if k == "retirer" and len(ch) == 3:
        return ED.retirer_grue(base, ch[2], bool(C.lire_bool(valeur)), ini, auj)
    if k == "reglage" and len(ch) in (3, 4):
        annee = int(ch[3]) if len(ch) == 4 else None
        v = valeur
        if ch[2] == "conges":
            v = [[x.strip() for x in p.split("→")] for p in str(valeur or "").replace("\n", ";").split(";") if "→" in p]
        return ED.regler(base, ch[2], v, annee, ini, auj)
    if k == "question" and len(ch) == 5:
        l = B.loueur(base, B.loueurs(base)[int(ch[2])]["nom"]) if ch[2].isdigit() and int(ch[2]) < len(B.loueurs(base)) else None
        if not l:
            raise ValueError("loueur inconnu")
        q = next((q for q in l.get("questions") or [] if q.get("id") == ch[3]), None)
        if not q:
            raise ValueError("question inconnue")
        if ch[4] == "reponse":
            q["reponse"] = str(valeur or "").strip()[:1000] or None
        elif ch[4] == "question":
            q["question"] = str(valeur or "").strip()[:1000]
        elif ch[4] == "traitee":
            q["traitee"] = bool(C.lire_bool(valeur))
        else:
            raise ValueError("champ inconnu")
        ED.historiser(base, f"question {ch[3]}", ch[4], None, valeur, "questions", "", ini, auj)
        return None
    raise ValueError("champ inconnu")


# ------------------------------------------------------------------ à gauche
def _bouton(chemin: str, valeur: str, corps: str, on: bool, titre: str = "") -> str:
    t = f' title="{C.esc(titre)}"' if titre else ""
    return (f'<button type="button" class="pg-bt{" on" if on else ""}" data-chemin="{C.esc(chemin)}" data-sorte="choix" '
            f'data-valeur="{C.esc(valeur)}"{t}>{corps}</button>')


def gauche(etat: Dict, base: Dict, auj: dt.date, dossiers: Dict[str, Dict]) -> str:
    vb = etat["vb"]
    lou = ""
    for l in B.loueurs(base):
        d = dossiers[l["nom"]]
        corps = (f'{PC.point(d["ecusson"])}<span class="nm">{C.esc(l["nom"])}</span>'
                 f'<span class="pc">{d["pct_a_jour"]} %</span><span class="mq">{d["manquants"]} ✗</span>')
        lou += _bouton("vb/loueur", l["nom"], corps, l["nom"] == vb.get("loueur"),
                       f"{E.LIBELLES.get(d['ecusson'])} · {d['pct_a_jour']} % à jour · {d['manquants']} prix manquants")
    out = C.section(1, "Loueurs", C.esc(vb.get("loueur") or "choisir"), f'<div class="pg-liste-bt">{lou}</div>', True, "loueurs")
    if vb.get("loueur"):
        tabs = ""
        for c in B.catalogues_du_loueur(base, vb["loueur"]):
            n = len(B.grues_du(base, c["nom"]))
            tabs += _bouton("vb/tableau", c["nom"], f'<span class="nm">{C.esc(c.get("type") or c["nom"])}</span><span class="pc">{n}</span>',
                            vb.get("tableau") == c["nom"], c["nom"])
        for t, lib in TABLEAUX_FIXES:
            tabs += _bouton("vb/tableau", t, f'<span class="nm">{C.esc(lib)}</span>', vb.get("tableau") == t)
        out += C.section(2, C.esc(vb["loueur"]), C.esc(_nom_tableau(base, vb.get("tableau"))), f'<div class="pg-liste-bt">{tabs}</div>', True, "tableaux")
    filtres = (C.ligne("Écusson", C.choix(["vb", "filtre"], vb.get("filtre") or "", FILTRES))
               + C.ligne("Recherche", C.texte(["vb", "recherche"], vb.get("recherche") or "", "modèle, poste…"))
               + C.ligne("", C.case(["vb", "tout"], bool(vb.get("tout")), "montrer les grues retirées")))
    out += C.section(3, "Filtre", C.esc(dict(FILTRES).get(vb.get("filtre") or "", "") if vb.get("filtre") else "aucun"), filtres, False, "filtre")
    autres = (_bouton("vb/tableau", "reglages", f'<span class="nm">Réglages</span>', vb.get("tableau") == "reglages")
              + _bouton("vb/tableau", "historique", f'<span class="nm">Historique</span><span class="pc">{len(base.get("historique") or [])}</span>',
                        vb.get("tableau") == "historique"))
    out += C.section(4, "Réglages et historique", "", f'<div class="pg-liste-bt">{autres}</div>'
                     + C.ligne("Initiales", C.texte(["initiales"], etat.get("initiales") or "", "facultatif"),
                               aide="Écrites dans l'historique ; gardées dans cette note, ce ne sont pas des comptes"), False, "reglages")
    out += C.section(5, "Importer", "JSON", C.ligne("Base", PC.import_base("Remplacer la base (JSON)"))
                     + C.aide("Une sauvegarde complète remplace la base de cet appareil."), False, "importer")
    return out


def _nom_tableau(base: Dict, t: Optional[str]) -> str:
    if not t:
        return ""
    c = B.catalogue(base, t)
    if c:
        return c.get("type") or t
    return dict(TABLEAUX_FIXES).get(t) or {"reglages": "Réglages", "historique": "Historique"}.get(t, t)


# ------------------------------------------------------------------ au centre
def _case(ref: Tuple, p: Dict, auj: dt.date, besoin: bool, sel: Optional[List], interne: bool = False) -> Tuple[str, Optional[str]]:
    ec = E.ecusson(p, auj, besoin=besoin)
    v = texte_valeur(p.get("valeur"))
    chemin = "b/" + (f"grue/{ref[1]}/{ref[2]}" if ref[0] == "grue" else f"poste/{ref[1]}/{p.get('_rang')}")
    choisi = sel is not None and list(ref) == list(sel)
    cls = "pg-c" + (" corr" if p.get("corrige") else "") + (" manq" if ec == E.MANQUANT else "") + (" sel" if choisi else "") + (" int" if interne else "")
    valeur_sel = f"{ref[0]}|{ref[1]}|{ref[2] if ref[0] == 'grue' else p.get('_rang')}"
    return (f'<td class="{cls}"><div class="pg-cw"><button type="button" class="pg-cpt" data-agir="1" data-chemin="vb/case" '
            f'data-valeur="{C.esc(valeur_sel)}" title="Voir cette case">{PC.point(ec)}</button>'
            f'<input class="pg-in" data-agir="1" data-chemin="{C.esc(chemin)}" data-coller="{C.esc(chemin)}" value="{C.esc(v)}" '
            f'inputmode="decimal" aria-label="Prix"></div></td>'), ec


def _filtrer(ecs: List[Optional[str]], filtre: str) -> bool:
    return not filtre or filtre in ecs


def tableau_catalogue(base: Dict, vb: Dict, cas: str, auj: dt.date) -> str:
    lg = B.logique(base, cas)
    champs = [c for c in B.champs(base, lg) if c.get("categorie") == "prix" or c.get("cle") in CHAMPS_PARC]
    grues = B.grues_du(base, cas, retirees=bool(vb.get("tout")))
    cherche = (vb.get("recherche") or "").lower()
    entete = "".join(f'<th class="n{" int" if c.get("fourni_par") == "interne" else ""}" title="{C.esc(c.get("libelle"))} · {C.esc(c.get("unite"))}'
                     f'{" · interne" if c.get("fourni_par") == "interne" else ""}">{C.esc(c.get("libelle"))}<small>{C.esc(c.get("unite"))}</small></th>'
                     for c in champs)
    rows = ""
    n_vus = 0
    for g in grues:
        if cherche and cherche not in str(g.get("modele") or "").lower():
            continue
        cases, ecs = "", []
        for c in champs:
            p = B.prix_grue(g, c["cle"])
            besoin = c.get("categorie") == "prix" and c.get("fourni_par") != "interne" and bool(c.get("essentiel"))
            h, ec = _case(("grue", g["id"], c["cle"]), p, auj, besoin, vb.get("case"), c.get("fourni_par") == "interne")
            cases += h
            ecs.append(ec if c.get("categorie") == "prix" else None)
        if not _filtrer(ecs, vb.get("filtre") or ""):
            continue
        n_vus += 1
        ecg = E.agreger([E.pour_ensemble(e, None, auj) for e in ecs if e], vide=E.MANQUANT)
        ret = B.retiree(g)
        bt = (f'<button type="button" class="pg-x" data-agir="1" data-chemin="b/retirer/{C.esc(g["id"])}" data-valeur="{"0" if ret else "1"}" '
              f'title="{"Remettre dans les listes" if ret else "Retirer des listes (les anciens calculs la gardent)"}">'
              f'{C.icon("undo-2" if ret else "minus")}</button>')
        rows += (f'<tr class="{"ret" if ret else ""}"><th class="pg-gn">{PC.point(ecg)}<span>{C.esc(g.get("modele"))}</span>'
                 f'<small>{C.esc(g["id"])}{" · retirée" if ret else ""}</small>{bt}</th>{cases}</tr>')
    vide = '' if n_vus else f'<tr><td colspan="{len(champs) + 1}" class="pg-info">Aucune grue{" pour ce filtre" if vb.get("filtre") or cherche else ""}.</td></tr>'
    return (f'<div class="pg-tab-w"><table class="pg-tab pg-base"><thead><tr><th>Grue</th>{entete}</tr></thead>'
            f'<tbody>{rows}{vide}</tbody></table></div>')


def postes_du_loueur(base: Dict, nom: str) -> Tuple[List[Dict], List[str]]:
    """(les postes que le tableau des accessoires du loueur montre, ses colonnes = catalogues)."""
    cats = [c["nom"] for c in B.catalogues_du_loueur(base, nom)]
    ps, cols = [], list(cats)
    for p in B.postes(base):
        cas = list((p.get("prix_par_cas") or {}).keys())
        if p.get("fourni_par") == nom:
            ps.append(p)
            cols += [c for c in cas if c not in cols]
        elif any(c in cats for c in cas) and p.get("code") != "FONDATION":
            ps.append(p)
    return ps, cols


def tableau_postes(base: Dict, vb: Dict, nom: str, auj: dt.date) -> str:
    ps, cols = postes_du_loueur(base, nom)
    cherche = (vb.get("recherche") or "").lower()
    entete = "".join(f'<th class="n">{C.esc((B.catalogue(base, c) or {}).get("type") or c)}<small>{C.esc(c)}</small></th>' for c in cols)
    rows = ""
    for p in ps:
        if cherche and cherche not in (str(p.get("poste")) + " " + str(p.get("code"))).lower():
            continue
        cases, ecs = "", []
        for cas in cols:
            if cas not in (p.get("prix_par_cas") or {}):
                cases += '<td class="pg-c na"></td>'
                continue
            pr = B.prix_poste(p, cas)
            pr["_rang"] = _rang_cas(base, cas)
            h, ec = _case(("poste", p["code"], cas), pr, auj, p.get("fourni_par") != "interne", vb.get("case"), p.get("fourni_par") == "interne")
            cases += h
            ecs.append(ec)
        if not _filtrer(ecs, vb.get("filtre") or ""):
            continue
        it = ' <small class="pg-tag">interne</small>' if p.get("fourni_par") == "interne" else ""
        rows += (f'<tr><th class="pg-gn"><span>{C.esc(p.get("poste"))}</span><small>{C.esc(p.get("code"))} · {C.esc(p.get("unite"))}{it}</small></th>'
                 f'{cases}</tr>')
    return (f'<div class="pg-tab-w"><table class="pg-tab pg-base"><thead><tr><th>Poste</th>{entete}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def tableau_questions(base: Dict, nom: str) -> str:
    li = next((i for i, l in enumerate(B.loueurs(base)) if l["nom"] == nom), None)
    l = B.loueurs(base)[li] if li is not None else {}
    out = ""
    for q in l.get("questions") or []:
        qid = C.esc(q.get("id"))
        out += (f'<div class="pg-q{" ok" if (q.get("reponse") or q.get("traitee")) else ""}"><div class="pg-q-h"><b>{qid}</b>'
                f'<label class="case"><input type="checkbox" data-agir="1" data-chemin="b/question/{li}/{qid}/traitee"{" checked" if q.get("traitee") else ""}>'
                f'<span>traitée</span></label></div>'
                f'<textarea class="pg-in t" data-agir="1" data-chemin="b/question/{li}/{qid}/question" rows="2">{C.esc(q.get("question") or "")}</textarea>'
                + (f'<small>Ce que nous savons : {C.esc(q.get("info_actuelle"))}</small>' if q.get("info_actuelle") else "")
                + f'<textarea class="pg-in t" data-agir="1" data-chemin="b/question/{li}/{qid}/reponse" rows="2" placeholder="réponse du loueur">'
                  f'{C.esc(q.get("reponse") or "")}</textarea></div>')
    return f'<div class="pg-qs">{out or "<p class=pg-info>Aucune question.</p>"}</div>'


def tableau_reglages(base: Dict) -> str:
    r = B.reglages(base)
    an = dt.date.today().year

    def champ(cle, unite, aide=""):
        v = r.get(cle)
        return C.ligne(C.esc(LIBELLES_REGLAGES[cle]), _num_agir(f"b/reglage/{cle}", v, unite), aide=aide)
    out = (champ("jours_par_mois", "j/mois") + champ("semaines_par_mois", "sem./mois") + champ("heures_normales", "h/j")
           + C.ligne("Jours du grutier", _choix_agir("b/reglage/jours_grutier", r.get("jours_grutier"), [("20 par mois", "20 par mois"),
                                                                                                        ("calendrier réel", "calendrier réel")])))
    out += '<div class="pg-sous">Par année</div>'
    annees = sorted({int(a) for a in list((r.get("prix_kwh") or {}).keys()) + list((r.get("taux_mo") or {}).keys()) if str(a).isdigit()} | {an, an + 1})
    for a in annees:
        out += C.ligne2((f"kWh {a}", _num_agir(f"b/reglage/prix_kwh/{a}", (r.get("prix_kwh") or {}).get(str(a)), "€/kWh"), ""),
                        (f"MO {a}", _num_agir(f"b/reglage/taux_mo/{a}", (r.get("taux_mo") or {}).get(str(a)), "€/h"), "Taux horaire de notre main-d'œuvre"))
        cg = "; ".join(f"{p[0]} → {p[1]}" for p in (r.get("conges") or {}).get(str(a)) or [])
        out += C.ligne(f"Congés {a}", f'<div class="field text"><input class="pg-in t" data-agir="1" data-chemin="b/reglage/conges/{a}" '
                                       f'value="{C.esc(cg)}" placeholder="2026-07-20 → 2026-08-07; …"></div>',
                       aide="Congés du bâtiment : périodes du → au, séparées par « ; »")
    out += '<div class="pg-sous">Fondations et imports</div>'
    out += (champ("taux_actualisation", "%/an", "Vide : pas d'actualisation des retours des années passées")
            + champ("coef_fondation_poutres", "×") + champ("coef_fondation_pieux", "×") + champ("arrondi_fondation_pieux", "€")
            + champ("seuil_ecart", "%") + champ("delai_relance", "j") + champ("mois_rappel", "mois"))
    return f'<div class="pg-form">{out}</div>'


LIBELLES_REGLAGES = {"jours_par_mois": "Jours par mois", "semaines_par_mois": "Semaines par mois", "heures_normales": "Heures normales",
                     "taux_actualisation": "Actualisation des fondations", "coef_fondation_poutres": "Poutres / dalle",
                     "coef_fondation_pieux": "Pieux / dalle", "arrondi_fondation_pieux": "Arrondi des pieux",
                     "seuil_ecart": "Seuil d'écart à l'import", "delai_relance": "Relance d'une demande", "mois_rappel": "Rappel annuel des loueurs"}


def _num_agir(chemin: str, v, unite: str) -> str:
    t = PC.nombre(v, 4) if V.est_nombre(v) else ""
    return (f'<div class="field"><input class="pg-in" data-agir="1" data-chemin="{C.esc(chemin)}" inputmode="decimal" value="{C.esc(t)}">'
            f'<span class="u">{C.esc(unite)}</span></div>')


def _choix_agir(chemin: str, v, options) -> str:
    return C.choix(chemin.split("/"), v, options).replace("data-chemin=", 'data-agir="1" data-chemin=', 1)


def tableau_historique(base: Dict) -> str:
    h = list(reversed(base.get("historique") or []))[:HISTORIQUE_AFFICHE]
    rows = "".join(f'<tr><td>{C.esc(x.get("date"))}</td><td>{C.esc(x.get("objet"))}</td><td>{C.esc(x.get("champ"))}</td>'
                   f'<td class="n">{C.esc(texte_valeur(x.get("avant")) if not isinstance(x.get("avant"), str) else x.get("avant"))}</td>'
                   f'<td class="n">{C.esc(texte_valeur(x.get("apres")) if not isinstance(x.get("apres"), str) else x.get("apres"))}</td>'
                   f'<td>{C.esc(x.get("origine"))}</td><td>{C.esc(x.get("motif"))}</td><td>{C.esc(x.get("initiales"))}</td></tr>' for x in h)
    vide = "" if h else '<tr><td colspan="8" class="pg-info">Aucun changement pour l\'instant.</td></tr>'
    return (f'<div class="pg-tab-w"><table class="pg-tab pg-hist"><thead><tr><th>Date</th><th>Objet</th><th>Champ</th><th class="n">Avant</th>'
            f'<th class="n">Après</th><th>Origine</th><th>Motif</th><th>Init.</th></tr></thead><tbody>{rows}{vide}</tbody></table></div>')


def centre(etat: Dict, base: Dict, auj: dt.date) -> Tuple[str, str]:
    vb = etat["vb"]
    t = vb.get("tableau")
    if t == "reglages":
        return "Réglages", tableau_reglages(base)
    if t == "historique":
        return "Historique", tableau_historique(base)
    if not vb.get("loueur"):
        return "Base des prix", '<div class="pg-vide"><div class="pg-vide-c"><b>Choisir un loueur à gauche</b></div></div>'
    if t == "postes":
        return f"Accessoires et services – {vb['loueur']}", tableau_postes(base, vb, vb["loueur"], auj)
    if t == "questions":
        return f"Questions – {vb['loueur']}", tableau_questions(base, vb["loueur"])
    if t and B.catalogue(base, t):
        return t, tableau_catalogue(base, vb, t, auj)
    return vb["loueur"], ""


# ------------------------------------------------------------------ à droite
def _kv(k: str, v: str) -> str:
    return f'<div class="pg-kv"><span>{C.esc(k)}</span><b>{v}</b></div>'


def droite(etat: Dict, base: Dict, auj: dt.date, dossiers: Dict[str, Dict]) -> str:
    vb = etat["vb"]
    out = ""
    el = D.electricite(base, auj)
    kwh = el["prix"]
    out += (f'<div class="pg-bloc"><div class="pg-bloc-h">Électricité {PC.ecusson(el["ecusson"], petit=True)}</div>'
            + _kv("Prix du kWh", C.esc(f"{PC.nombre(kwh['valeur'], 4)} €/kWh ({kwh['annee']})") if kwh else "—") + '</div>')
    if vb.get("loueur"):
        d = dossiers[vb["loueur"]]
        an = d["annees"]
        out += (f'<div class="pg-bloc"><div class="pg-bloc-h">Dossier {C.esc(vb["loueur"])}</div>'
                f'<div class="pg-ec-l">{PC.ecusson(d["ecusson"])}<span>{C.esc(("liste " + "/".join(str(a) for a in an)) if an else "aucun prix reçu")}</span></div>'
                + _kv("Prix à jour", f"{d['pct_a_jour']} %") + _kv("Prix connus", str(d["connus"]))
                + _kv("Prix manquants", f"{d['manquants']}" + (f" (dont {d['essentiels_manquants']} essentiels)" if d["essentiels_manquants"] else ""))
                + _kv("Questions ouvertes", str(d["questions_ouvertes"])) + '</div>')
    case = vb.get("case")
    if case:
        try:
            ref = tuple(case)
            p = ED._prix(base, ref)
            o, cle, nom = ED._cible(base, ref)
        except (ValueError, KeyError, TypeError):
            p = None
        if p is not None:
            unite = ""
            if ref[0] == "grue":
                ch = B.champ(base, B.logique(base, o.get("cas")) or "", ref[2]) or {}
                unite, lib = ch.get("unite") or "", ch.get("libelle") or ref[2]
                rang = ref[2]
            else:
                unite, lib = o.get("unite") or "", o.get("poste") or ref[1]
                rang = str(_rang_cas(base, ref[2]))
            ident = f"{ref[0]}/{ref[1]}/{rang}"
            ec = E.ecusson(p, auj)
            hist = [x for x in base.get("historique") or [] if x.get("objet") == nom and x.get("champ") in (ref[2], ref[1])][-5:]
            corps = (_kv("Valeur", C.esc(texte_valeur(p.get("valeur")) or "vide") + f" <small>{C.esc(unite)}</small>")
                     + _kv("Année", C.esc(p.get("annee") or "inconnue")) + _kv("Source", C.esc(p.get("source") or "—"))
                     + (_kv("Validité", C.esc(f"{(p.get('validite') or {}).get('du') or '…'} → {(p.get('validite') or {}).get('au') or '…'}")) if p.get("validite") else "")
                     + _kv("Écusson", PC.ecusson(ec) or "—"))
            act = ""
            if p.get("corrige"):
                c = p.get("correction") or {}
                corps += _kv("Valeur du loueur", C.esc(texte_valeur(p.get("valeur_loueur")) or "vide") + f" ({C.esc(p.get('annee_loueur') or 'année ?')})")
                corps += C.ligne("Motif", f'<div class="field text"><input class="pg-in t" data-agir="1" data-chemin="b/motif/{C.esc(ident)}" '
                                          f'value="{C.esc(c.get("motif") or "")}" placeholder="pourquoi cette correction ?"></div>')
                act += (f'<button type="button" class="btn sm" data-agir="1" data-chemin="b/revenir/{C.esc(ident)}" data-valeur="1">'
                        f'{C.icon("undo-2")}Revenir au prix du loueur</button>')
            av = p.get("a_verifier")
            act += (f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="b/verifier/{C.esc(ident)}" data-valeur="{"0" if av else "1"}">'
                    f'{C.icon("triangle-alert")}{"Plus à vérifier" if av else "Marquer à vérifier"}</button>')
            if hist:
                corps += '<div class="pg-sous">Historique de la case</div>' + "".join(
                    _kv(x.get("date") or "", C.esc(f"{texte_valeur(x.get('avant')) if not isinstance(x.get('avant'), str) else x.get('avant')} → "
                                                   f"{texte_valeur(x.get('apres')) if not isinstance(x.get('apres'), str) else x.get('apres')}")) for x in hist)
            out += (f'<div class="pg-bloc sel"><div class="pg-bloc-h">{C.esc(lib)}<small>{C.esc(nom)}</small></div>{corps}'
                    f'<div class="pg-fl-a">{act}</div></div>')
    sorties = EC.exporter([EC.ligne_export('data-telecharger="base_json"', "hard-drive", "Base des prix (JSON)",
                                           "la sauvegarde complète de la base de cet appareil")])
    return f'<div class="pbody"><div class="pg-res">{out}<div class="pg-sorties">{sorties}</div></div></div>'


def page(etat: Dict, base: Dict, auj: dt.date):
    dossiers = {l["nom"]: D.dossier(base, l["nom"], auj) for l in B.loueurs(base)}
    if not etat["vb"].get("loueur") and etat["vb"].get("tableau") not in ("reglages", "historique") and B.loueurs(base):
        etat["vb"]["loueur"] = B.loueurs(base)[0]["nom"]
        cats = B.catalogues_du_loueur(base, etat["vb"]["loueur"])
        etat["vb"]["tableau"] = cats[0]["nom"] if cats else "postes"
    titre, html = centre(etat, base, auj)
    vues = [{"cle": "base", "titre": titre, "court": "Tableau", "echelle": False, "svg": f'<div class="pg-detail">{html}</div>'}]
    return gauche(etat, base, auj, dossiers), vues, droite(etat, base, auj, dossiers), ""
