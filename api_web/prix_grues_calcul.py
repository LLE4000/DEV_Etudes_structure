# -*- coding: utf-8 -*-
"""La page « CALCUL » du module « Prix des grues » (cahier § 2) — trois colonnes :

- à GAUCHE, les entrées : la grue (catalogue, modèle avec son écusson et le parc du loueur, hauteur sous crochet,
  période, heures du grutier), puis seulement les rubriques utiles au catalogue choisi (châssis, conduite, fondation,
  électricité, télescopage, opérateur, services du loueur) ;
- au CENTRE, le détail du prix : la barre de la période, le tableau groupé (poste, choix Auto/Oui/Non, quantité, unité,
  prix, quantité facturée, total, source et écusson), où l'on agit directement — changer un choix, forcer un prix (la case
  passe en orange, sa justification dessous), ajouter une ligne libre ; les postes non retenus repliés en bas ;
- à DROITE, le résultat : le total HTVA, « INCOMPLET » s'il manque des prix, l'écusson du calcul et sa raison, matériel
  et main-d'œuvre, les prix manquants et les points d'attention.

Chaque prix ouvre sa fenêtre : sa source, son année, sa validité, la valeur du loueur sous notre correction,
« Corriger dans la base » (ou, pour la fondation, « Voir les chantiers »).

Rien n'est calculé ici : ``grues.prix.calcul`` rend les lignes, les totaux, les manquants et les points d'attention."""
from typing import Dict, List, Optional, Tuple

from grues.prix import base as B
from grues.prix import calcul as K
from grues.prix import ecussons as E
from grues.prix import fondations as F
from grues.prix import valeurs as V

from . import commun as C
from . import prix_grues_commun as PC

CH = "g"     # le chemin des saisies de la grue ouverte (api_web/prix_grues.modifier)
CHOIX_TXT = (("auto", "Auto"), ("oui", "Oui"), ("non", "Non"))
CHASSIS_TXT = (("standard", "Standard"), ("pied", "Pied d'ancrage"), ("roulant", "Roulant"), ("hors_standard", "Hors standard"))
CONDUITE_TXT = {"T": (("auto", "Auto"), ("cabine", "Cabine"), ("telecommande", "Télécom."), ("telecommande_externe", "Télécom. ext.")),
                "R": (("auto", "Auto"), ("cabine", "Cabine"), ("telecommande", "Télécom."))}
PERIODE_TXT = (("dates", "Dates"), ("mois", "Mois"), ("jours", "Jours"))
FAMILLE_MA_EXCLUS = set(K.HORAIRES_MA) | set(K.PERSONNEL_MA) | {"SUP_HS", "DEPL_GRUTIER", "BALLAST_SAM", "BALLAST_DIM", "CONTRIB_ENV",
                                                                K.POSTE_FONDATION}


def _ch(*k) -> List[str]:
    return [CH, *[str(x) for x in k]]


def _attr(*k) -> str:
    return C._attr_chemin(_ch(*k))


def _court(modele: str) -> str:
    """« Liebherr 125 EC-B 6 Litronic (> 39m HSC) » → « 125 EC-B 6 Litronic (> 39m HSC) » (le constructeur ôté)."""
    m = str(modele or "")
    for p in ("Liebherr ", "Wolff ", "Tadano ", "Hitachi "):
        if m.startswith(p):
            return m[len(p):]
    return m


# ------------------------------------------------------------------ à gauche : les entrées
def _ecusson_grue(base: Dict, g: Dict, aujourdhui) -> str:
    lg = B.logique(base, g.get("cas")) or ""
    ecs = []
    for c in B.champs_prix(base, lg):
        p = B.prix_grue(g, c["cle"])
        if V.nature(p.get("valeur")) not in (V.INCONNU, V.NON_DISPONIBLE):
            ecs.append(E.pour_ensemble(E.ecusson(p, aujourdhui), p, aujourdhui))
    return E.agreger(ecs, vide=E.MANQUANT) if ecs else E.MANQUANT


def options_grues(base: Dict, cas: str, aujourdhui, actuelle: Optional[str]) -> List[Tuple[str, str]]:
    out = []
    for g in B.grues_du(base, cas, retirees=False):
        e = E.LIBELLES.get(_ecusson_grue(base, g, aujourdhui), "")
        out.append((g["id"], f"{g.get('modele')} · {e}"))
    if actuelle and not any(k == actuelle for k, _ in out):
        g = B.grue(base, actuelle)
        if g:
            out.append((actuelle, f"{g.get('modele')} · retirée"))
    return out


def _parc(g: Dict, nom_loueur: str) -> str:
    n, d = g.get("nb_grues_chez_loueur"), g.get("nb_disponibles")
    if not V.est_nombre(n) and not V.est_nombre(d):
        return ""
    t = f"Parc {nom_loueur} : {PC.nombre(n, 0) if V.est_nombre(n) else '—'}"
    if V.est_nombre(d):
        t += f" · disponibles : {PC.nombre(d, 0)}"
    return f'<div class="pg-info">{C.esc(t)}</div>'


def rubriques(base: Dict, s: Dict, r: Dict, aujourdhui) -> str:
    """Les rubriques de la colonne de gauche (une seule ouverte à la fois : l'établi s'en charge)."""
    cats = B.catalogues(base)
    cas = s.get("cas")
    lg = B.logique(base, cas) if cas else None
    g = B.grue(base, s.get("grue")) if s.get("grue") else None
    out = []
    # 1. la grue et la période
    corps = C.ligne("Loueur – type", C.choix(_ch("cas"), cas or "", [("", "Choisir…")] + [(c["nom"], c["nom"]) for c in cats]))
    if cas:
        opts = options_grues(base, cas, aujourdhui, s.get("grue"))
        corps += C.ligne("Grue", C.choix(_ch("grue"), s.get("grue") or "", [("", "Choisir…" if opts else "Catalogue vide")] + opts))
        if g:
            corps += _parc(g, r.get("loueur") or "")
    if lg == "T":
        corps += C.ligne("HSC", C.num(_ch("hsc_m"), s.get("hsc_m"), "m", dec=1), aide="Hauteur sous crochet : l'alerte de variante, la demande au chantier")
    p = s.get("periode") or {}
    modes = PERIODE_TXT if lg != "MA" else (("jours", "Jours"), ("dates", "Dates"))
    mode = p.get("mode") or ("jours" if lg == "MA" else "dates")
    corps += C.ligne("Période", C.segments(_ch("periode", "mode"), mode, modes), full=True)
    if mode == "dates":
        corps += C.ligne2(("Du", C.texte(_ch("periode", "debut"), p.get("debut") or "", "AAAA-MM-JJ"), ""),
                          ("au", C.texte(_ch("periode", "fin"), p.get("fin") or "", "AAAA-MM-JJ"), ""))
    elif mode == "mois":
        corps += C.ligne("Durée", C.num(_ch("periode", "mois"), p.get("mois"), "mois", dec=2))
    else:
        corps += C.ligne("Durée", C.num(_ch("periode", "jours"), p.get("jours"), "j", dec=1))
    corps += C.ligne("Grutier", C.num(_ch("heures_jour"), s.get("heures_jour"), "h/j", dec=2),
                     aide="Heures par jour : au-delà de 8 h, heures supplémentaires")
    resume = f"{_court(g.get('modele'))}" if g else "à choisir"
    out.append(C.section(1, "Grue", C.esc(resume), corps, True, "grue"))
    if not lg or not g:
        return "".join(out)
    n = 2
    if lg == "T":
        ch = s.get("chassis") or "standard"
        c2 = C.ligne("Type", C.segments(_ch("chassis"), ch, CHASSIS_TXT))
        if ch == "roulant":
            c2 += C.ligne2(("Voie", C.num(_ch("voie_m"), s.get("voie_m"), "m", dec=1), "Longueur de la voie de roulement"),
                           ("Butées", C.num(_ch("butees"), s.get("butees"), "", entier=True), ""))
        out.append(C.section(n, "Châssis", C.esc(dict(CHASSIS_TXT)[ch].lower()), c2, False, "chassis"))
        n += 1
    if lg in ("T", "R"):
        cd = s.get("conduite") or "auto"
        eff = next((ln for ln in r["lignes"] if ln["code"] in ("CAB", "TEL", "TELV") and ln["retenu"]), None)
        txt = {"cabine": "cabine", "telecommande": "télécommande", "telecommande_externe": "télécommande extérieure"}
        res = txt.get(cd) if cd != "auto" else ("auto · " + ({"CAB": "cabine", "TEL": "télécommande", "TELV": "télécom. ext."}[eff["code"]] if eff else "—"))
        out.append(C.section(n, "Conduite", C.esc(res), C.ligne("Conduite", C.segments(_ch("conduite"), cd, CONDUITE_TXT[lg])), False, "conduite"))
        n += 1
    fd = s.get("fondation") or {}
    t = F.type_fondation(fd.get("type")) or "Dalle sur sol"
    c3 = (C.ligne("Type", C.choix(_ch("fondation", "type"), t, [(x, x) for x in F.TYPES]))
          + C.ligne("Démolition", C.case(_ch("fondation", "demolition"), bool(fd.get("demolition")), "comprise")))
    out.append(C.section(n, "Fondation", C.esc(f"{t.lower()}, {'avec' if fd.get('demolition') else 'sans'} démol."), c3, False, "fondation"))
    n += 1
    if lg in ("T", "R"):
        el = s.get("electricite") or {}
        kwh = B.prix_kwh(base)
        ind = f"{PC.nombre(kwh['valeur'], 3)} ({kwh['annee']})" if kwh else "—"
        c4 = (C.ligne("Prix du kWh", C.num(_ch("electricite", "prix_kwh"), el.get("prix_kwh"), "€/kWh", dec=4, auto=f"réglages · {ind}"),
                      aide="Celui du chantier ; vide : le dernier prix des réglages")
              + C.ligne("Raccordement", C.case(_ch("electricite", "inf_32a"), bool(el.get("inf_32a")), "inférieur à 32 A")))
        res = (f"{PC.nombre(el['prix_kwh'], 4)} €/kWh" if V.est_nombre(el.get("prix_kwh")) else f"kWh {ind}")
        out.append(C.section(n, "Électricité", C.esc(res), c4, False, "electricite"))
        n += 1
    if lg == "T":
        tl = s.get("telescopage") or {}
        c5 = C.ligne2(("Nombre", C.num(_ch("telescopage", "nombre"), tl.get("nombre"), "", entier=True), ""),
                      ("Jours", C.num(_ch("telescopage", "jours"), tl.get("jours"), "j", dec=1), "Jours par télescopage"))
        nb_ = tl.get("nombre") or 0
        out.append(C.section(n, "Télescopage", C.esc(f"{PC.nombre(nb_, 0)} × {PC.nombre(tl.get('jours') or 0, 1)} j" if nb_ else "non"), c5, False, "telescopage"))
        n += 1
    if lg == "MS":
        dispo = V.montant(B.prix_grue(g, "location_jour_avec_operateur")["valeur"]) is not None
        c6 = C.ligne("Opérateur", C.case(_ch("operateur"), bool(s.get("operateur")), "avec opérateur" + ("" if dispo else " (pas de prix)"),
                                         desactive=not dispo and not s.get("operateur")))
        out.append(C.section(n, "Opérateur", "avec" if s.get("operateur") else "sans", c6, False, "operateur"))
        n += 1
    if lg == "MA":
        out.append(C.section(n, C.esc(r.get("loueur") or "Loueur"), _resume_ma(r), _rubrique_ma(base, s, r), False, "loueur"))
        n += 1
    return "".join(out)


def _q(s: Dict, code: str):
    return (s.get("quantites") or {}).get(code)


def _resume_ma(r: Dict) -> str:
    dp = next((ln for ln in r["lignes"] if ln["code"] == "DEPL"), None)
    return C.esc(f"déplacement {PC.nombre(dp['quantite'], 1)} h" if dp and dp["quantite"] else "déplacement à saisir")


def _rubrique_ma(base: Dict, s: Dict, r: Dict) -> str:
    ma = s.get("ma") or {}
    out = C.ligne("Déplacement", C.num(_ch("quantites", "DEPL"), _q(s, "DEPL"), "h", dec=2),
                  aide="Heures de trajet aller et retour et de montage sur place (conditions du loueur)")
    out += C.ligne("Ballast", C.case(_ch("ma", "ballast_sam"), bool(ma.get("ballast_sam")), "samedi ou nuit"))
    out += C.ligne("", C.case(_ch("ma", "ballast_dim"), bool(ma.get("ballast_dim")), "dimanche ou nuit le week-end"))
    lignes = {ln["code"]: ln for ln in r["lignes"]}
    hs = [(c, t) for c, t in K.HORAIRES_MA.items() if c in lignes]
    for i in range(0, len(hs), 2):
        out += C.ligne2(*[(C.esc(t.capitalize()), C.num(_ch("quantites", c), _q(s, c), "h", dec=1), "") for c, t in hs[i:i + 2]])
    hp = ma.get("heures_personnel") or {}
    for c, t in (("RIGGER", "Riggers"), ("SUPERVISEUR", "Chef de levage")):
        if c in lignes:
            out += C.ligne2((t, C.num(_ch("quantites", c), _q(s, c), "pers.", entier=True), ""),
                            ("h/j", C.num(_ch("ma", "heures_personnel", c), hp.get(c), "h", dec=1, auto=f"{PC.nombre(s.get('heures_jour') or 8)} h"), ""))
    services = [ln for ln in r["lignes"] if ln["groupe"] == "loueur" and ln["code"] not in FAMILLE_MA_EXCLUS and ln["code"] not in ("DEPL", "BAL")]
    if services:
        out += '<div class="pg-sous">Services</div>'
        for ln in services:
            u = "j·qté" if ln["unite"] == "€/J" else "qté"
            out += C.ligne(C.esc(ln["libelle"]), C.num(_ch("quantites", ln["code"]), _q(s, ln["code"]), u, dec=1))
    return out


# ------------------------------------------------------------------ au centre : le détail du prix
def barre_periode(r: Dict) -> str:
    p = r["periode"]
    if r["logique"] == "MA":
        txt = f"{PC.nombre(p.get('jours'), 1)} j de location"
    else:
        dates = ""
        if p.get("debut") and p.get("fin"):
            dates = f"{_jjmmaa(p['debut'])} → {_jjmmaa(p['fin'])} · "
        txt = (f"{dates}{PC.nombre(p.get('mois'), 1)} mois · {PC.nombre(p.get('jours'), 1)} j facturés · "
               f"{PC.nombre(p.get('jours_grutier'), 1)} j grutier" + (" (calendrier réel)" if p.get("grutier_reel") else ""))
    mois = p.get("mois") or 0
    largeur = max(4, min(100, mois / 12 * 100)) if r["logique"] != "MA" else max(4, min(100, (p.get("jours") or 0) / 20 * 100))
    return (f'<div class="pg-periode"><div class="pg-bar" aria-hidden="true"><i style="width:{largeur:.0f}%"></i></div>'
            f'<span>{C.esc(txt)}</span></div>')


def _jjmmaa(iso: str) -> str:
    a, m, j = iso.split("-")
    return f"{j}/{m}/{a[2:]}"


def _menu(chemin: str, valeur: str, options, titre: str) -> str:
    """Un menu court du tableau (le choix Auto/Oui/Non, l'unité d'une ligne libre) : il part dès qu'on choisit."""
    opts = "".join(f'<option value="{C.esc(k)}"{" selected" if k == valeur else ""}>{C.esc(t)}</option>' for k, t in options)
    return f'<select class="pg-sel" data-agir="1" data-chemin="{chemin}" aria-label="{C.esc(titre)}">{opts}</select>'


def _choix_txt(ln: Dict) -> List[Tuple[str, str]]:
    return [("auto", f"Auto · {'oui' if ln['auto'] else 'non'}"), ("oui", "Oui"), ("non", "Non")]


def _cellule_prix(ln: Dict) -> str:
    """Le prix : la case se tape pour FORCER le prix de ce calcul (vide : le prix de la base) ; « compris », le prix de la
    base et son absence se lisent dans le texte indicatif."""
    o = ln["origine"] or {}
    n = V.nature(o.get("valeur"))
    if n == V.COMPRIS:
        ind = "compris"
    elif n == V.NOMBRE:
        ind = PC.nombre(V.montant(o.get("valeur")), 3)
    elif n == V.NON_DISPONIBLE:
        ind = "non dispo."
    elif n == V.SUR_DEMANDE:
        ind = "sur demande"
    else:
        ind = "à saisir"
    if ln["code"].startswith("LIBRE"):
        v = ln["prix"]
        return (f'<input class="pg-in" data-agir="1" data-chemin="{_attr("libres", ln["libre"], "prix")}" inputmode="decimal" '
                f'value="{C.esc(PC.nombre(v, 3) if v is not None else "")}" placeholder="à saisir" aria-label="Prix de {C.esc(ln["libelle"])}">')
    v = PC.nombre(ln["force"]["prix"], 3) if ln["force"] else ""
    return (f'<input class="pg-in{" force" if ln["force"] else ""}" data-agir="1" data-chemin="{_attr("forces", ln["code"], "prix")}" '
            f'inputmode="decimal" value="{C.esc(v)}" placeholder="{C.esc(ind)}" title="Taper un prix pour le forcer dans ce calcul" '
            f'aria-label="Prix de {C.esc(ln["libelle"])}">')


def _cellule_qte(ln: Dict) -> str:
    q = ln["quantite"]
    if ln["code"].startswith("LIBRE"):
        return (f'<input class="pg-in q" data-agir="1" data-chemin="{_attr("libres", ln["libre"], "quantite")}" inputmode="decimal" '
                f'value="{C.esc(PC.nombre(q, 2))}" aria-label="Quantité">')
    if ln["quantite_editable"]:
        return (f'<input class="pg-in q" data-agir="1" data-chemin="{_attr("quantites", ln["code"])}" inputmode="decimal" '
                f'value="{C.esc(PC.nombre(q, 2))}" aria-label="Quantité de {C.esc(ln["libelle"])}"'
                f'{" title=" + chr(34) + C.esc(ln["quantite_unite"]) + chr(34) if ln["quantite_unite"] else ""}>')
    u = f' <small>{C.esc(ln["quantite_unite"])}</small>' if ln["quantite_unite"] else ""
    return f'{C.esc(PC.nombre(q, 2))}{u}'


def _facturee(ln: Dict) -> str:
    f = ln["facturee"]
    if ln["unite"] in ("%", "coef"):
        return C.esc(PC.euros(f, 2))
    if ln["facturee_unite"] == "" and f == 1:
        return ""
    return C.esc(f"{PC.nombre(f, 2)} {ln['facturee_unite']}".strip())


def _source(ln: Dict) -> str:
    if ln["force"]:
        mot, ec = "forcé", None
    elif ln["manquant"]:
        mot, ec = ln["manquant_raison"], E.MANQUANT
    else:
        o = ln["origine"] or {}
        if ln["code"] == "FOND":
            mot = (ln["fondation"] or {}).get("source") or ""
        else:
            mot = "corrigé" if o.get("corrige") else (str(o.get("annee")) if o.get("annee") else "année ?")
        ec = ln["ecusson"]
    pt = PC.point(ec) if ec else '<span class="pg-force-mk">F</span>'
    return (f'<button type="button" class="pg-src" data-calcul="l-{C.esc(ln["code"])}" title="Source, année, écusson">{pt}'
            f'<span>{C.esc(mot)}</span></button>')


def _ligne_html(ln: Dict) -> str:
    cls = ["pg-l"]
    if ln["manquant"]:
        cls.append("manq")
    if ln["force"]:
        cls.append("force")
    if not ln["retenu"]:
        cls.append("non")
    det = f'<small>{C.esc(ln["detail"])}</small>' if ln.get("detail") else ""
    if ln["code"].startswith("LIBRE"):
        nom = (f'<input class="pg-in t" data-agir="1" data-chemin="{_attr("libres", ln["libre"], "libelle")}" value="{C.esc(ln["libelle"])}" '
               f'aria-label="Libellé de la ligne libre">')
        choix = (f'<button type="button" class="pg-x" data-agir="1" data-chemin="{_attr("libres", ln["libre"], "-")}" data-valeur="1" '
                 f'title="Retirer cette ligne">{C.icon("x")}</button>')
        unite = _menu(_attr("libres", ln["libre"], "unite"), ln["unite"], [(u, u) for u in ("FF", "€/J", "€/S", "€/M")], "Unité")
    else:
        nom = f'<span class="nm">{C.esc(ln["libelle"])}</span>'
        choix = _menu(_attr("choix", ln["code"]), ln["choix"], _choix_txt(ln), f"Choix pour {ln['libelle']}")
        unite = C.esc(ln["unite"])
    total = PC.euros(ln["total"], 2) if ln["retenu"] and ln["total"] is not None else ("manquant" if ln["manquant"] else (PC.euros(ln["montant"], 2) if ln["montant"] else ""))
    out = (f'<tr class="{" ".join(cls)}" id="pg-l-{C.esc(ln["code"])}"><td class="po">{nom}{det}</td><td class="ch">{choix}</td>'
           f'<td class="n qt">{_cellule_qte(ln)}</td><td class="n px">{_cellule_prix(ln)}<span class="un">{unite}</span></td>'
           f'<td class="n fa">{_facturee(ln)}</td><td class="n to">{C.esc(total)}</td><td class="so">{_source(ln)}</td></tr>')
    if ln["force"] and ln["retenu"]:
        j = ln["force"]["justification"]
        out += (f'<tr class="pg-just{"" if j else " manque"}"><td colspan="7"><label>Justification'
                f'<input class="pg-in t" data-agir="1" data-chemin="{_attr("forces", ln["code"], "justification")}" value="{C.esc(j)}" '
                f'placeholder="obligatoire : pourquoi ce prix ?"></label>'
                f'<button type="button" class="btn sm ghost" data-agir="1" data-chemin="{_attr("forces", ln["code"], "-")}" data-valeur="1">'
                f'{C.icon("undo-2")}Prix de la base</button></td></tr>')
    return out


ENTETE = ('<thead><tr><th>Poste</th><th>Choix</th><th class="n">Qté</th><th class="n">Prix · unité</th><th class="n">Facturé</th>'
          '<th class="n">Total</th><th>Source</th></tr></thead>')


def detail(r: Dict) -> str:
    """Le tableau du détail du prix, groupé, les postes non retenus repliés en bas."""
    lignes = {ln["code"]: ln for ln in r["lignes"]}
    corps = ""
    for gr in r["groupes"]:
        ret = [lignes[c] for c in gr["lignes"] if lignes[c]["retenu"]]
        if not ret and gr["cle"] != "libres":
            continue
        corps += (f'<tbody class="pg-grp"><tr class="pg-gh"><th colspan="5">{C.esc(gr["libelle"])}</th>'
                  f'<th class="n">{C.esc(PC.euros(gr["total"]))}</th><th></th></tr>{"".join(_ligne_html(ln) for ln in ret)}</tbody>')
    if not any(g["cle"] == "libres" for g in r["groupes"]):
        corps += ""
    non = [ln for ln in r["lignes"] if not ln["retenu"]]
    ajout = (f'<button type="button" class="et-add" data-agir="1" data-chemin="{_attr("libres", "+")}" data-valeur="1">'
             f'{C.icon("plus")}Ligne libre</button>')
    plie = ""
    if non:
        plie = (f'<details class="pg-non" data-plan-volet="pg-non"><summary>Postes non retenus ({len(non)})</summary>'
                f'<table class="pg-tab">{ENTETE}<tbody>{"".join(_ligne_html(ln) for ln in non)}</tbody></table></details>')
    return (f'{barre_periode(r)}<div class="pg-tab-w"><table class="pg-tab">{ENTETE}{corps}</table></div>'
            f'<div class="pg-pied">{ajout}</div>{plie}')


def vue(r: Dict, s: Dict) -> Dict:
    if not r["pret"]:
        corps = (f'<div class="pg-vide"><div class="pg-vide-c">{C.icon("arrow-left")}<b>{C.esc(r["ecusson_raison"].capitalize())} à gauche</b>'
                 f'<small>Le détail du prix, poste par poste, s\'affiche ici.</small></div></div>')
    else:
        corps = detail(r)
    return {"cle": "detail", "titre": "Détail du prix", "court": "Détail", "echelle": False, "svg": f'<div class="pg-detail">{corps}</div>'}


# ------------------------------------------------------------------ les fenêtres d'un prix
def _ligne_kv(k: str, v: str) -> str:
    return f'<div class="pg-kv"><span>{C.esc(k)}</span><b>{v}</b></div>'


def fenetre_ligne(ln: Dict, r: Dict) -> str:
    o = ln["origine"] or {}
    n = V.nature(o.get("valeur"))
    val = {V.COMPRIS: "compris (0 €)", V.NON_DISPONIBLE: "non disponible pour ce modèle", V.SUR_DEMANDE: "sur demande au loueur",
           V.INCONNU: "inconnu", V.TEXTE: C.esc(str(o.get("valeur")))}.get(n) or C.esc(PC.prix_unitaire(V.montant(o.get("valeur"))))
    corps = _ligne_kv("Prix de la base", f"{val} <small>{C.esc(ln['unite'])}</small>")
    if o.get("corrige"):
        c = o.get("correction") or {}
        corps += _ligne_kv("Valeur du loueur", C.esc(PC.prix_unitaire(V.montant(o.get("valeur_loueur")))) + f" ({C.esc(o.get('annee_loueur') or 'année ?')})")
        corps += _ligne_kv("Notre correction", C.esc(f"{c.get('date') or ''} · {c.get('motif') or 'sans motif'}"))
    corps += _ligne_kv("Année", C.esc(o.get("annee") or "inconnue"))
    v = o.get("validite") or {}
    if v:
        corps += _ligne_kv("Validité", C.esc(f"{v.get('du') or '…'} → {v.get('au') or '…'}"))
    corps += _ligne_kv("Source", C.esc(o.get("source") or "—"))
    corps += _ligne_kv("Écusson", PC.ecusson(ln["ecusson"]) or "—")
    if ln["force"]:
        corps += _ligne_kv("Prix forcé", C.esc(PC.prix_unitaire(ln["force"]["prix"])) + f" · {C.esc(ln['force']['justification'] or 'sans justification')}")
    corps += _ligne_kv("Choix", C.esc(f"{dict(CHOIX_TXT)[ln['choix']]} · Auto = {'oui' if ln['auto'] else 'non'} ({ln['auto_raison']})"))
    if ln.get("detail"):
        corps += _ligne_kv("Calcul", C.esc(ln["detail"]))
    actions = ""
    ref = o.get("ref") or ()
    if ln["code"] == "FOND":
        k = ln["fondation"] or {}
        if k.get("chantiers"):
            corps += '<div class="pg-sous">Chantiers</div>' + "".join(
                _ligne_kv(f"{x['chantier'] or x['id']} · {x['annee'] or 'année ?'}", f"{C.esc(PC.euros(x['cout_actualise']))} · {C.esc(F.LIBELLES_STATUT[x['statut']])}")
                for x in k["chantiers"])
        actions = (f'<button type="button" class="btn sm" data-agir="1" data-chemin="aller/fondations" '
                   f'data-valeur="{C.esc(F.cle_case(k.get("modele") or "", k.get("type") or "", bool(k.get("demolition"))))}">'
                   f'{C.icon("layers")}Voir les chantiers</button>')
    elif ref and ref[0] in ("grue", "poste"):
        actions = (f'<button type="button" class="btn sm" data-agir="1" data-chemin="aller/base" data-valeur="{C.esc("|".join(str(x) for x in ref))}">'
                   f'{C.icon("pencil-line")}Corriger dans la base</button>')
    return (f'<div class="flyout calcul pg-fl" id="ck-l-{C.esc(ln["code"])}"><div class="fl-h"><div><b>{C.esc(ln["libelle"])}</b>'
            f'<span>{C.esc(ln["code"])} · {C.esc(r.get("modele") or "")}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{C.icon("x")}</button></div>'
            f'<div class="ck-body pg-fl-b">{corps}<div class="pg-fl-a">{actions}</div></div></div>')


def volets(r: Dict) -> str:
    return "".join(fenetre_ligne(ln, r) for ln in r["lignes"])


# ------------------------------------------------------------------ à droite : le résultat
def _liste(titre: str, items: List[str], cls: str) -> str:
    if not items:
        return f'<div class="pg-liste {cls} zero"><b>{C.esc(titre)} (0)</b></div>'
    return f'<div class="pg-liste {cls}"><b>{C.esc(titre)} ({len(items)})</b><ul>{"".join(items)}</ul></div>'


def colonne(r: Dict, s: Dict, exporter: str = "") -> str:
    if not r["pret"]:
        return (f'<div class="pbody"><div class="pg-res"><div class="verdict pg-verdict"><div class="pg-total nul">— € <small>HTVA</small></div>'
                f'<p class="pg-info">{C.esc(r["ecusson_raison"].capitalize())}.</p></div></div></div>')
    tot = r["total_arrondi"]
    inc = ""
    if r["manquants"]:
        n = len(r["manquants"])
        inc = f'<div class="pg-incomplet">{C.icon("triangle-alert")}INCOMPLET – {n} prix manquant{"s" if n > 1 else ""}</div>'
    m, mo = r["materiel"], r["main_oeuvre"]
    part = 100 * m / (m + mo) if (m + mo) else 0
    rep = (f'<div class="pg-rep"><div class="pg-rep-b" aria-hidden="true"><i style="width:{part:.0f}%"></i></div>'
           f'<div class="pg-kv"><span><i class="pg-sw m"></i>Matériel</span><b>{C.esc(PC.euros(m))}</b></div>'
           f'<div class="pg-kv"><span><i class="pg-sw o"></i>Main-d\'œuvre</span><b>{C.esc(PC.euros(mo))}</b></div></div>')
    manq = [f'<li><a href="#pg-l-{C.esc(x["code"])}" data-vers-onglet="dessin">{C.esc(x["libelle"])}</a> <small>{C.esc(x["raison"])}</small> '
            f'<span class="pg-act">→ Saisir</span></li>' for x in r["manquants"]]
    att = [f'<li class="{C.esc(a["niveau"])}">{C.esc(a["texte"])}'
           + (f' <button type="button" class="btn sm ghost" data-agir="1" data-chemin="{_attr("grue")}" data-valeur="{C.esc(a["autre"])}">Prendre la variante</button>'
              if a.get("autre") else "") + '</li>' for a in r["attention"]]
    hyp = "".join(_ligne_kv(k, C.esc(v)) for k, v in r["hypotheses"])
    return (f'<div class="pbody"><div class="pg-res">'
            f'<div class="verdict pg-verdict"><div class="pg-total{" inc" if r["manquants"] else ""}">{C.esc(PC.euros(tot))} <small>HTVA</small></div>{inc}'
            f'<div class="pg-ec-l">{PC.ecusson(r["ecusson"], r["ecusson_raison"])}<span>{C.esc(r["ecusson_raison"])}</span></div></div>'
            f'{rep}{_liste("Prix manquants", manq, "manq")}{_liste("Points d’attention", att, "att")}'
            f'<details class="pg-hyp" data-plan-volet="pg-hyp"><summary>Hypothèses</summary>{hyp}</details>'
            f'<div class="pg-sorties">{exporter}</div></div></div>')
