# -*- coding: utf-8 -*-
"""La rubrique « HOURDIS » de la Bibliothèque Béton armé : les catalogues de hourdis alvéolés précontraints des
fabricants (``beton/donnees/hourdis``, lus par ``beton/hourdis/catalogue.py`` — Ergon 2018 aujourd'hui ; une autre
marque, c'est un fichier et une ligne de ``CATALOGUES``). Deux vues qui partagent les mêmes paramètres :

- « Profil » (ouverte par défaut) : la COUPE COTÉE du profil (le tracé relevé dans le guide du fabricant, ses cotes ;
  ``api_web/dessin_biblio_hourdis.py``), ses caractéristiques et sa mise en œuvre, puis le TABLEAU de ses variantes —
  M_Rd, V_Rd,c et M_Rd avec couche collaborante, par élément, la case choisie allumée ; avec M_Ed et V_Ed
  (facultatifs), les cases qui suffisent en vert et la première variante qui suffit (la moins armée) marquée ★ ;
- « Gamme » : les profils côte à côte (hauteur, poids, aire, inertie, modules, bornes de M_Rd et V_Rd,c) ; avec M_Ed et
  V_Ed, la première variante qui suffit de chaque profil, le plus léger qui suffit marqué ★.

« Par élément / par mètre » (demande de l'ingénieur du 11/11/2026 : « il faut diviser par 1,2 pour avoir des kN par
mètre, on utilise souvent ça ») : les valeurs du fabricant sont PAR ÉLÉMENT (module de 1,20 m) ; « par mètre » les
divise par la largeur du module, arrondies par défaut (``catalogue.par_metre``), et lit M_Ed et V_Ed par mètre aussi.

Les valeurs sont celles du catalogue (R3 : rien n'est recopié ici) ; « ce qui suffit » est jugé par
``catalogue.verifier`` et ``catalogue.gamme`` — l'écran choisit, compose et formate. Un paramètre illisible reprend
son défaut et ``message`` le dit ; une variante d'un autre profil (le profil vient de changer) reprend en silence la
première du profil choisi.
"""
from typing import Dict, List, Optional

from api_web import biblio_beton as B
from api_web import commun as C
from api_web import dessin_biblio_hourdis as D
from beton.hourdis import catalogue as K

NBSP = C.NBSP
VUES = (("profil", "Profil"), ("gamme", "Gamme"))
BASES = (("element", "par élément"), ("metre", "par mètre"))
PROFIL_DEFAUT = "SP 200"
# les trois valeurs d'une variante : (clé, symbole, unité, nom)
GRANDEURS = (("M_Rd", "M_Rd", "kN·m", "moment résistant de calcul par élément"),
             ("V_Rd", "V_Rd,c", "kN", "effort tranchant résistant de calcul par élément"),
             ("M_Rd_chape", "M_Rd", "kN·m", "moment résistant de calcul avec couche collaborante"))
# les caractéristiques de la vue « Gamme » : (clé, symbole, unité affichée, nom, facteur de l'unité du catalogue, décimales)
CARACS = (("h", "h", "mm", "hauteur", 1.0, 0), ("poids_kN_m2", "g", "kN/m²", "poids propre avec joints", 1.0, 2),
          ("A", "A", "10³ mm²", "aire de la section", 1e-3, 0), ("Ix", "I_x", "10⁶ mm⁴", "moment d'inertie", 1e-6, 0),
          ("W_sup", "W_sup", "10⁶ mm³", "module de flexion, fibre supérieure", 1e-6, 2),
          ("W_inf", "W_inf", "10⁶ mm³", "module de flexion, fibre inférieure", 1e-6, 2))


def _torons(cat: Dict, nom: str) -> str:
    """« 4 Ø 12,5 + 2 Ø 9,3 »."""
    return " + ".join(f"{n}{NBSP}Ø{NBSP}{C.nb(d)}" for n, d in K.torons(cat, nom))


def _n(x, nd: int = 0) -> str:
    return C.fr(x, nd)


# ================================================================================================ les paramètres
def _val(x, par_metre: bool) -> str:
    """Une résistance telle que le tableau l'écrit : un entier par élément (la valeur publiée), un dixième par mètre."""
    return C.fr(x, K.DECIMALES_PAR_METRE) if par_metre else _n(x)


def lire(params: Optional[Dict]):
    L = B.Lecture(params)
    cats = K.catalogues()
    cat = K.catalogue(L.choix("fabricant", tuple(k for k, _ in cats), cats[0][0]))
    noms = [p["nom"] for p in cat["profils"]]
    L.choix("vue", tuple(k for k, _ in VUES), "profil")
    nom_p = L.choix("profil", tuple(noms), PROFIL_DEFAUT if PROFIL_DEFAUT in noms else noms[0])
    p = K.profil(cat, nom_p)
    variantes = [v["nom"] for v in p["variantes"]]
    brut = L.brut.get("variante")
    if brut in variantes:
        var = brut
    else:
        var = variantes[0]
        autres = {v["nom"] for q in cat["profils"] for v in q["variantes"]}
        if brut not in (None, "") and brut not in autres:      # pas une variante du catalogue : un refus doux, dit
            L._dit("variante", f"ignoré : « {str(brut)[:40]} » n'est pas proposé")
    L.out["variante"] = var
    grandeur = L.choix("grandeur", tuple(g[0] for g in GRANDEURS), "M_Rd")
    if grandeur == "M_Rd_chape" and K.variante(cat, var)[1].get("M_Rd_chape") is None:
        grandeur = L.out["grandeur"] = "M_Rd"                  # une valeur que le fabricant ne publie pas ne s'allume pas
    L.choix("carac", tuple(c[0] for c in CARACS), "Ix")
    L.choix("base", tuple(k for k, _ in BASES), "element")
    L.nombre("M_Ed", None, 0, 100000, vide=None)
    L.nombre("V_Ed", None, 0, 100000, vide=None)
    L.case("chape", False)
    return L, cat, p


# ================================================================================================ la barre
def _barre(L, cat: Dict, p: Dict) -> str:
    o = L.out
    onglets = B._champ("vue", "", B._segments("vue", o["vue"], VUES)).replace('class="bib-p bib-p-case"', 'class="bib-p bib-onglets"')
    choix = (B._champ("fabricant", "Fabricant", B._select("fabricant", o["fabricant"], [(k, K.titre(c)) for k, c in K.catalogues()]),
                      "Catalogues transcrits dans l'application ; d'autres marques s'ajouteront")
             + B._champ("profil", "Profil", B._select("profil", o["profil"], [(q["nom"], f"{q['nom']} · h{NBSP}{q['h']}") for q in cat["profils"]])))
    pm = o["base"] == "metre"
    b_txt = C.nb(K.largeur_element(cat))
    par = "PAR MÈTRE de largeur de plancher" if pm else f"PAR ÉLÉMENT ({b_txt} m)"
    efforts = (B._champ("base", "Valeurs", B._segments("base", o["base"], BASES),
                        f"les tableaux du fabricant sont par élément (module de {b_txt} m, joint compris) ; « par mètre » : "
                        f"÷ {b_txt}, arrondi par défaut au dixième — M_Ed et V_Ed se lisent alors par mètre aussi")
               + B._champ("M_Ed", "M<sub>Ed</sub>", B._num("M_Ed", o["M_Ed"], "kN·m/m" if pm else "kN·m", "—"),
                          f"facultatif · moment de calcul {par}, poids propre du hourdis compris (ELU) : les variantes qui "
                          "suffisent s'allument en vert, la première est marquée ★")
               + B._champ("V_Ed", "V<sub>Ed</sub>", B._num("V_Ed", o["V_Ed"], "kN/m" if pm else "kN", "—"),
                          f"facultatif · effort tranchant de calcul {par.lower() if not pm else par}, à l'appui (ELU), poids propre compris")
               + B._champ("chape", "", B._case("chape", o["chape"], "Avec couche collaborante",
                                                f"M_Rd avec {cat['meta']['chape']['epaisseur_min']} mm de couche collaborante "
                                                f"{cat['meta']['chape']['classe']} (p. {cat['meta']['chape']['page']}) ; V_Rd,c reste "
                                                "la valeur nue : la valeur avec couche n'est pas publiée")))
    return B._barre(f'<div class="bib-onglets-l">{onglets}</div>' + B._groupe("Catalogue", choix) + B._groupe("Efforts", efforts))


# ================================================================================================ la vue « Profil »
def _caracteristiques(cat: Dict, p: Dict) -> str:
    m = cat["meta"]
    r = p["reservations"]
    lignes = [
        ("Dimensions", "", f"h{NBSP}{_n(p['h'])} · b{NBSP}{_n(p['b'])}{NBSP}mm (module {_n(m['largeur_nominale']['valeur'])})"),
        ("Poids propre", "", f"{C.nb(p['poids_kN_m'])}{NBSP}kN/m · {C.nb(p['poids_kN_m2'])}{NBSP}kN/m² (avec joints)"),
        ("Section", "", f"A{NBSP}{_n(p['A'] / 1e3)}·10³{NBSP}mm² · I<sub>x</sub>{NBSP}{_n(p['Ix'] / 1e6)}·10⁶{NBSP}mm⁴"),
        ("Modules", "", f"W<sub>sup</sub>{NBSP}{C.fr(p['W_sup'] / 1e6, 2)} · W<sub>inf</sub>{NBSP}{C.fr(p['W_inf'] / 1e6, 2)}·10⁶{NBSP}mm³"),
        ("Appui minimal", "a", f"{_n(p['appui']['beton_acier'])}{NBSP}mm béton ou acier · {_n(p['appui']['maconnerie'])}{NBSP}mm maçonnerie"),
        ("Transport · joints", "", f"{_n(p['transport'])}{NBSP}kg/m · {C.nb(p['joints'])}{NBSP}l/m²"
                                   + (f" (p.{NBSP}20 : {C.nb(p['joints_p20'])})" if p.get("joints_p20") is not None else "")),
        ("Hourdis rétrécis", "", " / ".join(_n(x) for x in p["retrecis"]) + f"{NBSP}mm"),
        ("Réservations max.", "", f"angles {_n(r['angles'][0])}×{_n(r['angles'][1])} · abouts {_n(r['abouts'][0])}×{_n(r['abouts'][1])}"
                                  f" · rives {_n(r['faces_laterales'][0])}×{_n(r['faces_laterales'][1])} · Ø{NBSP}{_n(r['circulaire'])}"
                                  f" · centrales {_n(r['rectangulaire'][0])}×{_n(r['rectangulaire'][1])}{NBSP}mm"),
    ]
    if p.get("ancrage_alveole") is not None:
        lignes.append(("Cheville dans l'alvéole", "", f"{C.nb(p['ancrage_alveole'])}{NBSP}kN admissibles ({C.esc(m['ancrage_alveole']['court'])})"))
    corps = "".join(f'<div class="bib-hd-l"><dt>{t}</dt><dd>{(s + " ") if s else ""}{v}</dd></div>' for t, s, v in lignes)
    return f'<dl class="bib-hd-c">{corps}</dl>'


def _profil(L, cat: Dict, p: Dict, hote: str) -> Dict:
    o = L.out
    pm = o["base"] == "metre"
    um, uv, par = ("kN·m/m", "kN/m", "par mètre de largeur") if pm else ("kN·m", "kN", "par élément")
    ver = K.verifier(p, o["M_Ed"], o["V_Ed"], o["chape"], cat if pm else None)
    nom_v, grandeur = o["variante"], o["grandeur"]
    th = ('<tr><th class="nm">Variante<span class="u">torons</span></th>'
          + B._th("M_Rd", um, f"moment résistant de calcul {par} (ELU)")
          + B._th("V_Rd,c", uv, f"effort tranchant résistant de calcul {par} (ELU), sans couche collaborante")
          + B._th("M_Rd", f"{um} + couche",
                  f"moment résistant avec {cat['meta']['chape']['epaisseur_min']} mm de couche collaborante {cat['meta']['chape']['classe']}")
          + "</tr>")
    corps, lignes_extrait, case_extrait = "", [], None
    for i, li in enumerate(ver["lignes"]):
        on = li["nom"] == nom_v
        etoile = '<i class="mk">★</i>' if li["nom"] == ver["premiere"] else ""
        cellules = ""
        for j, (cle, *_r) in enumerate(GRANDEURS):
            val = li[cle]
            if val is None:
                cellules += '<td class="na" title="non publié">—</td>'
                continue
            sel = {"vue": "profil", "fabricant": o["fabricant"], "profil": p["nom"], "variante": li["nom"], "grandeur": cle}
            if pm:
                sel["base"] = "metre"
            juge = (li["suffit_V"] if cle == "V_Rd" else
                    li["suffit_M"] if (cle == "M_Rd_chape") == li["avec_chape"] else None)
            cls = " ".join(k for k, v in (("cur", on and cle == grandeur), ("ok", juge is True and not (on and cle == grandeur))) if v)
            cellules += B._td(_val(val, pm), sel, cls)
            if on and cle == grandeur:
                case_extrait = (i, j + 1)
        corps += (f'<tr{" class=" + chr(34) + "on" + chr(34) if on else ""}><th class="nm">{C.esc(li["nom"])}{etoile}'
                  f'<small class="bib-hd-to">{_torons(cat, li["nom"])}</small></th>{cellules}</tr>')
        lignes_extrait.append([li["nom"], _val(li["M_Rd"], pm), _val(li["V_Rd"], pm),
                               _val(li["M_Rd_chape"], pm) if li["M_Rd_chape"] is not None else "—"])
    tab = B._tableau(th, corps, "bib-hd-tab")
    figure = (f'<figure class="bib-hd-f"><div class="bib-hd-s">{D.coupe("bib-hd-coupe", p)}</div>'
              f'<figcaption>{C.esc(p["nom"])} · {C.esc(D.legende_alveoles(p))} · relevé du guide {C.esc(K.titre(cat))}, '
              f'p.{NBSP}{p["page"]} · cotes en mm</figcaption></figure>')
    tableau = f'<div class="bib-hd">{figure}{_caracteristiques(cat, p)}<div class="bib-hd-t">{tab}</div></div>'
    # la bande : la case choisie
    _, v = K.variante(cat, nom_v)
    g = next(x for x in GRANDEURS if x[0] == grandeur)
    li_v = next(li for li in ver["lignes"] if li["nom"] == nom_v)
    val, sym = li_v[grandeur], g[1]
    unite = (g[2] + "/m") if pm else g[2]
    nom = g[3].replace("par élément", par) if "par élément" in g[3] else f"{g[3]}, {par}"
    m = cat["meta"]
    b_txt = C.nb(K.largeur_element(cat))
    phrase = f"{C.esc(nom_v)} · {_torons(cat, nom_v)} · {C.esc(K.titre(cat))}, p.{NBSP}{p['page']}"
    if grandeur == "M_Rd_chape":
        phrase += f" · avec {m['chape']['epaisseur_min']}{NBSP}mm de couche collaborante {m['chape']['classe']}"
    # l'autre lecture de la même valeur : par élément ↔ par mètre (÷ la largeur du module, arrondi par défaut)
    if pm:
        phrase += f" · soit {_n(v[grandeur])}{NBSP}{g[2]} par élément de {b_txt}{NBSP}m"
    else:
        phrase += f" · soit {_val(K.par_metre(cat, v[grandeur]), True)}{NBSP}{g[2]}/m (÷{NBSP}{b_txt}{NBSP}m)"
    if ver["demande"]:
        effort = " · ".join(t for t in ((f"M<sub>Ed</sub>{NBSP}={NBSP}{C.nb(o['M_Ed'])}{NBSP}{um}" if o["M_Ed"] is not None else ""),
                                        (f"V<sub>Ed</sub>{NBSP}={NBSP}{C.nb(o['V_Ed'])}{NBSP}{uv}" if o["V_Ed"] is not None else "")) if t)
        phrase += (f" — pour {effort}" + (" (couche collaborante)" if o["chape"] else "") + " : "
                   + (f"<b>{C.esc(ver['premiere'])}</b> ★, la moins armée qui suffit" if ver["premiere"] else "<b>aucune variante de ce profil ne suffit</b>"))
    avert = []
    if ver["demande"] and ver["premiere"] is None:
        avert.append("Aucune variante de ce profil ne suffit : la vue « Gamme » montre les profils qui suffisent.")
    if grandeur == "M_Rd_chape" or o["chape"]:
        avert.append(f"Avec couche collaborante : épaisseur ≥ {m['chape']['epaisseur_min']}{NBSP}mm au point le plus haut du hourdis, "
                     f"béton {m['chape']['classe']}, {C.esc(m['chape']['treillis'])} (p.{NBSP}{m['chape']['page']}) ; V<sub>Rd,c</sub> avec "
                     "couche n'est pas publié — la valeur nue s'applique.")
    termes = _termes(cat, p, v)
    sources = _sources(cat, p)
    extrait = B._extrait(["Variante", f"M_Rd ({um})", f"V_Rd,c ({uv})", f"M_Rd avec {m['chape']['epaisseur_min']} mm de couche ({um})"],
                         lignes_extrait, case_extrait, f"{p['nom']} — {K.titre(cat)}, valeurs de calcul {par} (p. {p['page']})"
                         + (f" ÷ {b_txt} m, arrondies par défaut" if pm else ""), sources[0])
    texte = _val(val, pm)
    return {"titre": f"Hourdis {K.titre(cat)}", "barre": _barre(L, cat, p), "tableau": tableau,
            "bande": B._bande(texte, unite, sym, nom, phrase, hote, avert),
            "pourquoi": B._pourquoi(f"{C.esc(nom_v)} · {C.esc(K.titre(cat))}", f"guide technique du fabricant, p.{NBSP}{p['page']}",
                                    termes, sources, avert),
            "sources": B._sources(*sources),
            "fiche": B._fiche(f"Hourdis {nom_v} ({K.titre(cat)})", sym, val, unite, o, termes, sources, extrait, avert, texte=texte,
                              phrase=phrase),
            "params": o, "message": L.message()}


def _termes(cat: Dict, p: Dict, v: Dict) -> List[Dict]:
    m = cat["meta"]
    ch = m["chape"]
    out = [
        {"symbole": "M_Rd", "nom": "MOMENT RÉSISTANT DE CALCUL", "valeur": v["M_Rd"], "unite": "kN·m",
         "regle": f"par élément, ELU : M_Ed ≤ M_Rd (p. {p['page']})"},
        {"symbole": "V_Rd,c", "nom": "effort tranchant résistant de calcul", "valeur": v["V_Rd"], "unite": "kN",
         "regle": f"par élément, ELU : V_Ed ≤ V_Rd,c (p. {p['page']}) ; avec couche collaborante : non publié"},
    ]
    if v.get("M_Rd_chape") is not None:
        out.append({"symbole": "M_Rd", "nom": "avec couche collaborante", "valeur": v["M_Rd_chape"], "unite": "kN·m",
                    "regle": f"{ch['epaisseur_min']} mm de couche {ch['classe']} au point le plus haut (p. {ch['page']})"})
    b_txt = C.nb(K.largeur_element(cat))
    regle_m = (f"÷ {b_txt} m (largeur du module, joint compris : p. {m['largeur_nominale']['page']}), arrondi par défaut au dixième")
    out += [{"symbole": "m_Rd", "nom": "moment résistant par mètre de largeur", "valeur": K.par_metre(cat, v["M_Rd"]), "unite": "kN·m/m",
             "regle": "M_Rd " + regle_m},
            {"symbole": "v_Rd,c", "nom": "effort tranchant résistant par mètre de largeur", "valeur": K.par_metre(cat, v["V_Rd"]),
             "unite": "kN/m", "regle": "V_Rd,c " + regle_m}]
    if v.get("M_Rd_chape") is not None:
        out.append({"symbole": "m_Rd", "nom": "par mètre, avec couche collaborante", "valeur": K.par_metre(cat, v["M_Rd_chape"]),
                    "unite": "kN·m/m", "regle": "M_Rd avec couche " + regle_m})
    out += [
        {"symbole": "", "nom": "torons", "valeur": _torons(cat, v["nom"]).replace(NBSP, " "), "unite": "",
         "regle": m["notation"]["texte"]},
        {"symbole": "h", "nom": "hauteur", "valeur": p["h"], "unite": "mm", "regle": f"p. {m['pages']['caracteristiques']}"},
        {"symbole": "g", "nom": "poids propre avec joints", "valeur": p["poids_kN_m2"], "unite": "kN/m²",
         "regle": f"{C.nb(p['poids_kN_m'])} kN/m par élément (p. {m['pages']['caracteristiques']}) ; à compter dans M_Ed et V_Ed"},
        {"symbole": "I_x", "nom": "moment d'inertie", "valeur": p["Ix"] / 1e6, "unite": "10⁶ mm⁴", "regle": f"p. {m['pages']['caracteristiques']}"},
        {"symbole": "", "nom": "béton", "valeur": m["beton"]["classe"], "unite": "", "regle": m["beton"]["texte"]},
        {"symbole": "", "nom": "feu", "valeur": m["feu"]["court"], "unite": "", "regle": m["feu"]["texte"]},
        {"symbole": "", "nom": "conditions", "valeur": f"{m['classe_consequence']['court']} · {m['environnement']['court']}", "unite": "",
         "regle": f"{m['classe_consequence']['texte']} (p. {m['classe_consequence']['page']}) ; {m['environnement']['texte']}"},
    ]
    return out


def _sources(cat: Dict, p: Dict) -> List[str]:
    m = cat["meta"]
    pg = m["pages"]
    return [f"{m['document']} — p. {p['page']} (tableau et coupe), p. {pg['caracteristiques']} (caractéristiques), p. {pg['appui']} "
            f"(appuis), p. {pg['retrecis']} (rétrécis), p. {pg['reservations']} (réservations), p. {pg['ancrage_alveole']} (chevilles)",
            f"Transcription : {m['transcription']}"]


# ================================================================================================ la vue « Gamme »
def _gamme(L, cat: Dict, p: Dict, hote: str) -> Dict:
    o = L.out
    pm = o["base"] == "metre"
    um, uv = ("kN·m/m", "kN/m") if pm else ("kN·m", "kN")
    g = K.gamme(cat, o["M_Ed"], o["V_Ed"], o["chape"], pm)
    carac = o["carac"]
    th = ('<tr><th class="nm">Profil</th>' + "".join(B._th(sym, unite, nom) for _, sym, unite, nom, _, _ in CARACS)
          + B._th("M_Rd", um, "de la variante la moins armée à la plus armée" + (" (avec couche collaborante)" if o["chape"] else "")
                  + (", par mètre de largeur" if pm else ", par élément"))
          + B._th("V_Rd,c", uv, "de la variante la moins armée à la plus armée" + (", par mètre de largeur" if pm else ", par élément"))
          + ('<th class="tx" title="la première variante qui suffit (la moins armée)">Suffit</th>' if g["demande"] else "") + "</tr>")
    corps, lignes_extrait, case_extrait = "", [], None
    for i, (q, li) in enumerate(zip(cat["profils"], g["lignes"])):
        on = q["nom"] == p["nom"]
        cellules, ext = "", [q["nom"]]
        for j, (cle, _s, _u, _n2, f, nd) in enumerate(CARACS):
            sel = {"vue": "gamme", "fabricant": o["fabricant"], "profil": q["nom"], "carac": cle}
            if pm:
                sel["base"] = "metre"
            cur = on and cle == carac
            cellules += B._td(C.fr(q[cle] * f, nd), sel, "cur" if cur else "")
            ext.append(C.fr(q[cle] * f, nd))
            if cur:
                case_extrait = (i, j + 1)
        cellules += (f'<td>{_val(li["M_min"], pm)}–{_val(li["M_max"], pm)}</td><td>{_val(li["V_min"], pm)}–{_val(li["V_max"], pm)}</td>')
        if g["demande"]:
            mk = '<i class="mk">★</i>' if q["nom"] == g["leger"] else ""
            cellules += (f'<td class="tx{" ok" if li["premiere"] else ""}">{C.esc(li["premiere"]) if li["premiere"] else "—"}{mk}</td>')
        corps += f'<tr{" class=" + chr(34) + "on" + chr(34) if on else ""}><th class="nm">{C.esc(q["nom"])}</th>{cellules}</tr>'
        lignes_extrait.append(ext)
    tableau = f'<div class="bib-hd bib-hd-g"><div class="bib-hd-t">{B._tableau(th, corps, "bib-hd-tab")}</div></div>'
    cle, sym, unite, nom, f, nd = next(c for c in CARACS if c[0] == carac)
    val = p[cle] * f
    phrase = f"{C.esc(p['nom'])} · {C.esc(K.titre(cat))}, p.{NBSP}{cat['meta']['pages']['caracteristiques']}"
    if g["demande"]:
        phrase += (" — " + (f"le plus léger qui suffit : <b>{C.esc(g['leger'])}</b> ★ ({C.nb(K.profil(cat, g['leger'])['poids_kN_m2'])}{NBSP}kN/m²)"
                            if g["leger"] else "<b>aucun profil ne suffit</b>"))
    avert = []
    if g["demande"]:
        avert.append("M<sub>Ed</sub> et V<sub>Ed</sub> comptent le poids propre du hourdis : un profil plus lourd les augmente — "
                     "refaire le calcul avec le poids du profil retenu.")
    termes = [{"symbole": sym, "nom": nom.upper(), "valeur": val, "unite": unite, "regle": f"p. {cat['meta']['pages']['caracteristiques']}"}]
    termes += [{"symbole": s, "nom": n, "valeur": p[k] * fa, "unite": u, "regle": f"p. {cat['meta']['pages']['caracteristiques']}"}
               for k, s, u, n, fa, _ in CARACS if k != carac]
    sources = _sources(cat, p)
    extrait = B._extrait(["Profil"] + [f"{s} ({u})" for _, s, u, _, _, _ in CARACS], lignes_extrait, case_extrait,
                         f"Gamme {K.titre(cat)} — caractéristiques (p. {cat['meta']['pages']['caracteristiques']})", sources[0])
    texte = C.fr(val, nd)
    return {"titre": f"Hourdis {K.titre(cat)}", "barre": _barre(L, cat, p), "tableau": tableau,
            "bande": B._bande(texte, unite, sym, nom, phrase, hote, avert),
            "pourquoi": B._pourquoi(f"{C.esc(p['nom'])} · {C.esc(K.titre(cat))}", "caractéristiques du fabricant", termes, sources, avert),
            "sources": B._sources(*sources),
            "fiche": B._fiche(f"Hourdis {p['nom']} ({K.titre(cat)}) · {nom}", sym, val, unite, o, termes, sources, extrait, avert, texte=texte,
                              phrase=phrase),
            "params": o, "message": L.message()}


def rendre(params: Optional[Dict], hote: str) -> Dict:
    L, cat, p = lire(params)
    return _gamme(L, cat, p, hote) if L.out["vue"] == "gamme" else _profil(L, cat, p, hote)
