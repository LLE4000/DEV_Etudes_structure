# -*- coding: utf-8 -*-
"""Le DESSIN QUI SE MANIPULE des assemblages : chaque objet cliquable mène à un champ qui EXISTE (26/11/2026).

Le propriétaire : « avant, dans mon module d'assemblage métallique, je cliquais sur le profilé, je pouvais le choisir ;
je cliquais sur l'assemblage, je pouvais choisir les boulons, le plat. Avant, je cliquais sur le dessin et je pouvais
tout choisir. […] Si oui, que tu réactives ça, parce que c'était assez pratique. »

Le kit des schémas (``acier/assemblages/.../schemas.py``) écrit sur chaque objet ce qu'il règle ; le site coupait ces
attributs. ``api_web/dessin_assemblages.manipulable`` les traduit dans le vocabulaire de ``site/web/js/plan.js`` :
une cote se tape (``data-plan-cote``), une pièce ouvre son sélecteur (``data-plan-pk``) ou mène à son champ
(``data-plan-champ``), « + » et « − » règlent les rangées (``data-plan-clic``).

Ce que cette suite verrouille, sur les SEPT configurations (et des variantes : ailes soudées, profilé personnalisé, les
trois niveaux de cotation, le pied de poteau complet) :

  1. le CONTRAT (R1 : le dessin est un raccourci, jamais le seul accès) — chaque cote cliquable porte le chemin d'un
     champ du cadre Données rendu, et la valeur que le moteur a relue ; chaque sélecteur ouvert par une pièce existe et
     se rend ; chaque champ visé est dans les Données ; chaque « + » / « − » règle une valeur admise ;
  2. rien ne PROMET un geste qu'il ne tient pas : pas de classe ``ed``, de ``tabindex`` ni de ``role="button"`` sans
     cible ; le glisser du kit (non branché) ne laisse aucun attribut ;
  3. un MINIMUM de cibles par configuration (sans lui, la suite passerait à vide) ;
  4. l'ALLER-RETOUR : la valeur tapée sur une cote part au moteur par le chemin du champ (``modifier``), et le dessin
     refait la porte ; « + » ajoute une rangée, et il n'y a pas de « − » à une rangée ni de « + » au maximum ;
  5. des MUTATIONS : la traduction abîmée (elle ne traduit plus, vise un champ absent des Données, laisse un « + » au
     maximum, garde le glisser) fait rougir les contrôles ci-dessus.

Lancement : python3 tests/test_dessin_assemblages.py
"""
import collections
import importlib.util
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from api_web import assemblages as A  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_assemblages as DA  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


SOUDEE = "Soudée"
# le minimum de cibles d'une configuration, toutes vues confondues : (cotes, sélecteurs, champs ou sélecteurs en tout)
MINI = {"doubles_cornieres": (14, 6), "plat_ame": (12, 5), "platine_about": (12, 5), "plat_ame_poteau": (12, 5),
        "doubles_cornieres_poteau": (14, 6), "platine_about_poteau": (12, 5), "platine_beton": (12, 1)}


def etat_de(cle, **modifs):
    etat = A.nouvel_etat()
    etat["assemblages"] = [A.nouvel_assemblage(cle)]
    u = etat["assemblages"][0]["values"]
    inconnues = [k for k in modifs if k not in u]
    assert not inconnues, inconnues
    u.update(modifs)
    return etat


def chemins_donnees(r):
    return set(re.findall(r'data-chemin="([^"]+)"', r["donnees"]))


def fautes_contrat(etat, module=DA, vues=None):
    """Les manquements au contrat des dessins de l'assemblage ouvert : ([texte], compte des cibles)."""
    el = A._actif(etat)
    cfg = A._cfg(el)
    r = A.resultat(etat)
    R = A.calculer(cfg, el)[0]
    cfg = A.pour(cfg, R)
    don = chemins_donnees(r)
    roles = set(A._selecteurs(cfg))
    if vues is None:
        vues = r["vues"]
        if module is not DA:                      # la traduction mutée, sur les mêmes dessins du kit
            unites = {k: A._unite(f) for k, f in cfg.E.CHAMPS.items()}
            niveau = int(el.get("cotation", 1))
            vues = [{"cle": c, "svg": module.manipulable(d.svg(identifiant=f"asm_{c}"), R.u, don, roles, unites, A.COMPTES_MAXI, A.SIGNEES)}
                    for c, _t, _co, d, _l in cfg.vues(R, niveau, A._hl(R))]
    f, compte = [], collections.Counter()
    for v in vues:
        svg, ou = v["svg"], v["cle"]
        c = module.cibles(svg) if hasattr(module, "cibles") else DA.cibles(svg)
        compte.update(cotes=len(c["cotes"]), pk=len(c["pk"]), champs=len(c["champs"]), clics=len(c["clics"]))
        for x in c["cotes"]:
            k = x["chemin"][2:]
            if x["chemin"] not in don:
                f.append(f"{ou} : la cote {x['chemin']} n'est pas un champ des Données")
            elif abs(float(R.u.get(k)) - x["valeur"]) > 1e-9:
                f.append(f"{ou} : la cote {k} porte {x['valeur']}, le moteur a relu {R.u.get(k)}")
            elif x["unite"] != A._unite(cfg.E.CHAMPS[k]):
                f.append(f"{ou} : la cote {k} dit « {x['unite']} », le champ « {A._unite(cfg.E.CHAMPS[k])} »")
            if bool(x.get("signe")) != (k in A.SIGNEES):
                f.append(f"{ou} : la cote {k} — signe {x.get('signe')}")
        for role in c["pk"]:
            if role not in roles:
                f.append(f"{ou} : le sélecteur « {role} » n'existe pas dans cette configuration")
        for ch in c["champs"]:
            if ch not in don:
                f.append(f"{ou} : le champ visé {ch} n'est pas dans les Données")
        for x in c["clics"]:
            k = x["chemin"][2:]
            n = int(float(R.u.get(k)))
            if x["chemin"] not in don or abs(x["valeur"] - n) != 1 or not 1 <= x["valeur"] <= A.COMPTES_MAXI.get(k, 50):
                f.append(f"{ou} : « ± » {x} (n = {n})")
        # rien ne promet un geste qu'il ne tient pas
        for balise in re.findall(r"<g[^>]*>", svg):
            geste = re.search(r' data-plan-(cote|pk|champ|clic)="', balise)
            promet = re.search(r'class="[^"]*\bed\b', balise) or 'tabindex="0"' in balise or 'role="button"' in balise
            if promet and not geste:
                f.append(f"{ou} : un objet promet un geste sans cible — {balise[:90]}")
            if geste and not ('tabindex="0"' in balise and 'role="button"' in balise and "aria-label=" in balise):
                f.append(f"{ou} : une cible sans clavier ni nom — {balise[:90]}")
        if re.search(r'data-d(rag2?|sym2?)="', svg):
            f.append(f"{ou} : un attribut du glisser du kit est resté (le glisser n'est pas branché)")
    return f, compte


print("=== 1. Le contrat, sur les sept configurations ===")
import api_web.assemblages_pied  # noqa: E402,F401  (enregistre le pied de poteau)

total = collections.Counter()
for cle in A.ordre():
    etat = etat_de(cle)
    f, n = fautes_contrat(etat)
    total.update(n)
    mc, mp = MINI[cle]
    chk(f"{cle} : chaque objet cliquable des dessins mène à un champ des Données ou à un sélecteur de la configuration, avec "
        f"la valeur du moteur ; rien ne promet un geste sans cible ; {n['cotes']} cotes, {n['pk']} sélecteurs, "
        f"{n['champs']} champs, {n['clics']} « ± » (au moins {mc} cotes, {mp} pièces)",
        not f and n["cotes"] >= mc and n["pk"] + n["champs"] >= mp, f"{f[:3]} {dict(n)}")
chk(f"au moins 100 cotes cliquables sur les sept configurations ({total['cotes']}), 30 pièces ({total['pk'] + total['champs']})",
    total["cotes"] >= 100 and total["pk"] + total["champs"] >= 30, str(dict(total)))

print("\n=== 2. Les pièces : ce qu'un appui ouvre ===")
for cle, attendu in (("doubles_cornieres", {"prof_P", "prof_S", "boulon", "corniere"}), ("plat_ame", {"prof_P", "prof_S", "boulon"}),
                     ("platine_about", {"prof_P", "prof_S", "boulon"}), ("plat_ame_poteau", {"prof_P", "prof_S", "boulon"}),
                     ("doubles_cornieres_poteau", {"prof_P", "prof_S", "boulon", "corniere"}),
                     ("platine_about_poteau", {"prof_P", "prof_S", "boulon"})):
    etat = etat_de(cle)
    r = A.resultat(etat)
    pk = set(x for v in r["vues"] for x in DA.cibles(v["svg"])["pk"])
    rendus = {role: A.selecteur(etat, role) for role in pk}
    chk(f"{cle} : les profilés, les boulons" + (", les cornières" if "corniere" in attendu else "")
        + " ouvrent leur SÉLECTEUR — celui du résultat, rendu par le moteur",
        pk == attendu and all(f'id="pk-{role}"' in h and "pk-row" in h for role, h in rendus.items()), str(pk))
    ch = set(x for v in r["vues"] for x in DA.cibles(v["svg"])["champs"])
    piece = "v/tp_u" if "cornieres" not in cle else None
    chk(f"{cle} : les efforts mènent à « V_Ed », l'étiquette d'un groupe de boulons à ses rangées"
        + (", la pièce d'attache à son épaisseur" if piece else ""),
        "v/V_Ed" in ch and any(x in ch for x in ("v/n1_u", "v/n1S_u")) and (piece is None or piece in ch), str(ch))
# des ailes soudées : les cordons mènent à leur gorge ; il n'y a plus de boulons à choisir de ce côté
for cle in ("doubles_cornieres", "doubles_cornieres_poteau"):
    etat = etat_de(cle, fix_P=SOUDEE, fix_S=SOUDEE)
    f, n = fautes_contrat(etat)
    r = A.resultat(etat)
    ch = set(x for v in r["vues"] for x in DA.cibles(v["svg"])["champs"])
    pk = set(x for v in r["vues"] for x in DA.cibles(v["svg"])["pk"])
    chk(f"{cle}, les deux ailes soudées : les cordons mènent à leur gorge (a_S, a_P) ; aucun boulon à choisir ; le contrat tient",
        not f and {"v/a_S", "v/a_P"} <= ch and "boulon" not in pk and n["clics"] == 0, f"{f[:2]} {ch} {pk}")
# un profilé personnalisé : la pièce garde son sélecteur (le catalogue), ses dimensions sont dans les Données
etat = etat_de("plat_ame")
A.modifier(etat, ["v", "prof_S"], A.PERSO)
f, n = fautes_contrat(etat)
chk("plat d'âme, poutre secondaire « Personnalisé » : le contrat tient (la poutre ouvre toujours son sélecteur)", not f, str(f[:3]))
# les trois niveaux de cotation
for niveau in (0, 1, 2):
    for cle in ("doubles_cornieres", "platine_about_poteau"):
        etat = etat_de(cle)
        A.modifier(etat, ["cotation"], str(niveau))
        f, n = fautes_contrat(etat)
        # (« Simple » : la vue ne porte presque aucune cote — les pièces, elles, restent cliquables)
        chk(f"{cle}, cotes « {A.NIVEAUX_COTATION[niveau][1]} » : le contrat tient ({n['cotes']} cotes cliquables, "
            f"{n['pk'] + n['champs']} pièces" + (" ; les cotes calculées du niveau « Toutes » restent inertes)" if niveau == 2 else ")"),
            not f and n["cotes"] >= (0 if niveau == 0 else 10) and n["pk"] + n["champs"] >= 5, f"{f[:3]} {dict(n)}")
# le pied de poteau : la platine à la main (tiges, élément soudé) et la vérification complète
etat = etat_de("platine_beton")
r = A.resultat(etat)
ch = set(x for v in r["vues"] for x in DA.cibles(v["svg"])["champs"])
chk("pied de poteau, platine à la main : chaque tige mène à SA ligne de la table (v/tiges/i/x), l'élément soudé à la sienne, "
    "le massif, le mortier et la platine à leurs champs",
    {"v/tiges/0/x", "v/tiges/3/x", "v/elems/0/type", "v/bc_u", "v/t_g", "v/tp_u"} <= ch, str(sorted(ch)))
ex = [i for i, _lab, d in A.exemples(A.CONFIGS["platine_beton"]) if d.get("mode") not in (None, "manuel")]
if ex:
    A.modifier(etat, ["exemple"], ex[0])
    f, n = fautes_contrat(etat)
    chk(f"pied de poteau, vérification complète (exemple {ex[0]}) : le contrat tient ({n['cotes']} cotes, "
        f"{n['pk'] + n['champs']} pièces)", not f and n["cotes"] >= 6, f"{f[:3]} {dict(n)}")

print("\n=== 3. L'aller-retour : une cote tapée, une rangée de plus ===")
for cle, k in (("doubles_cornieres", "p1S_u"), ("plat_ame", "e1_u"), ("platine_about", "bp_u"), ("plat_ame_poteau", "hp_u"),
               ("doubles_cornieres_poteau", "gA_u"), ("platine_about_poteau", "p3_u"), ("platine_beton", "tp_u")):
    etat = etat_de(cle)
    r = A.resultat(etat)
    cotes = {x["chemin"]: x for v in r["vues"] for x in DA.cibles(v["svg"])["cotes"]}
    x = cotes.get(f"v/{k}")
    if x is None:
        chk(f"{cle} : la cote {k} est cliquable", False, str(sorted(cotes)))
        continue
    nouvelle = x["valeur"] + 5
    msg = A.modifier(etat, x["chemin"].split("/"), C.nb(nouvelle))
    r2 = A.resultat(etat)
    cotes2 = {y["chemin"]: y for v in r2["vues"] for y in DA.cibles(v["svg"])["cotes"]}
    chk(f"{cle} : {k} = {C.nb(nouvelle)} tapé SUR la cote part au moteur par le chemin du champ ; le champ des Données et le "
        "dessin refait portent la nouvelle valeur",
        msg is None and A._actif(etat)["values"][k] == nouvelle and cotes2[f"v/{k}"]["valeur"] == nouvelle
        and re.search(rf'data-chemin="v/{k}"[^>]*value="{re.escape(C.nb(nouvelle))}"', r2["donnees"]) is not None, f"{msg} {cotes2.get('v/' + k)}")
for cle, k in (("plat_ame", "n1_u"), ("doubles_cornieres", "n1S_u"), ("platine_about_poteau", "n1_u")):
    etat = etat_de(cle)
    u = A._actif(etat)["values"]
    n0 = int(u[k])
    r = A.resultat(etat)
    clics = [x for v in r["vues"] for x in DA.cibles(v["svg"])["clics"] if x["chemin"] == f"v/{k}"]
    plus = next(x for x in clics if x["valeur"] == n0 + 1)
    msg = A.modifier(etat, plus["chemin"].split("/"), plus["valeur"])
    chk(f"{cle} : « + » du groupe ajoute une rangée ({n0} → {n0 + 1}), « − » en retire une — par le champ {k}",
        msg is None and int(u[k]) == n0 + 1 and sorted(x["valeur"] for x in clics) == [n0 - 1, n0 + 1], f"{msg} {clics}")
    # à une rangée : pas de « − » ; au maximum : pas de « + » (le bouton reste dessiné, éteint)
    for n, absent in ((1, 0), (A.COMPTES_MAXI[k], A.COMPTES_MAXI[k] + 1)):
        cfg = A._cfg(A._actif(etat))
        R = A.moteur(cfg, dict(u, **{k: n}))
        niveau = 1
        don = chemins_donnees(A.resultat(etat))
        svgs = [DA.manipulable(d.svg(identifiant="x"), R.u, don, set(A._selecteurs(cfg)), {}, A.COMPTES_MAXI)
                for _c, _t, _co, d, _l in A.pour(cfg, R).vues(R, niveau, None)]
        vals = [x["valeur"] for s in svgs for x in DA.cibles(s)["clics"] if x["chemin"] == f"v/{k}"]
        eteints = sum(len(re.findall(r'<g class="[^"]*\bgb\b[^"]*\boff\b', s)) for s in svgs)
        chk(f"{cle}, {k} = {n} : pas de « {'−' if n == 1 else '+'} » cliquable (le bouton est éteint, sans geste)",
            absent not in vals and len(vals) >= 1 and eteints >= 1, f"{vals} {eteints}")

print("\n=== 4. Le raccourci ne mène qu'à ce qui existe (R1) ===")
etat = etat_de("plat_ame")
el = A._actif(etat)
cfg = A._cfg(el)
R = A.calculer(cfg, el)[0]
don = chemins_donnees(A.resultat(etat))
brut = A.pour(cfg, R).vues(R, 1, None)[0][3].svg(identifiant="x")
complet = DA.cibles(DA.manipulable(brut, R.u, don, set(A._selecteurs(cfg))))
sans = DA.cibles(DA.manipulable(brut, R.u, don - {"v/e1_u", "v/tp_u", "v/prof_S"}, set(A._selecteurs(cfg)) - {"boulon"}))
chk("un champ ABSENT des Données n'est pas une cible : sans « e1 », sa cote redevient inerte ; sans « tp », le plat mène à "
    "sa largeur ; sans le champ du profilé, la poutre ne s'ouvre plus ; sans le sélecteur des boulons, ils mènent à leur champ",
    "v/e1_u" in {x["chemin"] for x in complet["cotes"]} and "v/e1_u" not in {x["chemin"] for x in sans["cotes"]}
    and "v/tp_u" in complet["champs"] and "v/tp_u" not in sans["champs"] and "v/bp_u" in sans["champs"]
    and "prof_S" in complet["pk"] and "prof_S" not in sans["pk"]
    and "boulon" in complet["pk"] and "boulon" not in sans["pk"] and "v/boulon_u" in sans["champs"], f"{sans}")
vide = DA.manipulable(brut, R.u, set(), set())
chk("sans cadre Données (aucun champ, aucun sélecteur), le dessin est INERTE : ni cible, ni classe « ed », ni tabindex",
    not any(DA.cibles(vide).values()) and not re.search(r'class="[^"]*\bed\b', vide) and "tabindex" not in vide and 'data-drag="' not in vide)
chk("une valeur qui n'est pas un nombre (un champ vidé) ne fait pas une cote cliquable",
    "v/e1_u" not in {x["chemin"] for x in DA.cibles(DA.manipulable(brut, dict(R.u, e1_u=""), don, set()))["cotes"]})
chk("le dessin n'est PAS abîmé par la traduction : mêmes formes, mêmes textes (seuls les attributs des groupes changent)",
    re.sub(r"<g[^>]*>", "<g>", DA.manipulable(brut, R.u, don, set(A._selecteurs(cfg)))) == re.sub(r"<g[^>]*>", "<g>", brut))

print("\n=== 5. Mutations : la traduction abîmée fait rougir les contrôles ===")
SOURCE = open(os.path.join(RACINE, "api_web", "dessin_assemblages.py"), encoding="utf-8").read()


def mutant(ancien, nouveau):
    assert SOURCE.count(ancien) == 1, (ancien, SOURCE.count(ancien))
    spec = importlib.util.spec_from_loader("api_web.dessin_assemblages_mutant", loader=None)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "api_web"
    exec(compile(SOURCE.replace(ancien, nouveau), "dessin_assemblages_mutant.py", "exec"), m.__dict__)
    return m


def fautes_absent(m):
    """Le contrôle 4 rejoué sur une traduction ``m`` : les cibles posées vers ce qui n'existe pas."""
    s = m.cibles(m.manipulable(brut, R.u, don - {"v/e1_u", "v/tp_u", "v/prof_S"}, set(A._selecteurs(cfg)) - {"boulon"}))
    f = []
    if "v/e1_u" in {x["chemin"] for x in s["cotes"]}:
        f.append("la cote e1 reste cliquable sans son champ")
    if "v/tp_u" in s["champs"] or "prof_S" in s["pk"] or "boulon" in s["pk"]:
        f.append("une pièce mène à un champ ou à un sélecteur absent")
    return f


def fautes_maxi(m):
    """« + » au maximum de rangées : la traduction ``m`` le laisse-t-elle cliquable ?"""
    Rm = A.moteur(cfg, dict(el["values"], n1_u=A.COMPTES_MAXI["n1_u"]))
    s = m.manipulable(A.pour(cfg, Rm).vues(Rm, 1, None)[0][3].svg(identifiant="x"), Rm.u, don, set(), {}, A.COMPTES_MAXI)
    return [x for x in m.cibles(s)["clics"] if x["valeur"] > A.COMPTES_MAXI["n1_u"]]


MUTATIONS = [
    ("la traduction ne traduit plus (le dessin du kit tel quel)",
     "    return _BALISE.sub(balise, svg)\n", "    return svg\n",
     lambda m: fautes_contrat(etat_de("plat_ame"), m)[0]),
    ("une cote cliquable sans champ dans les Données",
     '            if f"v/{cle}" in chemins and v is not None:', "            if v is not None:", fautes_absent),
    ("une pièce mène à un champ absent des Données",
     "        if chemin in chemins:\n            return \"champ\", chemin", "        if True:\n            return \"champ\", chemin", fautes_absent),
    ("une pièce ouvre un sélecteur que la configuration n'a pas",
     '            if essai[1] in roles and f"v/{essai[2]}" in chemins:', "            if True:", fautes_absent),
    ("« + » reste cliquable au maximum de rangées",
     '1 <= suivant <= maxi.get(cle_a, 50):', '1 <= suivant:', fautes_maxi),
    ("le glisser du kit garde ses attributs (un curseur « main » sans geste)",
     "        for k in _GLISSER:\n            attrs.pop(k, None)\n", "",
     lambda m: fautes_contrat(etat_de("plat_ame"), m)[0]),
    ("un objet sans cible garde sa promesse (classe « ed », tabindex)",
     "        if not actif:\n            for k in _GESTE:\n                attrs.pop(k, None)", "        if False:\n            for k in _GESTE:\n                attrs.pop(k, None)",
     lambda m: [1] if re.search(r'class="[^"]*\bed\b', m.manipulable(brut, R.u, set(), set())) else []),
    ("la valeur d'une cote n'est plus celle que le moteur a relue",
     '                cote = {"chemin": f"v/{cle}", "valeur": v,', '                cote = {"chemin": f"v/{cle}", "valeur": v + 1,',
     lambda m: fautes_contrat(etat_de("platine_about"), m)[0]),
]
chk("témoin : la traduction telle quelle passe ces contrôles", not any(c(DA) for _t, _a, _n, c in MUTATIONS))
survivants = [titre for titre, ancien, nouveau, controle in MUTATIONS if not controle(mutant(ancien, nouveau))]
chk(f"{len(MUTATIONS)} mutations de la traduction : toutes vues rouges", not survivants, str(survivants))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)
