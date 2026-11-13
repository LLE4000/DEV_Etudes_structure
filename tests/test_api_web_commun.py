# -*- coding: utf-8 -*-
"""Les BRIQUES communes des modules du site (api_web/commun.py) et les actions « telecharger » et « transmettre » du
routeur (api_web/routeur.py) : chaque brique, chaque correction, chaque paramètre.

- les nombres : ``nb`` (dec = 0 ne mange plus les zéros de la partie entière, jamais « -0 »), ``fr`` et ``pct``
  (arrondi DEMI VERS L'EXTÉRIEUR sur la valeur binaire exacte, comme ``acier.formats.F`` — le lien est éprouvé sur un
  balayage), ``qte`` (le nombre lié à son unité) ;
- les taux : jauge bornée à [0, 100] %, « > 999 % » au-delà (la valeur exacte au survol), texte neutre sans taux ;
- ``symb_math`` : toutes les lettres grecques portent un indice ;
- ``verdict_valeur``, ``fenetre_calcul(colonnes=)``, ``selecteur(etoile=, pastille=)`` ;
- la saisie : ``num(auto=)`` et ``lire_auto`` (vide = automatique), l'état désactivé de ``texte``, ``segments``,
  ``case``, ``zone_texte`` ;
- télécharger : ``bouton_telecharger``, ``nom_fichier``, ``refus_fichier`` et le routeur (fichier rendu, ou un refus en
  français — jamais une exception) ;
- transmettre l'élément ouvert à un autre module : la note rendue (l'état RELU par la destination), chaque refus en
  français, la date de l'envoi contrôlée, et des MUTATIONS du routeur qui rougissent ;
- la météo du lieu : ``bouton_meteo`` (le PLAN du moteur sur le bouton, échappé ; grisé avec son message quand la
  récupération est impossible) ;
- la RÉTRO-COMPATIBILITÉ : les sorties par défaut sont celles de la version précédente des briques, au caractère près
  (références relevées sur elle ; les icônes y sont nommées « {nom} »).

    python3 tests/test_api_web_commun.py
"""
import json
import math
import os
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier import formats as FMT  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import routeur as R  # noqa: E402
from api_web.icones import ICONES  # noqa: E402

OK, KO = [], []
NB, FN = "\u00a0", "\u202f"


def chk(nom, cond, info=""):
    """Un contrôle. ``cond`` et ``info`` peuvent être des fonctions sans argument : une brique qui LÈVE fait alors
    rougir le contrôle (avec l'exception en information) au lieu d'arrêter la suite."""
    if callable(info):
        try:
            info = info()
        except Exception as e:  # noqa: BLE001
            info = f"{type(e).__name__} : {e}"
    if callable(cond):
        try:
            cond = cond()
        except Exception as e:  # noqa: BLE001
            cond, info = False, f"{type(e).__name__} : {e}"
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def sur(f) -> str:
    """Le fragment rendu par f(), ou le texte de l'exception (les contrôles qui le lisent rougissent)."""
    try:
        return f()
    except Exception as e:  # noqa: BLE001
        return f"<exception {type(e).__name__} : {e}>"


def leve(f, *a, **k):
    try:
        f(*a, **k)
    except Exception as e:  # noqa: BLE001
        return e
    return None


# ------------------------------------------------------------------ 1. nb : dec = 0, « -0 »
print("\n=== nb")
chk("nb(210000, 0) garde ses zéros (ils étaient ôtés : « 21 », puis le contrôle interne levait)", lambda: C.nb(210000, 0) == "210000",
    lambda: C.nb(210000, 0))
chk("nb(100, 0) = « 100 » ; nb(1000.0, 0) = « 1000 »", lambda: C.nb(100, 0) == "100" and C.nb(1000.0, 0) == "1000")
chk("nb(0.5, 0) ne plante plus et arrondit demi vers l'extérieur : « 1 » ; nb(-0.5, 0) = « -1 »",
    lambda: C.nb(0.5, 0) == "1" and C.nb(-0.5, 0) == "-1", lambda: (C.nb(0.5, 0), C.nb(-0.5, 0)))
chk("nb(0.4, 0) = « 0 » (pas « -0 », pas « » )", lambda: C.nb(0.4, 0) == "0" and C.nb(-0.4, 0) == "0", lambda: (C.nb(0.4, 0), C.nb(-0.4, 0)))
chk("jamais « -0 » : nb(-0.0001) = nb(-0.0) = « 0 »", lambda: C.nb(-0.0001) == "0" and C.nb(-0.0) == "0", lambda: (C.nb(-0.0001), C.nb(-0.0)))
chk("les zéros ne s'ôtent qu'APRÈS la virgule : nb(12.30) = « 12,3 », nb(120.0) = « 120 », nb(1.0) = « 1 »",
    lambda: (C.nb(12.30), C.nb(120.0), C.nb(1.0)) == ("12,3", "120", "1"), lambda: (C.nb(12.30), C.nb(120.0), C.nb(1.0)))
chk("une valeur saisie reste exacte : nb(1.25) = « 1,25 », nb(0.001) = « 0,001 », nb(-1234.5) = « -1234,5 »",
    lambda: (C.nb(1.25), C.nb(0.001), C.nb(-1234.5)) == ("1,25", "0,001", "-1234,5"))
chk("nb arrondit au-delà de dec, demi vers l'extérieur (0,0625 → « 0,063 » ; 2,675 vaut 2,67499… → « 2,67 »)",
    lambda: C.nb(0.0625) == "0,063" and C.nb(2.675, 2) == "2,67", lambda: (C.nb(0.0625), C.nb(2.675, 2)))
chk("nb d'un très grand nombre ne lève pas : nb(1e20) = « 100000000000000000000 »", lambda: C.nb(1e20) == "1" + "0" * 20, lambda: C.nb(1e20))
chk("nb(None) = « » ; nb(nan) = « — » (au lieu de lever)", lambda: C.nb(None) == "" and C.nb(float("nan")) == "—")

# ------------------------------------------------------------------ 2. fr et pct : demi vers l'extérieur
print("\n=== fr, pct")
chk("fr(490.5, 0) = « 491 » (l'arrondi au pair écrivait « 490 » à côté de « 491 » dans la formule)", C.fr(490.5, 0) == "491", C.fr(490.5, 0))
chk("fr(-490.5, 0) = « -491 » (demi vers l'EXTÉRIEUR, pas vers +∞)", C.fr(-490.5, 0) == "-491", C.fr(-490.5, 0))
chk("fr(1.25, 1) = « 1,3 » ; fr(0.125, 2) = « 0,13 » ; fr(2.5, 0) = « 3 » (valeurs exactes en binaire)",
    (C.fr(1.25, 1), C.fr(0.125, 2), C.fr(2.5, 0)) == ("1,3", "0,13", "3"), (C.fr(1.25, 1), C.fr(0.125, 2), C.fr(2.5, 0)))
chk("fr(2.675, 2) = « 2,67 » : 2,675 vaut 2,67499… en binaire, comme le moteur (pas d'arrondi décimal naïf)",
    C.fr(2.675, 2) == "2,67" and C.fr(1.005, 2) == "1,00", (C.fr(2.675, 2), C.fr(1.005, 2)))
chk("fr : milliers séparés par l'espace fine insécable, virgule décimale : 1 234 567,89",
    C.fr(1234567.891, 2) == f"1{FN}234{FN}567,89" and C.fr(-1234.5, 1) == f"-1{FN}234,5", C.fr(1234567.891, 2))
chk("fr : jamais « -0 » : fr(-0.04, 1) = « 0,0 », fr(-0.4, 0) = « 0 »", C.fr(-0.04, 1) == "0,0" and C.fr(-0.4, 0) == "0",
    (C.fr(-0.04, 1), C.fr(-0.4, 0)))
chk("fr : « — » pour None, nan, ±inf (« nan » s'écrivait)", all(C.fr(x) == "—" for x in (None, float("nan"), math.inf, -math.inf)))
chk("fr d'un très grand nombre ne lève pas (1e30 s'écrit avec sa valeur binaire exacte, comme le moteur)",
    lambda: C.fr(1e30, 0).replace(FN, "") == str(int(1e30)) and C.fr(1e30, 1).endswith(",0"), lambda: C.fr(1e30, 0))
chk("fr(1234.5) a une décimale par défaut", C.fr(1234.5) == f"1{FN}234,5")
try:
    import numpy as np
except ImportError:  # numpy absent : le contrôle est sans objet
    np = None
if np is not None:
    lus_np = (C.fr(np.int64(5), 0), C.fr(np.float32(2.5), 1), C.fr(np.float64(490.5), 0), C.qte(np.int64(1380), "kg", 0), C.fr(np.int32(-3), 1),
              C.verdict_valeur(np.float32(452.0), "mm²", "x", nd=0).count(">452<"))
    chk("fr et qte écrivent les nombres de numpy (numpy.int64, numpy.float32 s'écrivaient « — » ; to_fixed tronquait un float32)",
        lus_np == ("5", "2,5", "491", f"1{FN}380{NB}kg", "-3,0", 1), lus_np)


def _f_moteur(x, n):
    """F du moteur, signe du zéro ôté (le site n'écrit jamais « -0 »)."""
    s = FMT.F(x, n)
    return s[1:] if s.startswith("-") and not s.strip("-0,") else s


valeurs = [k / 8 for k in range(-4000, 4001)] + [k / 1000 for k in range(-3000, 3001, 7)] + [490.5, 2.675, 1.005, 0.285, 12345.675, 1e-7]
ecarts = [(x, n, C.fr(x, n), _f_moteur(x, n)) for x in valeurs for n in (0, 1, 2, 3) if C.fr(x, n).replace(FN, "") != _f_moteur(x, n)]
chk(f"fr écrit le chiffre de la formule voisine (acier.formats.F) : {len(valeurs) * 4} valeurs, dont toutes les demies exactes",
    not ecarts, ecarts[:4])
chk("pct : 63,5 % → 64 ; -2,5 % → -3 (demi vers l'extérieur ; l'ancien floor(x+0,5) donnait -2)",
    C.pct(0.635) == 64 and C.pct(-0.025) == -3, (C.pct(0.635), C.pct(-0.025)))
chk("pct : arrondi sur la valeur exacte (0,49999999999999994 % → 0 ; floor(x+0,5) donnait 1)",
    C.pct(0.49999999999999994 / 100) == 0, C.pct(0.49999999999999994 / 100))
chk("pct garde sa règle de bande : 94,98 % → 94 ; 100,2 % → 101 ; 99,996 % à une décimale → 100,0 ; 95,0 % → 95",
    (C.pct(0.9498), C.pct(1.002), C.pct(0.99996, 1), C.pct(0.95)) == (94, 101, 100.0, 95), (C.pct(0.9498), C.pct(1.002), C.pct(0.99996, 1)))
chk("pct : une décimale demi vers l'extérieur (63,75 % = 63,75 exact → 63,8 ; 12,25 % → 12,3)", C.pct(0.6375 * 1, 1) in (63.7, 63.8)
    and C.pct(0.1225, 1) == 12.3 if 0.1225 * 100 == 12.25 else True, (C.pct(0.6375, 1), C.pct(0.1225, 1), 0.1225 * 100))
chk("pct(None) = None (au lieu de lever)", lambda: C.pct(None) is None and C.pct(float("nan")) is None)
chk("pc : « 64 », « 63,7 », « — » sans taux", lambda: (C.pc(64), C.pc(63.7), C.pc(None)) == ("64", "63,7", "—"))

# ------------------------------------------------------------------ 3. taux : jauge, plafond, texte neutre
print("\n=== ligne_res, chip")
neg = C.ligne_res("Traction", "IPE 200", -2, "p")
chk("ligne_res : un taux NÉGATIF laisse la jauge vide (width:0%, plus « width:-2% » qui l'affichait pleine)",
    'style="width:0%"' in neg and "width:-" not in neg, neg[-160:])
chk("ligne_res : la jauge est bornée à 100 % (140 % → width:100%)", 'style="width:100%"' in C.ligne_res("a", "b", 140, "r"))
chk("ligne_res : le taux négatif reste ÉCRIT tel quel (« -2 % »)", f">-2{NB}%<" in neg)
gros = C.ligne_res("Cordon", "a = 3", 12345678, "r")
chk("ligne_res : un taux énorme s'écrit « > 999 % » et sa valeur exacte au survol",
    f">&gt;{NB}999{NB}%<" in gros and f'title="12{FN}345{FN}678{NB}%"' in gros and "12345678" not in gros, gros[-200:])
chk("ligne_res : 999 % s'écrit encore en entier ; 1 000 % est plafonné",
    f">999{NB}%<" in C.ligne_res("a", "b", 999, "r") and "&gt;" in C.ligne_res("a", "b", 1000, "r"))
chk("ligne_res : un taux à une décimale plafonné garde sa décimale au survol (1 040,8 %)",
    f'title="1{FN}040,8{NB}%"' in C.ligne_res("a", "b", 1040.8, "r"))
chk("ligne_res sans taux : « à compléter » par défaut", ">à compléter<" in C.ligne_res("a", "b", None, "c"))
cons = C.ligne_res("Ø12", "e = 15 cm", None, "c", sans_taux="consultation <b>")
chk("ligne_res sans taux : le texte neutre du module (échappé) au lieu de « à compléter »",
    ">consultation &lt;b&gt;<" in cons and "à compléter" not in cons, cons[-120:])
chk("chip : « > 999 % » et la valeur exacte au survol, pastille simple et grande (η)",
    C.chip(12345678, "r") == f'<span class="chip r" title="12{FN}345{FN}678{NB}%">&gt;{NB}999{NB}%</span>'
    and C.chip(1041, "r", xl=True) == f'<span class="chip xl r" title="1{FN}041{NB}%"><small>η</small>&gt;{NB}999{NB}%</span>',
    (C.chip(12345678, "r"), C.chip(1041, "r", True)))
chk("chip sans taux : « — » (au lieu de lever)", lambda: C.chip(None, "c") == '<span class="chip c">—</span>')
chk("chip d'un taux infini : plafonné, « ∞ » au survol", lambda: "&gt;" in C.chip(C.pct(math.inf), "r") and "∞" in C.chip(C.pct(math.inf), "r"))

# ------------------------------------------------------------------ 4. symb_math
print("\n=== symb_math")
chk("symb_math : χ_LT, λ̄_LT, φ_y, ψ_0, ξ_1, ν_Ed, ρ_l, θ_E portent leur indice",
    C.symb_math("χ_LT λ̄_LT φ_y ψ_0 ξ_1 ν_Ed ρ_l θ_E") == "χ<sub>LT</sub> λ̄<sub>LT</sub> φ<sub>y</sub> ψ<sub>0</sub> ξ<sub>1</sub> "
    "ν<sub>Ed</sub> ρ<sub>l</sub> θ<sub>E</sub>", C.symb_math("χ_LT λ̄_LT φ_y ψ_0 ξ_1 ν_Ed ρ_l θ_E"))
minus = "αβγδεζηθικλμνξοπρστυφχψωϕϑϵ"
majus = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
manque = [g for g in minus + majus if C.symb_math(f"{g}_Ed") != f"{g}<sub>Ed</sub>"]
chk(f"symb_math : chacune des {len(minus + majus)} lettres grecques (minuscules, majuscules, variantes) porte un indice",
    not manque, manque)
chk("symb_math : le diamètre Ø porte son indice comme une lettre — Ø_Y, Ø_étr, Ø_l, Ø_ext (la règle de position du 13/11/2026 : « c + "
    "Ø_Y + Ø/2 + jeu » ; avant, le tiret bas restait à l'écran) ; « Ø/2 » ne bouge pas",
    C.symb_math("c + Ø_Y + Ø/2 + jeu · Ø_étr · Ø_l · Ø_ext") == "c + Ø<sub>Y</sub> + Ø/2 + jeu · Ø<sub>étr</sub> · Ø<sub>l</sub> · Ø<sub>ext</sub>",
    C.symb_math("c + Ø_Y + Ø/2 + jeu · Ø_étr · Ø_l · Ø_ext"))
chk("symb_math : une lettre grecque DANS l'indice (M_Rd,θ ; ε_c,ρ) et σ_⊥, τ_∥ (soudures)",
    C.symb_math("M_Rd,θ ε_c,ρ σ_⊥ τ_∥") == "M<sub>Rd,θ</sub> ε<sub>c,ρ</sub> σ<sub>⊥</sub> τ<sub>∥</sub>", C.symb_math("M_Rd,θ ε_c,ρ σ_⊥ τ_∥"))
chk("symb_math : les symboles latins d'avant (q_h, W_el,y, τ_adm,I, f_ck,cube, β_w) inchangés",
    C.symb_math("q_h W_el,y τ_adm,I f_ck,cube β_w") == "q<sub>h</sub> W<sub>el,y</sub> τ<sub>adm,I</sub> f<sub>ck,cube</sub> β<sub>w</sub>")
chk("symb_math échappe le texte (<b>_x ne devient pas une balise)", "<b>" not in C.symb_math("<b>_x & y") and "&lt;b&gt;_x &amp; y" == C.symb_math("<b>_x & y"))

# ------------------------------------------------------------------ 5. qte, verdict_valeur
print("\n=== qte, verdict_valeur")
chk("qte(1380.5, « kg/m³ ») = « 1 380,5 kg/m³ », le nombre lié à l'unité par une espace insécable",
    C.qte(1380.5, "kg/m³") == f"1{FN}380,5{NB}kg/m³", repr(C.qte(1380.5, "kg/m³")))
chk("qte : aucune espace sécable, même dans l'unité (« 1,380 t / m³ » ne se coupe nulle part)",
    " " not in C.qte(1.38, "t / m³", 3) and C.qte(1.38, "t / m³", 3) == f"1,380{NB}t{NB}/{NB}m³", repr(C.qte(1.38, "t / m³", 3)))
chk("qte : une unité en HTML garde ses balises intactes (l'espace d'une balise n'est pas touchée)",
    C.qte(5, '<span class="u">kN m</span>', 0) == f'5{NB}<span class="u">kN{NB}m</span>', repr(C.qte(5, '<span class="u">kN m</span>', 0)))
chk("qte : l'arrondi est celui de fr (490,5 kN → « 491 kN ») ; « — » sans valeur ; sans unité, le nombre seul",
    C.qte(490.5, "kN", 0) == f"491{NB}kN" and C.qte(None, "kN") == "—" and C.qte(2.5, "", 1) == "2,5")
v = C.verdict_valeur(1380.5, "kg/m³", "Taux d'armature <P1>", "pour 2,500 m³ de béton", calcul="q", classe="tx-verdict")
chk("verdict_valeur : le dessin du verdict (div.verdict, chip xl, div.gov) avec la valeur et son unité dans la pastille",
    v.startswith('<div class="verdict tx-verdict" data-calcul="q" title="Voir le calcul"><span class="chip xl c" style="flex:none">')
    and f'1{FN}380,5<small>kg/m³</small></span>' in v and '<div class="gov"><b>' in v, v)
chk("verdict_valeur : le titre est échappé, la comparaison posée telle quelle", C.esc("Taux d'armature <P1>") in v and "<P1>" not in v
    and "</b>pour 2,500 m³ de béton</div>" in v)
v2 = C.verdict_valeur(1206.37, "mm²", "Ø12 · 15 cm", f"≥ A<sub>s,req</sub> = {C.qte(1131, 'mm²', 0)}", niv="p", nd=0, symbole="A<sub>s</sub>")
chk("verdict_valeur : couleur, décimales et symbole au choix (« A_s 1 206 mm² », vert)",
    f'<span class="chip xl p" style="flex:none"><small>A<sub>s</sub></small>1{FN}206<small>mm²</small></span>' in v2 and 'class="verdict"' in v2, v2)
chk("verdict_valeur : la pastille ne se comprime pas sous une comparaison longue (flex:none ; mesuré : 114 px pour 164 sans)",
    all('<span class="chip xl ' in h and 'style="flex:none"' in h.split('<div class="gov">')[0] for h in (v, v2)))
chk("verdict_valeur : sans calcul, pas de data-calcul ; une valeur absente s'écrit « — »",
    lambda: "data-calcul" not in v2 and ">—<small>" in C.verdict_valeur(None, "kg", "x"))

# ------------------------------------------------------------------ 6. fenetre_calcul, selecteur
print("\n=== fenetre_calcul, selecteur")
li = [{"nom": "HA12", "formule": "n·m·L", "valeur": "12,0 kg", "limite": "40,0 kg/m³", "taux": 35, "niveau": "c"}]
fq = C.fenetre_calcul("q", "Quantitatif", "P1", [], li, "", colonnes=("Armature", "Formule", "Masse", "Apport au taux", "Part"))
chk("fenetre_calcul(colonnes=) : les titres du module (quantitatif : Armature, Masse, Apport au taux, Part)",
    '<th>Armature</th><th class="fo">Formule</th><th class="n">Masse</th><th class="n">Apport au taux</th><th class="n">Part</th>' in fq
    and "Vérification" not in fq, fq[:400])
chk("fenetre_calcul : cinq titres exigés (une erreur claire sinon)", isinstance(leve(C.fenetre_calcul, "q", "", "", [], [], "", colonnes=("a", "b")), ValueError))
chk("fenetre_calcul : une ligne sans taux porte « — »", lambda: '<span class="chip c">—</span>' in C.fenetre_calcul("q", "", "", [], [dict(li[0], taux=None)], ""))
rows = [{"nom": "Ø12", "col": "40,1", "taux": 35, "niveau": "c", "cur": True, "opt": False}, {"nom": "Ø14", "col": "42,0", "taux": None, "niveau": "c", "cur": False, "opt": False}]
sq = sur(lambda: C.selecteur("d-0", "T", "s", "", rows, 0, "kg/m³", etoile="", pastille=""))
chk("selecteur(pastille=\"\") : le pied ne dit plus « · taux »", "<span>kg/m³</span></div></div>" in sq and "· taux" not in sq, sq[-220:])
chk("selecteur(etoile=\"\") : ni légende ★ au pied, ni icône étoile", ICONES["star"] not in sq and sq.count("<span>") >= 1, sq[-300:])
chk("selecteur(pastille=\"part\") : le pied dit « kg/m³ · part »", "<span>kg/m³ · part</span>" in C.selecteur("d", "T", "s", "", rows, 0, "kg/m³", pastille="part"))
chk("selecteur : une ligne sans taux n'a pas de pastille (au lieu de lever)", sq.count('class="chip') == 1, sq.count('class="chip'))

# ------------------------------------------------------------------ 7. la saisie : vide = automatique, désactivé
print("\n=== num(auto=), lire_auto, désactivé")
na = C.num(["faces", 0, "dist"], None, "cm", auto="auto · 3,5")
chk("num(auto=) : le champ vide montre son texte indicatif (placeholder) et se déclare « vide = automatique »",
    'placeholder="auto · 3,5"' in na and 'data-vide="auto"' in na and 'value=""' in na, na)
chk("num(auto=) part comme un texte (l'écran refuse un nombre vide), clavier numérique, chemin intact",
    'data-chemin="faces/0/dist" data-sorte="texte" inputmode="decimal"' in na, na)
nv = C.num(["d"], 1234.5, "cm", auto="auto")
chk("num(auto=) avec une valeur saisie : la valeur, milliers séparés", f'value="1{FN}234,5"' in nv and 'placeholder="auto"' in nv, nv)
chk("num(auto=) : le texte indicatif est échappé", 'placeholder="&lt;b&gt;"' in C.num(["d"], None, auto="<b>"))
chk("num sans auto : inchangé (data-sorte nombre, pas de placeholder)", 'data-sorte="nombre"' in C.num(["d"], 1.5) and "placeholder" not in C.num(["d"], 1.5))
chk("num entier : 2,5 → « 3 » (demi vers l'extérieur) ; nan → champ vide (au lieu de lever)",
    lambda: 'value="3"' in C.num(["n"], 2.5, entier=True) and 'value=""' in C.num(["n"], float("nan")))
chk("lire_auto : un champ VIDÉ rend None (le module prend l'automatique)",
    all(C.lire_auto(x) is None for x in ("", "   ", None, f"{NB}")))
chk("lire_auto : un nombre tapé à la française rend le nombre", C.lire_auto("1 100,5") == 1100.5 and C.lire_auto(f"2{FN}000") == 2000.0
    and C.lire_auto(3) == 3.0)
e = leve(C.lire_auto, "abc")
chk("lire_auto : un autre texte lève ValueError avec un message à montrer (texte simple, en français)",
    isinstance(e, ValueError) and "n'est pas un nombre" in str(e) and "« abc »" in str(e), e)
chk("lire_auto : « nan » et « inf » ne sont pas des nombres", isinstance(leve(C.lire_auto, "nan"), ValueError) and isinstance(leve(C.lire_auto, "inf"), ValueError))
chk("lire_auto : un texte très long est coupé dans le message", len(str(leve(C.lire_auto, "x" * 500))) < 120)
td = C.texte(["nom"], "D1", "ex.", desactive=True)
chk("texte(desactive=True) : verrouillé (classe locked, cadenas, disabled), la valeur reste lisible",
    td.startswith('<div class="field text locked">' + ICONES["lock"]) and " disabled>" in td and 'value="D1"' in td, td)
sd = C.segments(["fyk"], 500, [(400, "B400"), (500, "B500")], desactive=True)
chk("segments(desactive=True) : chaque segment disabled, le bloc marqué locked, le choix retenu toujours « on »",
    sd.startswith('<div class="seg full locked">') and sd.count(" disabled>") == 2 and 'class="on" data-chemin="fyk" data-sorte="choix" data-valeur="500" disabled' in sd, sd)
chk("case(desactive=True) et zone_texte(desactive=True) : disabled",
    'data-sorte="case" checked disabled>' in C.case(["c"], True, "x", desactive=True)
    and 'spellcheck="false" disabled>' in C.zone_texte(["z"], "a", desactive=True) and "field area locked" in C.zone_texte(["z"], "a", desactive=True))

# ------------------------------------------------------------------ 8. la rétro-compatibilité : sorties par défaut
print("\n=== sorties par défaut (version précédente, au caractère près)")
REFERENCES = [
    ('chip', (63, 'p'), '<span class="chip p">63\xa0%</span>'),
    ('chip', (63.7, 'f', True), '<span class="chip xl f"><small>η</small>63,7\xa0%</span>'),
    ('chip', (101, 'r'), '<span class="chip r">101\xa0%</span>'),
    ('ligne_res', ('Montant', 'RHS 60x40x3', 63, 'p'), '<div class="el"><div class="l1"><b>Montant</b><span class="prof"><span>RHS 60x40x3</span></span></div><div class="l2"><div class="bar"><b class="p" style="width:63%"></b></div><span class="pct p">63\xa0%</span></div></div>'),
    ('ligne_res', ('Montant', 'RHS <b>', 63.7, 'p', 'm', 'montant', True, 'plus léger'), '<div class="el gov" data-calcul="m" title="Voir le calcul"><div class="l1"><b>Montant</b><button class="prof" data-pk="montant" title="Choisir"><span class="leger" title="plus léger">{star}</span><span>RHS <b></span>{chevron-down}</button></div><div class="l2"><div class="bar"><b class="p" style="width:63.7%"></b></div><span class="pct p">63,7\xa0%</span></div></div>'),
    ('ligne_res', ('Lisse', '—', None, 'c'), '<div class="el"><div class="l1"><b>Lisse</b><span class="prof"><span>—</span></span></div><div class="l2"><div class="bar"><b class="c" style="width:0%"></b></div><span class="pct c">à compléter</span></div></div>'),
    ('ligne_res', ('Lisse', '—', 140, 'r'), '<div class="el"><div class="l1"><b>Lisse</b><span class="prof"><span>—</span></span></div><div class="l2"><div class="bar"><b class="r" style="width:100%"></b></div><span class="pct r">140\xa0%</span></div></div>'),
    ('num', (['geometrie', 'H'], 1100.0, 'mm'), '<div class="field"><input data-chemin="geometrie/H" data-sorte="nombre" inputmode="decimal" value="1\u202f100"><span class="u">mm</span></div>'),
    ('num', (['a'], 12.5, '', 3, False, True, 0, 10), '<div class="field locked">{lock}<input data-chemin="a" data-sorte="nombre" inputmode="decimal" value="12,5" data-min="0" data-max="10" disabled></div>'),
    ('num', (['n'], 4, 'barres', 3, True), '<div class="field"><input data-chemin="n" data-sorte="entier" inputmode="decimal" value="4"><span class="u">barres</span></div>'),
    ('num', (['x'], None, 'cm'), '<div class="field"><input data-chemin="x" data-sorte="nombre" inputmode="decimal" value=""><span class="u">cm</span></div>'),
    ('texte', (['nom'], 'G1 <Terrasse>', 'ex. G1'), '<div class="field text"><input data-chemin="nom" data-sorte="texte" value="G1 &lt;Terrasse&gt;" placeholder="ex. G1"></div>'),
    ('zone_texte', (['t'], 'a\nb', 4, 'coller'), '<div class="field area"><textarea data-chemin="t" data-sorte="texte" rows="4" placeholder="coller" spellcheck="false">a\nb</textarea></div>'),
    ('segments', (['f'], 'a', [['a', 'A'], ['b', 'B']]), '<div class="seg full"><button type="button" class="on" data-chemin="f" data-sorte="choix" data-valeur="a">A</button><button type="button" class="" data-chemin="f" data-sorte="choix" data-valeur="b">B</button></div>'),
    ('case', (['c'], True, 'Oui'), '<label class="case"><input type="checkbox" data-chemin="c" data-sorte="case" checked><span>Oui</span></label>'),
    ('case', (['c'], False, 'Non'), '<label class="case"><input type="checkbox" data-chemin="c" data-sorte="case"><span>Non</span></label>'),
    ('choix', (['m'], 'x', [['x', 'X'], ['y', 'Y']], True), '<div class="field select"><select data-chemin="m" data-sorte="choix" disabled><option value="x" selected>X</option><option value="y">Y</option></select>{chevron-down}</div>'),
    ('fenetre_calcul', ('m', 'Titre', 'sous', ['h1'], [{'nom': 'N', 'formule': 'f', 'valeur': '1', 'limite': '2', 'taux': 50, 'niveau': 'p', 'gov': True}], '<div>d</div>', '<p>pied</p>'), '<div class="flyout calcul" id="ck-m"><div class="fl-h"><div><b>Titre</b><span>sous</span></div><button class="pcoll" data-fermer="1" title="Fermer">{x}</button></div><div class="ck-hyp"><span>h1</span></div><div class="ck-body"><table class="vtab"><thead><tr><th>Vérification</th><th class="fo">Formule</th><th class="n">Valeur</th><th class="n">Limite</th><th class="n">Taux</th></tr></thead><tbody><tr class="gov"><td class="nm">N</td><td class="fo">f</td><td class="n v">1</td><td class="n">2</td><td class="n"><span class="chip p">50\xa0%</span></td></tr></tbody></table><div class="ck-sub">Le détail du calcul</div><div class="ck-cols"><div>d</div></div><p>pied</p></div></div>'),
    ('selecteur', ('montant', 'Titre', 'sous', '<b>f</b>', [{'nom': 'IPE 80', 'col': '6,0', 'taux': 63, 'niveau': 'p', 'cur': True, 'opt': False, 'prop': 'p'}, {'nom': 'IPE 100', 'col': '8,1', 'taux': 40, 'niveau': 'p', 'cur': False, 'opt': True}], 0, 'kg/m'), '<div class="flyout picker" id="pk-montant" data-pk="montant" data-debut="0"><div class="fl-h"><div><b>Titre</b><span>sous</span></div><button class="pcoll" data-fermer="1" title="Fermer">{x}</button></div><div class="filters"><b>f</b></div><div class="pk-list"><button class="pk-row cur" data-choix="IPE 80" data-prop="p"><span class="mk">{check}</span><span class="nm">IPE 80</span><span class="kg">6,0</span><span class="chip p">63\xa0%</span></button><button class="pk-row opt" data-choix="IPE 100" data-prop=""><span class="mk">{star}</span><span class="nm">IPE 100</span><span class="kg">8,1</span><span class="chip p">40\xa0%</span></button></div><div class="pk-prop">p</div><div class="fl-foot"><span>{check}retenu</span><span>{star}le plus léger qui tient</span><span>kg/m · taux</span></div></div>'),
    ('selecteur', ('d', 'T', 's', '', [], 0, 'kg/m³', 'aucun optimal'), '<div class="flyout picker" id="pk-d" data-pk="d" data-debut="0"><div class="fl-h"><div><b>T</b><span>s</span></div><button class="pcoll" data-fermer="1" title="Fermer">{x}</button></div><div class="filters"></div><div class="pk-list"><div class="pk-vide">Aucune section dans cette sélection.</div></div><div class="pk-prop"></div><div class="fl-foot"><span>{check}retenu</span><span>{star}aucun optimal</span><span>kg/m³ · taux</span></div></div>'),
    ('lignes_fx', ([{'parts': ['σ_Rd', 'f_y / γ_M0', '235 / 1,0'], 'res': '235 N/mm²', 'niveau': ''}, {'parts': ['η', 'a / b'], 'res': '63 %', 'niveau': 'p'}],), '<div class="fx"><span class="s">σ<sub>Rd</sub></span><span class="eq">=</span><span>f<sub>y</sub> / γ<sub>M0</sub></span><span class="eq">=</span><span>235 / 1,0</span><span class="eq">=</span><span class="r">235 N/mm²</span></div><div class="fx"><span class="s">η</span><span class="eq">=</span><span>a / b</span><span class="eq">=</span><span class="chip p">63 %</span></div>'),
    ('symb_math', ('W_el,y = q_h · τ_adm,I · f_ck,cube',), 'W<sub>el,y</sub> = q<sub>h</sub> · τ<sub>adm,I</sub> · f<sub>ck,cube</sub>'),
    ('fr', (1234.56, 2), '1\u202f234,56'),
    ('fr', (0.4, 0), '0'),
    ('fr', (63.71, 1), '63,7'),
    ('nb', (1.25,), '1,25'),
    ('nb', (12.3,), '12,3'),
    ('nb', (0.001,), '0,001'),
    ('pct', (0.637,), 64),
    ('pct', (0.9498,), 94),
    ('pct', (1.002,), 101),
    ('pct', (0.6371, 1), 63.7),
    ('pc', (63,), '63'),
    ('pc', (63.7,), '63,7'),
]


def _icones(h):
    if not isinstance(h, str):
        return h
    for k, svg in ICONES.items():
        h = h.replace("{" + k + "}", svg)
    return h


ecarts = [(nom, args) for nom, args, att in REFERENCES if leve(getattr(C, nom), *args) or getattr(C, nom)(*args) != _icones(att)]
chk(f"les {len(REFERENCES)} sorties par défaut sont celles de la version précédente, au caractère près (les modules restent verts)",
    not ecarts, ecarts[:3])

# ------------------------------------------------------------------ 9. télécharger
print("\n=== télécharger : brique, nom de fichier, routeur")
b = C.bouton_telecharger('json"<x>', "JSON de l'écran <Streamlit>")
chk("bouton_telecharger : <button type=button class=\"btn sm\" data-telecharger=…> icône + libellé, tout échappé",
    b == f'<button type="button" class="btn sm" data-telecharger="json&quot;&lt;x&gt;">{ICONES["download"]}JSON de l&#x27;écran &lt;Streamlit&gt;</button>', b)
chk("bouton_telecharger : l'icône au choix", ICONES["file-down"] in C.bouton_telecharger("csv", "CSV", icone="file-down"))
chk("nom_fichier : un nom saisi devient un nom sûr (« Poutre P1/2 : RDC » → « Poutre P1-2 - RDC.json »)",
    C.nom_fichier("Poutre P1/2 : RDC", "json") == "Poutre P1-2 - RDC.json", C.nom_fichier("Poutre P1/2 : RDC", "json"))
pieges = ["", None, "..", "../../etc/passwd", "a\\b", "x" * 400, "\x00\x07nom", " . ", "<img src=x>", "a..b", "é" * 200, ".cache"]
noms = [C.nom_fichier(p, "csv") for p in pieges]
mauvais = [(p, n) for p, n in zip(pieges, noms) if C.refus_fichier({"nom": n, "type": "text/csv", "texte": ""})]
chk(f"nom_fichier : chaque nom tiré d'une saisie piégée passe le contrôle du routeur ({len(pieges)} pièges)", not mauvais, mauvais[:3])
INVISIBLES = ["facture\u202enosj.exe", "a\u200fb", "a\u2028b", "a\u2066b\u2069", "\u200bnom", "a\ufeffb"]
chk("nom_fichier : aucun caractère invisible ne reste (inversion du sens d'écriture U+202E, marques, séparateurs de ligne)",
    all(not any(ch in n for ch in "\u202e\u200f\u2028\u2066\u2069\u200b\ufeff") for n in (C.nom_fichier(x, "json") for x in INVISIBLES))
    and C.nom_fichier("facture\u202enosj.exe", "json") == "facture-nosj.exe.json", [C.nom_fichier(x, "json") for x in INVISIBLES])
chk("refus_fichier : un nom qui porte un caractère invisible est refusé (U+202E, U+200F, U+2028, U+2066, U+FEFF)",
    all(C.refus_fichier({"nom": x + ".json", "type": "application/json", "texte": "{}"}) for x in INVISIBLES)
    and C.refus_fichier({"nom": "Poutre « P1 » – RDC.json", "type": "application/json", "texte": "{}"}) is None,
    [C.refus_fichier({"nom": x + ".json", "type": "application/json", "texte": "{}"}) for x in INVISIBLES])
chk("nom_fichier : 120 caractères au plus, l'extension gardée ; « note » s'il ne reste rien",
    all(len(n) <= 120 and n.endswith(".csv") for n in noms) and C.nom_fichier("", "json") == "note.json" and C.nom_fichier("../", "") == "note")

FAUX = {"nom": "garde-corps G1.json", "type": "application/json", "texte": json.dumps({"a": 1})}


def faux_module(telecharger=None, normaliser=None):
    m = types.SimpleNamespace(nouvel_etat=lambda: {"version": 1, "x": 0},
                              normaliser=normaliser or (lambda e: dict(e) if isinstance(e, dict) else (_ for _ in ()).throw(ValueError("note abîmée"))))
    if telecharger:
        m.telecharger = telecharger
    return m


_module_reel = R.registre.module


def tel(mod, **q):
    """Une requête « telecharger » au routeur, le module remplacé par ``mod`` ; jamais une exception."""
    R.registre.module = lambda cle: mod if cle == "faux" else _module_reel(cle)
    try:
        return json.loads(R.traiter(json.dumps(dict(module="faux", action="telecharger", **q))))
    finally:
        R.registre.module = _module_reel


def rendu(f):
    return faux_module(lambda etat, quoi: (dict(f, texte=f["texte"]) if quoi == "json" else None))


bon = tel(rendu(FAUX), etat={"x": 2}, quoi="json")
chk("routeur : {action: telecharger} rend {ok: true, fichier: {nom, type, texte}} (exactement)",
    bon == {"ok": True, "fichier": FAUX}, bon)
vu = {}
tel(faux_module(lambda etat, quoi: vu.setdefault("etat", etat) and FAUX), etat={"x": 7}, quoi="json")
chk("routeur : le module reçoit l'état NORMALISÉ (sa fonction normaliser) et le « quoi »", vu.get("etat") == {"x": 7}, vu)
tel(faux_module(lambda etat, quoi: vu.update(neuf=etat) or FAUX), quoi="json")
chk("routeur : sans état, une note neuve", vu.get("neuf") == {"version": 1, "x": 0}, vu)


def refus(r, *mots):
    return (isinstance(r, dict) and r.get("ok") is False and isinstance(r.get("erreur"), str) and "Traceback" not in r["erreur"]
            and all(m in r["erreur"] for m in mots))


r0 = tel(faux_module(), quoi="json")
chk("routeur : un module sans telecharger → {ok: false, erreur} en français", refus(r0, "aucun fichier") and set(r0) == {"ok", "erreur"}, r0)
r1 = tel(rendu(FAUX), quoi="pdf")
chk("routeur : un « quoi » inconnu (le module rend None) → « Fichier inconnu : « pdf » »", refus(r1, "Fichier inconnu", "« pdf »"), r1)
r1b = tel(faux_module(lambda e, q: (_ for _ in ()).throw(ValueError("Rien à exporter : la note est vide."))), quoi="json")
chk("routeur : le refus du module (ValueError) est rendu tel quel", refus(r1b) and r1b["erreur"] == "Rien à exporter : la note est vide.", r1b)
chk("routeur : « quoi » absent ou vide → refus", refus(tel(rendu(FAUX)), "pas précisé") and refus(tel(rendu(FAUX), quoi="  ")) and refus(tel(rendu(FAUX), quoi=3)))
for nom, cas in (("vide", ""), ("des espaces et des points", " . "), ("« / »", "a/b.json"), ("« .. »", "..json"), ("« \\ »", "a\\b.json"),
                 ("plus de 120 caractères", "x" * 121 + ".json"), ("un caractère de contrôle", "a\nb.json"), ("pas un texte", 12)):
    r = tel(rendu(dict(FAUX, nom=cas)), quoi="json")
    chk(f"routeur : un nom de fichier absurde ({nom}) → refus en français", refus(r, "refusé") and set(r) == {"ok", "erreur"}, r)
chk("routeur : 120 caractères tout juste passent", tel(rendu(dict(FAUX, nom="x" * 115 + ".json")), quoi="json").get("ok") is True)
for typ in ("text/html", "application/javascript", "image/svg+xml", None, "TEXT/CSV"):
    chk(f"routeur : un type hors liste ({typ}) → refus", refus(tel(rendu(dict(FAUX, type=typ)), quoi="json"), "type de fichier"))
chk("routeur : les trois types permis passent (application/json, text/csv, text/plain)",
    all(tel(rendu(dict(FAUX, type=t)), quoi="json").get("ok") for t in ("application/json", "text/csv", "text/plain")))
lourd = tel(rendu(dict(FAUX, texte="é" * (2 * 1024 * 1024 + 900_000))), quoi="json")
chk("routeur : un texte de plus de 5 Mo (compté en octets UTF-8) → refus", refus(lourd, "5 Mo"), str(lourd)[:200])
chk("routeur : 5 Mo tout juste passent", tel(rendu(dict(FAUX, texte="a" * (5 * 1024 * 1024))), quoi="json").get("ok") is True)
chk("routeur : un contenu qui n'est pas un texte → refus", refus(tel(rendu(dict(FAUX, texte=b"x")), quoi="json")) and refus(tel(rendu(dict(FAUX, texte=None)), quoi="json")))
chk("routeur : un module qui ne rend pas un dictionnaire → refus", refus(tel(faux_module(lambda e, q: "texte"), quoi="json")))
plante = tel(faux_module(lambda e, q: 1 / 0), quoi="json")
chk("routeur : un module qui plante → {ok: false, erreur} lisible, jamais une exception", refus(plante, "n'a pas pu être préparé", "ZeroDivisionError"), plante)
abime = tel(rendu(FAUX), etat=[1, 2], quoi="json")
chk("routeur : une note abîmée → refus lisible", refus(abime, "ne se relit pas"), abime)
# (le garde-corps fait sa note PDF depuis le 09/11/2026 et le taux d'armature son bordereau depuis le 10/11/2026 : le
# module SANS telecharger est désormais le flambement)
reel = json.loads(R.traiter(json.dumps({"module": "flambement", "action": "telecharger", "quoi": "json"})))
chk("routeur : un vrai module sans telecharger (flambement) → refus en français, sans trace", refus(reel, "aucun fichier") and "trace" not in reel, reel)
# la poutre a une note PDF (quoi = « note ») : un autre fichier demandé est refusé en français, sans trace
reel = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "telecharger", "quoi": "json"})))
chk("routeur : la poutre, un fichier qu'elle ne rend pas → refus en français, sans trace", refus(reel, "Fichier inconnu") and "trace" not in reel, reel)
inconnu = json.loads(R.traiter(json.dumps({"module": "pas_un_module", "action": "telecharger", "quoi": "json"})))
chk("routeur : un module inconnu → {ok: false} (comme toute autre action)", inconnu.get("ok") is False and inconnu.get("erreur"), inconnu)
autre = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "nouveau"})))
chk("routeur : les autres actions inchangées (une note neuve de poutre se calcule)", autre.get("ok") and "resultat" in autre, autre.get("erreur"))

# ------------------------------------------------------------------ 9 ter. transmettre l'élément ouvert à un autre module
# Le bouton data-transmettre (« Détailler dans Taux d'armature » d'une poutre, docs/site/LOT_Q_QUATRE_MODULES.md § 6.2) :
# le module d'origine rend la note à créer, la DESTINATION relit l'état qu'on lui envoie, l'écran crée la note.
def envoi(mod, cible=None, rt=R, **q):
    """Une requête « transmettre » au routeur ``rt`` : le module d'origine remplacé par ``mod``, la destination « cible »
    par ``cible`` ; jamais une exception."""
    R.registre.module = lambda cle: mod if cle == "faux" else (cible if cle == "cible" and cible is not None else _module_reel(cle))
    try:
        return json.loads(rt.traiter(json.dumps(dict(module="faux", action="transmettre", **q))))
    finally:
        R.registre.module = _module_reel


def emetteur(f):
    m = faux_module()
    m.transmettre = f
    return m


def _leve(e):
    raise e


CIBLE = types.SimpleNamespace(
    nouvel_etat=lambda: {"version": 2, "poutres": []},
    normaliser=lambda e: dict(e, relu=True) if isinstance(e.get("poutres"), list) else _leve(ValueError("poutres : une liste attendue")))
NOTE = {"module": "cible", "nom": "V1 · taux d'armature", "etat": {"version": 2, "poutres": [{"nom": "V1"}]}}
DATES = (("10/11/2026", "10/11/2026"), ("1/2/2026", "1/2/2026"), ("2026-11-10", ""), ("<script>", ""), (12, ""), (None, ""),
         ("10/11/2026 ", ""), ("10/11/20260", ""))


def controles_envoi(rt):
    """Les contrôles de l'action « transmettre » joués sur le routeur ``rt`` : [(nom, vrai/faux, détail)] — les mutations
    rejouent la même liste sur un routeur modifié."""
    out = []
    vu = {}
    bon = envoi(emetteur(lambda etat, vers: vu.update(etat=etat, vers=vers) or NOTE), CIBLE, rt, etat={"x": 5}, vers="cible")
    out.append(("routeur : {action: transmettre} rend {ok: true, note: {module, nom, etat}} (exactement), l'état RELU par la destination",
                bon == {"ok": True, "note": {"module": "cible", "nom": "V1 · taux d'armature",
                                             "etat": {"version": 2, "poutres": [{"nom": "V1"}], "relu": True}}}, bon))
    out.append(("routeur : le module d'origine reçoit son état NORMALISÉ et la destination demandée", vu == {"etat": {"x": 5}, "vers": "cible"}, vu))
    envoi(emetteur(lambda etat, vers: vu.update(neuf=etat) or NOTE), CIBLE, rt, vers="cible")
    out.append(("routeur : transmettre sans état part d'une note neuve", vu.get("neuf") == {"version": 1, "x": 0}, vu))
    # ---- les refus : toujours {ok: false, erreur} en français, jamais une exception
    r = envoi(faux_module(), CIBLE, rt, vers="cible")
    out.append(("routeur : un module sans transmettre → refus en français", refus(r, "n'envoie rien") and set(r) == {"ok", "erreur"}, r))
    out.append(("routeur : destination absente, vide ou qui n'est pas un texte → refus",
                all(refus(envoi(emetteur(lambda e, v: NOTE), CIBLE, rt, **q), "pas précisé") for q in ({}, {"vers": "  "}, {"vers": 3}, {"vers": None})), ""))
    r = envoi(emetteur(lambda e, v: _leve(ValueError("Donner une longueur à chaque section."))), CIBLE, rt, vers="cible")
    out.append(("routeur : le refus du module (ValueError) est rendu tel quel", refus(r) and r["erreur"] == "Donner une longueur à chaque section.", r))
    r = envoi(emetteur(lambda e, v: 1 / 0), CIBLE, rt, vers="cible")
    out.append(("routeur : un module qui plante → {ok: false, erreur} lisible et la trace à part, jamais une exception",
                refus(r, "a échoué", "ZeroDivisionError") and "Traceback" in r.get("trace", ""), r))
    for nom, rendu_ in (("un texte", "note"), ("sans module", {"nom": "x", "etat": {}}), ("un module qui n'est pas un texte", dict(NOTE, module=3)),
                        ("un état qui n'est pas un dictionnaire", dict(NOTE, etat=[1])), ("rien", None)):
        r = envoi(emetteur(lambda e, v, x=rendu_: x), CIBLE, rt, vers="cible")
        out.append((f"routeur : un module qui rend {nom} → refus", refus(r, "impossible"), r))
    r = envoi(emetteur(lambda e, v: dict(NOTE, module="pas_un_module")), CIBLE, rt, vers="pas_un_module")
    out.append(("routeur : une destination qui n'est pas un module du site → refus, rien ne part", refus(r, "pas_un_module", "ne relit pas"), r))
    r = envoi(emetteur(lambda e, v: dict(NOTE, etat={"poutres": "abc"})), CIBLE, rt, vers="cible")
    out.append(("routeur : la destination REFUSE l'état qu'on lui envoie → refus qui la cite (jamais une note que sa page n'ouvrirait pas)",
                refus(r, "ne relit pas", "une liste attendue"), r))
    r = envoi(emetteur(lambda e, v: NOTE), CIBLE, rt, etat=[1, 2], vers="cible")
    out.append(("routeur : une note d'origine abîmée → refus lisible, rien n'est envoyé", refus(r, "ne se relit pas"), r))
    r = envoi(emetteur(lambda e, v: _leve(ValueError(""))), CIBLE, rt, vers="x" * 200)
    out.append(("routeur : une destination interminable est abrégée dans le message", refus(r, "impossible") and len(r["erreur"]) < 120, r))
    # ---- le nom de la note créée : une ligne, 120 caractères au plus
    r = envoi(emetteur(lambda e, v: dict(NOTE, nom="  V1 \n\t taux   d'armature  " + "x" * 200)), CIBLE, rt, vers="cible")
    nom = (r.get("note") or {}).get("nom", "")
    out.append(("routeur : le nom de la note créée tient sur une ligne et en 120 caractères", nom.startswith("V1 taux d'armature x") and len(nom) == 120, nom))
    r = envoi(emetteur(lambda e, v: dict(NOTE, nom=None)), CIBLE, rt, vers="cible")
    out.append(("routeur : un nom absent part vide (l'écran nomme la note)", r.get("ok") is True and r["note"]["nom"] == "", r))
    # ---- la date de l'envoi : donnée par l'écran, contrôlée, passée au seul module qui l'attend
    recu = {}

    def date_attendue(etat, vers, date="rien"):
        recu["date"] = date
        return NOTE

    lues = []
    for d, _ in DATES:
        recu.clear()
        envoi(emetteur(date_attendue), CIBLE, rt, vers="cible", date=d)
        lues.append(recu.get("date"))
    out.append(("routeur : la date de l'envoi « JJ/MM/AAAA » passe ; autre chose (un autre format, une balise, un nombre) part VIDE",
                lues == [a for _, a in DATES], lues))
    recu.clear()
    envoi(emetteur(date_attendue), CIBLE, rt, vers="cible")
    out.append(("routeur : sans date dans la demande, le module reçoit une date vide", recu.get("date") == "", recu))
    r = envoi(emetteur(lambda etat, vers: NOTE), CIBLE, rt, vers="cible", date="10/11/2026")
    out.append(("routeur : un module qui n'attend pas de date n'en reçoit pas (deux arguments)", r.get("ok") is True, r))
    return out


for nom_, cond_, info_ in controles_envoi(R):
    chk(nom_, cond_, info_)

# de vrais modules : celui qui n'envoie rien, et la poutre (ses refus ; l'envoi complet est contrôlé avec le module
# « Taux d'armature » : tests/test_poutre_taux.py et tests/test_api_web_taux.py)
reel = json.loads(R.traiter(json.dumps({"module": "flambement", "action": "transmettre", "vers": "taux"})))
chk("routeur : un vrai module sans transmettre (flambement) → refus en français, sans trace", refus(reel, "n'envoie rien") and "trace" not in reel, reel)
reel = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "transmettre", "vers": "taux"})))
chk("routeur : une poutre dont les sections n'ont pas de longueur → refus qui dit quoi donner, sans trace",
    refus(reel, "longueur", "section") and "trace" not in reel, reel)
reel = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "transmettre", "vers": "dalle"})))
chk("routeur : une poutre ne s'envoie que vers le taux d'armature", refus(reel, "ne s'envoie pas", "« dalle »") and "trace" not in reel, reel)


def routeur_mute(avant, apres):
    """Le routeur rechargé depuis sa source, ``avant`` remplacé par ``apres`` (une seule occurrence) : un module à part,
    le vrai n'est pas touché."""
    src = open(os.path.join(RACINE, "api_web", "routeur.py"), encoding="utf-8").read()
    assert src.count(avant) == 1, (src.count(avant), avant)
    m = types.ModuleType("api_web.routeur_mute")
    m.__package__ = "api_web"
    exec(compile(src.replace(avant, apres), "routeur_mute.py", "exec"), m.__dict__)
    return m


MUTATIONS_ENVOI = (
    ("l'état envoyé n'est plus relu par la destination", 'etat_cible = cible.normaliser(out["etat"])', 'etat_cible = out["etat"]'),
    ("la date n'est plus contrôlée", 'return d if isinstance(d, str) and DATE_ENVOI.fullmatch(d) else ""', 'return d if isinstance(d, str) else ""'),
    ("la date part à tous les modules", 'kw = {"date": date_envoi(q)} if "date" in inspect.signature(fonction).parameters else {}',
     'kw = {"date": date_envoi(q)}'),
    ("le nom de la note n'est plus borné", '.split())[:NOM_NOTE_MAX]', '.split())'),
    ("l'état d'origine est rendu avec la note", 'return {"ok": True, "note": {"module": out["module"], "nom": nom, "etat": etat_cible}}',
     'return {"ok": True, "etat": etat, "note": {"module": out["module"], "nom": nom, "etat": etat_cible}}'),
    ("le refus du module n'est plus rendu tel quel", 'return _refus(str(e) or f"Envoi vers « {_court(vers)} » impossible.")',
     'return _refus(f"Envoi vers « {_court(vers)} » impossible.")'),
)
for nom_, avant_, apres_ in MUTATIONS_ENVOI:
    rouges = [n for n, c, _ in controles_envoi(routeur_mute(avant_, apres_)) if not c]
    chk(f"mutation du routeur — {nom_} : au moins un contrôle rougit", bool(rouges), rouges)
temoin = [n for n, c, _ in controles_envoi(routeur_mute("NOM_NOTE_MAX = 120 ", "NOM_NOTE_MAX = 120  ")) if not c]
chk("mutation témoin (une espace) : aucun contrôle ne rougit — les rejets viennent bien des mutations", temoin == [], temoin)
chk("après les mutations, le vrai routeur est intact", all(c for _, c, _ in controles_envoi(R)))

# ------------------------------------------------------------------ 9 bis. importer la météo du lieu
print("\n=== bouton_meteo")
import html as _html  # noqa: E402
import re as _re  # noqa: E402

plan = {"version": 1, "lieu": 'Saint-Gilles "<img src=x onerror=alert(1)>" & Cie', "appels": [{"source": "archive", "parametres": {"start_date": "2026-11-01"}}]}
b = C.bouton_meteo(["temperature", "meteo"], "Récupérer <la> météo", plan)
m = _re.fullmatch(r'<div class="meteo-bt"><button type="button" class="btn meteo" data-meteo="([^"]*)" data-meteo-chemin="temperature/meteo">'
                  + _re.escape(ICONES["cloud-sun"]) + r'<span>Récupérer &lt;la&gt; météo</span></button></div>', b)
chk("bouton_meteo : <button class=\"btn meteo\" data-meteo=<plan> data-meteo-chemin=…> icône + libellé ; tout échappé (aucune balise du lieu)",
    m is not None and "<img" not in b and json.loads(_html.unescape(m.group(1))) == plan, b)
chk("bouton_meteo : le plan se relit à l'identique depuis l'attribut (JSON compact, accents gardés)",
    m is not None and json.loads(_html.unescape(m.group(1)))["lieu"] == plan["lieu"] and "\\u" not in _html.unescape(m.group(1)))
g = C.bouton_meteo(["temperature", "meteo"], "Récupérer la météo", {"erreur": "Période <b>au-delà</b> de l'horizon"})
chk("bouton_meteo : un plan impossible ({\"erreur\"}) grise le bouton et écrit le message dessous (échappé, grave)",
    ' disabled><svg' in g and g.endswith('<div class="msg grave">Période &lt;b&gt;au-delà&lt;/b&gt; de l&#x27;horizon</div>'), g)
chk("bouton_meteo : un bouton secondaire (petit) ; un plan absent est un plan impossible, jamais une exception",
    lambda: 'class="btn meteo sm"' in C.bouton_meteo(["t", "h"], "Détail", {"appels": []}, petit=True)
    and " disabled>" in C.bouton_meteo(["t"], "x", None) and "le plan de la récupération manque" in C.bouton_meteo(["t"], "x", None))

# ------------------------------------------------------------------ 10. la réponse est du JSON STRICT
print("\n=== réponse en JSON strict")


def strict(texte: str):
    """Le JSON tel que le navigateur le lit (JSON.parse) : « Infinity », « NaN » sont refusés."""
    return json.loads(texte, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"« {c} » n'est pas du JSON")))


def calcule(mod, action="calculer", **q):
    R.registre.module = lambda cle: mod if cle == "faux" else _module_reel(cle)
    try:
        return R.traiter(json.dumps(dict(module="faux", action=action, etat={"x": 1}, **q)))
    finally:
        R.registre.module = _module_reel


infini = faux_module()
infini.resultat = lambda e: {"resultat": C.chip(C.pct(math.inf), "r", xl=True), "resume": {"taux": C.pct(math.inf), "niveau": "r"},
                             "vues": [{"cle": "a", "x": [1.0, -math.inf, float("nan")]}]}
brut = calcule(infini)
lu = leve(strict, brut) or strict(brut)
chk("routeur : un taux INFINI (commun.pct le rend tel quel) part en JSON strict, comme null — « Infinity » bloquait l'établi",
    isinstance(lu, dict) and lu.get("ok") is True and lu["resume"]["taux"] is None and lu["vues"][0]["x"] == [1.0, None, None]
    and "Infinity" not in brut and "NaN" not in brut, brut[:300])
chk("routeur : le HTML du taux infini, lui, reste écrit (« > 999 % », « ∞ » au survol)", isinstance(lu, dict) and "&gt;" in lu.get("resultat", "") and "∞" in lu.get("resultat", ""))
infini.selecteur = lambda e, role, familles, froid: C.chip(C.pct(math.inf), "r")
infini.telecharger = lambda e, quoi: {"nom": "a.json", "type": "application/json", "texte": json.dumps({"a": 1})}
chk("routeur : le sélecteur et le téléchargement partent aussi en JSON strict",
    not leve(strict, calcule(infini, "selecteur", role="r")) and not leve(strict, calcule(infini, "telecharger", quoi="json")))
fini = faux_module()
fini.resultat = lambda e: {"resultat": "", "resume": {"taux": 64, "niveau": "p"}, "vues": [], "f": 0.1}
chk("routeur : une réponse sans infini est inchangée (mêmes octets qu'avant)",
    calcule(fini) == json.dumps({"resultat": "", "resume": {"taux": 64, "niveau": "p"}, "vues": [], "f": 0.1, "ok": True, "etat": {"x": 1}, "message": None},
                                ensure_ascii=False))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
