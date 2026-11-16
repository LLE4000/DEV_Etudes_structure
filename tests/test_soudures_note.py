# -*- coding: utf-8 -*-
"""Note de calcul « Soudures » (lot 3) : ``acier/soudures/note.py`` et le
peintre SVG partagé ``ndc_pdf/svg.py``.

Chaque garantie rougit si on la retire :
  1. FORME : DEUX pages A4 paysage, aucun débordement de colonne, sur les
     huit configurations par défaut et cinq variantes (CHS par tronçons, bout
     à bout partiel, cercle + segment libres, console en méthode ELS, HEA 200
     sous six efforts) et sur un refus (gorge 2 mm) ; moins de 2 s par note.
  2. CONTENU p.1 : statut, η max, cordon et point gouvernants, critère, les
     quatre méthodes comparées avec la méthode choisie marquée, la ligne de
     configuration (dimensions, cordons, acier de référence), la ligne des
     efforts et de la méthode, tableau des CORDONS, des EFFORTS, des
     VÉRIFICATIONS PAR POINT (σ⊥, τ⊥, τ∥, σ_c, η), CONCLUSION.
  3. DESSIN PEINT depuis le MÊME SVG que l'écran : ses textes sont dans le
     PDF (titre de la vue, cotes, efforts, encart de la gorge) ; le cartouche
     porte l'échelle.
  4. PAGE 2 : MÉTHODE ET DÉTAIL, les sections de ``methode.py`` (un fragment
     de chaque section retrouvé mot pour mot), références, carte des taux,
     cartouche (cas, configuration, méthode, acier, date).
  5. MÉTHODE ELS : la note dit « efforts de service », k et son équivalent
     Eurocode, et le statut suit (console à 94 % attention sous 40 kN).
  6. PEINTRE partagé : arcs d'un ``path`` échantillonnés (un cercle devient
     ≥ 24 points), hachures d'un ``pattern`` peintes en lignes découpées,
     dégradé de trait peint en tronçons, ``dy`` d'un texte honoré, chemin
     à trous (evenodd) accepté ; le peintre du plancher reste un alias.

Lancement : python3 tests/test_soudures_note.py (depuis la racine).
"""
import io
import os
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402

from acier.soudures import etat as ET  # noqa: E402
from acier.soudures import methode as ME  # noqa: E402
from acier.soudures import note as NT  # noqa: E402
from ndc_pdf import svg as SV  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def texte(pdf: bytes, page: int = 0) -> str:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return doc[page].get_text()


def norm(s: str) -> str:
    return " ".join(s.replace(" ", " ").split())


def serre(s: str) -> str:
    """Sans aucun espace : les titres à lettres espacées (track) ressortent « D E S S I N » de l'extraction."""
    return norm(s).replace(" ", "")


ET_ = ET.nouvel_etat()


def cas_de(cfg, params=None, efforts=None, methode=None):
    c = ET.nouveau_cas(cfg)
    c["params"].update(params or {})
    c["efforts"].update(efforts or {})
    c["methode"].update(methode or {})
    return c


# ================================================================
print("\n=== 1. Forme : deux pages, aucun débordement, treize cas ===")
cas_liste = [(cfg, cas_de(cfg)) for cfg in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")]
cas_liste += [("C6 CHS par tronçons", cas_de("C6", {"profil": "CHS 114.3x5", "troncons": {"n": 4, "l": 50}}, {"N": 60, "Vz": 15, "My": 4, "T": 1})),
              ("C7 partielle", cas_de("C7", {"penetration": "partielle", "p": 6})),
              ("C8 cercle + segment", cas_de("C8", {"elements": [{"type": "arc", "xc": 0, "yc": 0, "r": 60, "ang1": 0, "ang2": 360, "a": 4, "cote": 1, "troncons": {"n": 3, "l": 50}},
                                                                   {"type": "segment", "x1": -120, "y1": -80, "x2": 120, "y2": -80, "a": 5, "cote": 1}]},
                                            {"N": 20, "Vx": 10, "Vy": -30, "Mx": 2, "Mz": 1})),
              ("C3 ELS", cas_de("C3", None, {"V": 40}, {"mode": "els"})),
              ("C5 HEA 200 six efforts", cas_de("C5", {"profil": "HEA 200"}, {"N": -40, "Vz": 60, "Vy": 10, "My": -25, "Mz": 3, "T": 2}))]
pdfs = {}
for nom, cas in cas_liste:
    t0 = time.time()
    pdf = NT.generer_note(ET_, cas, {"nom_projet": "Hall B", "partie": "console", "indice": "A"})
    dt = time.time() - t0
    pdfs[nom] = pdf
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    chk(f"{nom} : 2 pages A4 paysage, variante normale sans débordement, {dt:.2f} s",
        doc.page_count == 2 and all(abs(p.rect.width - 842) < 1 and abs(p.rect.height - 595) < 1 for p in doc)
        and NT.derniere_variante == "normale" and not NT.derniers_avertissements and dt < 2.0,
        f"{doc.page_count} p · {NT.derniere_variante} · {NT.derniers_avertissements} · {dt:.2f} s")
pdf_refus = NT.generer_note(ET_, cas_de("C3", {"a": 2.0}))
chk("refus (gorge 2 mm) : la note sort quand même, « CALCUL IMPOSSIBLE », message du moteur",
    pymupdf.open(stream=pdf_refus, filetype="pdf").page_count == 2 and "CALCUL IMPOSSIBLE" in texte(pdf_refus) and "ne compte pas" in texte(pdf_refus))

# ================================================================
print("\n=== 2. Contenu de la page 1 (console par défaut) ===")
t1 = norm(texte(pdfs["C3"]))
chk("bandeau : CONFORME, η max 61 %, cordon droit · haut (le côté tendu, premier des deux bouts à égalité), critère",
    "CONFORME" in t1 and "η max 61 %" in t1 and "cordon droit, haut" in t1 and "σ_c ≤ fu/(βw·γM2)" in t1)
chk("la méthode CHOISIE seulement : le bandeau la nomme, pas de comparaison des quatre méthodes dans la note",
    "Eurocode, méthode directionnelle (EN 1993-1-8 §4.5.3.2)" in t1 and "COMPARAISON" not in serre(t1) and "EC3 simplifiée" not in t1 and "ELS bureau k" not in t1)
chk("VÉRIFICATION AU POINT GOUVERNANT, écrite comme à la main : A_w = 2 × 5 × 190, |f| = √(…) = 764 N/mm, σ_c = √(105,8² + 3 × (105,8² + 31,58²)) = 218,5, "
    "lim_c = 360/(0,8 × 1,25) = 360, η = max(η_c, η⊥) = 60,7 %",
    "VÉRIFICATIONAUPOINTGOUVERNANT" in serre(t1) and "A_w = 2·a·l_eff = 2 × 5 × 190 = 1900 mm²" in t1
    and "|f| = √(f_x² + f_y² + f_z²) = √(0² + −157,9² + 747,9²) = 764 N/mm" in t1
    and "√(105,8² + 3 × (105,8² + 31,58²)) = 218,5 N/mm²" in t1 and "360/(0,8 × 1,25) = 360 N/mm²" in t1 and "η = max(η_c, η⊥)" in t1 and "60,7 %" in t1)
chk("ligne de configuration : plat 200 × 10 sur l'aile d'un HEB 200, e 150, deux faces, cordons d'angle a 5 continu, acier de référence S235",
    "Console fléchie : plat 200 × 10 sur l'aile d'un HEB 200, V à e 150 mm de la face, deux faces, cordons d'angle a 5 continu" in t1
    and "référence S235 (fu 360, βw 0,8)" in t1)
chk("ligne des efforts et de la méthode : V = 60,0 kN, efforts de calcul (ELU), limites 360 / 259,2 / 207,8",
    "V = 60,0 kN" in t1 and "efforts de calcul (ELU)" in t1 and "360,0" in t1 and "259,2" in t1 and "207,8" in t1)
chk("tableau CORDONS : cordon droit et gauche, a 5,0, l_eff 190, |f| 764, η 61 %", "cordon droit" in t1 and "cordon gauche" in t1 and "190" in t1 and "764" in t1)
chk("tableau EFFORTS : V et N avec le sens ; propriétés du groupe (A_w 1 900 mm², J)", "Charge verticale vers le bas" in t1 and "A_w 1 900 mm²" in t1 and "J 576" in t1)
chk("VÉRIFICATIONS PAR POINT : en tête σ⊥ +106 (traction : la console s'arrache en haut), τ⊥ 106, τ∥ 32, σ_c 219 ; en pied σ⊥ −106 (compression)",
    "106" in t1 and "-106" in t1 and "32" in t1 and "219" in t1)
chk("CONCLUSION : taux maximal, méthode nommée en clair, cordons qui portent", "CONCLUSION" in serre(t1) and "Taux maximal 61 % sur cordon droit, haut, selon l'Eurocode" in t1 and "Les cordons portent" in t1)
chk("légende et notations en pied de page", "σ_c = √(σ⊥² + 3(τ⊥² + τ∥²))" in t1 and "Dessins : pièces au trait fort" in t1 and "EN 22553" in t1)
chk("ligne de méthode : Eurocode, directionnelle, γM2 = 1,25, limites 360,0 / 259,2 / 207,8", "méthode directionnelle (§4.5.3.2), γM2 = 1,25" in t1 and "207,8 N/mm²" in t1)

# ================================================================
print("\n=== 3. Le dessin peint depuis le même SVG ===")
chk("les trois scènes du kit peintes en vectoriel : titre des vues, HEB 200 du catalogue, cotes e = 150 / h = 200 / b_p = 200, V = 60,0 kN, ◉ 61 %",
    "DESSINS—élévation·coupehorizontale·vuedeface" in serre(t1) and "HEB 200" in t1 and "e = 150" in t1 and "h = 200" in t1
    and "Console · plat 200 × 10" in t1 and "V = 60,0 kN" in t1 and "◉ 61 %" in t1)
chk("encart de la gorge : titre, a = 5,0, σ⊥ / τ⊥ / τ∥, σ_c", "LA GORGE AU POINT GOUVERNANT" in t1 and "a = 5,0" in t1 and "σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)) = 218,5 N/mm²" in t1)
t2 = norm(texte(pdfs["C3"], 1))
chk("cartouche : cas, configuration, méthode, acier, date, échelle 1:n", "ÉCHELLEDUDESSIN" in serre(t2) and "1:" in t2 and "référence S235" in t2 and "Eurocode (ELU) · directionnelle" in t2)
t5 = norm(texte(pdfs["C5"]))
chk("C5 : vue de face et élévation, IPE 200 sur platine t 15, symboles a 5 (semelles) et a 4 (âme), M_y et V_z en élévation",
    "IPE 200 sur platine t 15" in t5 and "a 5" in t5 and "a 4" in t5 and "M_y = 30,00 kN·m" in t5 and "V_z = 60,0 kN" in t5
    and "DESSINS—vuedeface·élévation" in serre(t5))
t6 = norm(texte(pdfs["C6 CHS par tronçons"]))
chk("C6 par tronçons : quatre tronçons dans les cordons, message §4.3.2.2 en hypothèse", "pourtour · tronçon 4" in t6 and "§4.3.2.2" in t6 and "espacement" in t6)
t7 = norm(texte(pdfs["C7"]))
chk("C7 pleine pénétration : la pièce gouverne, coupe A-A de la soudure (chanfrein en X)", "la pièce gouverne" in t7 and "pleine pénétration" in t7 and "chanfrein en X" in t7)
t8 = norm(texte(pdfs["C8 cercle + segment"]))
chk("C8 : arc par tronçons et segment nommés sur le dessin avec leur gorge, bloc des efforts", "arc 1 · tronçon 1 · a 4" in t8 and "cordon 2 · a 5" in t8 and "V_y = " in t8)

# ================================================================
print("\n=== 4. Page 2 : méthode et détail ===")
chk("MÉTHODE ET DÉTAIL, références", "MÉTHODE ET DÉTAIL" in t2 and "EN 1993-1-8" in t2 and "ANB" in t2 and "LECORDONETSAGORGE" in serre(t2))
gardees = [sec for sec in ME.SECTIONS if not sec["titre"].startswith("Méthode ELS")]       # note Eurocode : la règle du bureau n'est pas imprimée
frag_ok = all(norm(sec["paragraphes"][0])[:60] in t2 for sec in gardees)
chk("un fragment de chaque section de methode.py de la méthode choisie, mot pour mot (source unique) ; la section ELS absente d'une note Eurocode",
    frag_ok and len(gardees) == len(ME.SECTIONS) - 1 and norm(next(s for s in ME.SECTIONS if s["titre"].startswith("Méthode ELS"))["paragraphes"][0])[:60] not in t2)
chk("carte des taux : cordon droit et gauche avec a5 · 190 mm et 61 %", "CARTEDESTAUX" in serre(t2) and "cordon droit · a5 · 190 mm" in t2 and "61 %" in t2)
t8b = norm(texte(pdfs["C5 HEA 200 six efforts"], 1))
chk("HEA 200 : 24 points ne tiennent pas en page 1 → le pire point de chacun des 8 cordons, détail des 24 en page 2",
    "le pire point de chaque cordon (24 points calculés) — détail en page 2" in norm(texte(pdfs["C5 HEA 200 six efforts"])) and "VÉRIFICATIONSPARPOINT" in serre(t8b))
t6b = norm(texte(pdfs["C6 CHS par tronçons"], 1))
chk("CHS par tronçons : 48 points de calcul → page 1 le pire de chacun des 4 tronçons, page 2 idem (pas 48 lignes)",
    "le pire point de chaque cordon (48 points calculés)" in t6 and "le pire point de chacun des 4 cordons (48 points calculés)" in t6b)

# ================================================================
print("\n=== 5. Méthode ELS ===")
te = norm(texte(pdfs["C3 ELS"]))
chk("ELS : « efforts de service », k = 100 kg/(mm·cm) = 98,1 N/mm², équivalent Eurocode 138,6 = 141, console à 104 % NON CONFORME (V_ser 40 kN, sans retours)",
    "efforts de service" in te and "98,1 N/mm²" in te and "138,6 N/mm² = 141 kg/(mm·cm)" in te and "NON CONFORME" in te and "η max 104 %" in te)
chk("ELS : la vérification écrit la règle du bureau et son équivalent (τ_adm = k·g/10 = 100 × 9,81/10 = 98,1, F_adm = 98,1 × 5 = 491 N/mm, k équivalent 141) ; "
    "page 2 sans le texte Eurocode",
    "τ_adm = k·g/10 = 100 × 9,81/10 = 98,1 N/mm²" in te and "F_adm = τ_adm·a = 98,1 × 5 = 491 N/mm" in te and "= 141 kg/(mm·cm)" in te
    and "Méthode ELS du bureau (efforts de service)" in te and "EUROCODE:EN1993-1-8§4.5.3" not in serre(norm(texte(pdfs["C3 ELS"], 1)))
    and "MÉTHODEELSDUBUREAU" in serre(norm(texte(pdfs["C3 ELS"], 1))))
te_r = norm(texte(NT.generer_note(ET_, cas_de("C3", {"retours": True}, {"V": 40}, {"mode": "els"}))))
chk("ELS avec retours d'angle (l'exemple de l'étude) : 94 %, CONFORME · ATTENTION", "η max 94 %" in te_r and "ATTENTION" in serre(te_r))

# ================================================================
print("\n=== 6. Le peintre partagé ===")
pts = SV._chemin("M10,10 L20,10 A5,5 0 1 1 20,20 Z M0,0 L1,1")
chk("path : M L A Z puis un second sous-chemin ; l'arc est échantillonné (≥ 6 points), premier sous-chemin fermé",
    len(pts) == 2 and pts[0][1] is True and len(pts[0][0]) >= 8 and pts[1][1] is False)
cercle = SV._chemin("M50,0 A50,50 0 1 1 -50,0 A50,50 0 1 1 50,0 Z")[0][0]
chk("un cercle en deux arcs devient ≥ 24 points, tous à 50 ± 0,5 du centre",
    len(cercle) >= 24 and all(abs((x * x + y * y) ** 0.5 - 50) < 0.5 for x, y in cercle))
buf = io.BytesIO()
c = rl_canvas.Canvas(buf, pagesize=landscape(A4))
svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="h" patternUnits="userSpaceOnUse" width="7" height="7" '
       'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="#6B7280" stroke-width="0.7"/></pattern>'
       '<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="90" x2="100" y2="90"><stop offset="0" stop-color="#2E7D46"/>'
       '<stop offset="1" stop-color="#9C3341"/></linearGradient></defs>'
       '<polygon points="10,10 60,10 60,40 10,40" fill="#E2E8F0"/><polygon points="10,10 60,10 60,40 10,40" fill="url(#h)"/>'
       '<path d="M70,10 L95,10 L95,40 L70,40 Z M78,18 L88,18 L88,32 L78,32 Z" fill="#F1F5F9" fill-rule="evenodd" stroke="#111"/>'
       '<line x1="0" y1="90" x2="100" y2="90" stroke="url(#g)" stroke-width="6"/>'
       '<text x="50" y="60" dy="10" font-size="8" text-anchor="middle">centre</text></svg>')
n0 = len(c._code)
SV.peindre_svg(c, svg, 20, 20, 300, 300)
code = "\n".join(c._code[n0:])
chk("hachures : des lignes sont tracées sous une découpe (W n … l … S), dégradé : 8 tronçons de couleurs distinctes, texte avec dy",
    code.count(" l") >= 20 and " W n" in code and code.count(" RG") >= 8 and "centre" in code)
from plancher.charges import peintre as PE  # noqa: E402
chk("le peintre du plancher est un alias du peintre partagé (Mat, transformation) avec ses motifs en aplats",
    PE.Mat is SV.Mat and PE.transformation is SV.transformation and "url(#chp_hc)" in PE.MOTIFS)

# ================================================================
print("\n=== 7. Lot A, deuxième passe : méthode ELS calculée (S1), provenance de f_u (S8), hypothèse d'épaisseur (S3) ===")
import contextlib  # noqa: E402
import importlib  # noqa: E402
import subprocess  # noqa: E402

from acier import materiaux as MA  # noqa: E402
from acier.bibliotheques import DB, find  # noqa: E402
from acier.soudures import criteres as K  # noqa: E402

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def mutant(obj, nom, valeur):
    """Une mutation : ``obj.nom`` remplacé le temps d'un contrôle, puis remis."""
    ancien = getattr(obj, nom)
    setattr(obj, nom, valeur)
    try:
        yield
    finally:
        setattr(obj, nom, ancien)


def pdftotext(pdf: bytes) -> str:
    p = subprocess.run(["pdftotext", "-layout", "-", "-"], input=pdf, capture_output=True, check=True)
    return norm(p.stdout.decode("utf-8"))


def attendus_els():
    """Le calcul, refait ici : f_vw,d/γ_F avec f_u de la table (3 ≤ t ≤ 100 mm, tableau 3.3 ANB) et β_w (tableau 4.1),
    la marge du bureau 1 − 98,1/τ (g = 9,81). S235 138,6 ; S275 148,5 ; S355 160,8 ; marges 29 % et 39 %."""
    tau = {n: K.k_equivalent_eurocode(MA.fu(n, 10.0), find(DB["aciers"], n)["bw"])["tau_N_mm2"] for n in ("S235", "S275", "S355")}
    fr = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731
    marge = {n: f"{(1 - K.tau_adm_bureau(K.K_BUREAU) / tau[n]) * 100:.0f} %" for n in tau}
    return [f"{fr(tau['S235'])} N/mm² pour le S235", f"{fr(tau['S275'])} pour le S275", f"{fr(tau['S355'])} pour le S355",
            f"plus sûre de {marge['S235']} ; sur du S355, de {marge['S355']}"]


def ctl_methode_els() -> bool:
    """S1 : la page 2 d'une note ELS imprime les équivalents Eurocode CALCULÉS depuis la table (148,5 et 160,8, marge
    39 % en S355), jamais les anciens nombres du tableau 3.1 de l'EN (155,8 ; 174,5 ; 43 %)."""
    t2 = pdftotext(NT.generer_note(ET_, cas_de("C3", None, {"V": 40}, {"mode": "els"})))
    att = attendus_els()
    return all(a in t2 for a in att) and "174,5" not in t2 and "155,8" not in t2 and att[2] == "160,8 pour le S355" and att[1] == "148,5 pour le S275"


def ctl_doc_etude() -> bool:
    """La copie de docs/acier/SOUDURES_ETUDE.md (un document ne lit pas le code : copie NOMMÉE, liée ici) dit les mêmes nombres."""
    with open(os.path.join(RACINE, "docs", "acier", "SOUDURES_ETUDE.md"), encoding="utf-8") as f:
        doc = norm(f.read())
    att = attendus_els()
    return (att[1] in doc and att[2] in doc and "174,5" not in doc and "155,8" not in doc
            and "plus sûre de 29 %** sur du S235" in doc and "39 % de marge" in doc)


def ctl_provenance_fu() -> bool:
    """S8 : la note dit d'où vient la f_u de référence (la pièce, son épaisseur, la bande, le tableau), comme l'écran ;
    S3 : sur un support « plat » sans épaisseur, l'hypothèse t_A = t est écrite (pdftotext)."""
    t = pdftotext(NT.generer_note(ET_, cas_de("C3")))
    tp = pdftotext(NT.generer_note(ET_, cas_de("C3", {"poteau": "plat"})))
    return ("Acier de référence S235 (la pièce la plus faible) : fu = 360 N/mm² (pièce B, S235 de 10 mm : 3 ≤ t ≤ 100 mm, tableau 3.3 ANB)"
            in t and "plat support" not in t
            and "pièce A (plat support) : épaisseur non saisie, prise égale à t = 10 mm" in tp)


chk("S1 · note ELS, page 2 : équivalents Eurocode calculés depuis la table (138,6 ; 148,5 ; 160,8 ; marges 29 % et 39 %)", ctl_methode_els())
chk("S1 · la copie de la doc d'étude (SOUDURES_ETUDE.md) dit les mêmes nombres (copie nommée, liée par ce contrôle)", ctl_doc_etude())
chk("S8, S3 · note : provenance de f_u (pièce, épaisseur, bande, tableau 3.3 ANB) et hypothèse d'épaisseur du plat support", ctl_provenance_fu())
F_U_EN = {"S235": 360.0, "S275": 430.0, "S355": 510.0}         # le tableau 3.1 de l'EN (le défaut d'avant)
with mutant(MA, "fu", lambda n, t, produit="lamine": F_U_EN[n]):
    importlib.reload(ME)
    rouge = not ctl_methode_els()
importlib.reload(ME)
chk("mutation « f_u du tableau 3.1 de l'EN » (le texte calculé redonne 155,8 et 174,5) : le contrôle de la méthode ELS rougit",
    rouge and ctl_methode_els())
with mutant(NT.ET, "texte_fu", lambda ac: "—"):
    chk("mutation « provenance de f_u absente » : le contrôle de la note rougit", not ctl_provenance_fu())
with mutant(NT.ET, "texte_hypothese", lambda ac: None):
    chk("mutation « hypothèse d'épaisseur tue » : le contrôle de la note rougit", not ctl_provenance_fu())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
