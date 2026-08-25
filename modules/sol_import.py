# -*- coding: utf-8 -*-
# =============================================================
#  sol_import.py — Import de sondages CPT SANS intelligence artificielle
#  VERSION 1.0
#
#  Remplace l'import PDF par IA de raideur_sol v4.0 : plus de clé API,
#  plus d'appel réseau, plus de valeur « estimée à l'œil ». Tout est lu
#  directement dans le fichier, de façon déterministe et reproductible.
#
#  Quatre sources, de la plus fidèle à la plus dégradée :
#    1. GEF   (.gef)          — format d'échange géotechnique BE/NL,
#                               valeurs numériques au pas réel (2 cm).
#    2. CSV / TXT             — colonnes profondeur / qc / fs.
#    3. PDF à courbes vectorielles — les polylignes du dessin SONT les
#       données ; on les relit et on les recale sur les traits de grille
#       du cadre. Précision mesurée : écart médian 0,0005 % sur un
#       rapport de contrôle (voir tests/test_sol_import.py).
#    4. PDF à tableau de valeurs — extraction du tableau texte.
#
#  Le PDF raster (scan, photo) n'est PAS traité : il n'y a rien à y lire
#  de fiable sans calibrage manuel. L'outil le détecte et le dit.
#
#  Aucune dépendance à Streamlit : ce module est testable seul.
# =============================================================

import csv
import io
import math
import re

try:
    import pymupdf as _fitz
    _HAS_PDF = True
except ImportError:                       # pragma: no cover
    try:
        import fitz as _fitz
        _HAS_PDF = True
    except ImportError:
        _HAS_PDF = False


# =============================================================
#  STRUCTURE DE SORTIE COMMUNE
#
#  Un sondage importé :
#    {"nom": str,
#     "points": [(z_m, qc_MPa, fs_kPa|None), ...],   trié par z croissant
#     "source": str,          libellé lisible de la provenance
#     "nappe_m": float|None,
#     "avertissements": [str, ...],
#     "calibration": dict|None}   pour le PDF vectoriel, permet à
#                                 l'utilisateur de corriger le recalage
# =============================================================
def _sondage(nom, points, source, nappe=None, avert=None, calib=None):
    pts = sorted(((float(z), float(q), (None if f is None else float(f)))
                  for (z, q, f) in points if z is not None and q is not None),
                 key=lambda t: t[0])
    return {"nom": str(nom), "points": pts, "source": str(source),
            "nappe_m": nappe, "avertissements": list(avert or []),
            "calibration": calib}


# =============================================================
#  1. GEF — Geotechnical Exchange Format
#
#  En-tête de lignes #MOT= valeurs, puis #EOH= et les données.
#  Les colonnes sont décrites par #COLUMNINFO= indice, unité, nom, code
#  où le CODE de quantité est normalisé :
#      1  = longueur de pénétration (m)
#      2  = résistance de cône qc (MPa)
#      3  = frottement latéral fs (MPa)
#      4  = rapport de frottement Rf (%)
#      11 = profondeur corrigée de l'inclinaison (m)
# =============================================================
_GEF_QTY = {1: "z_brut", 2: "qc", 3: "fs", 4: "rf", 11: "z_corr"}


def parse_gef(contenu: bytes, nom_fichier: str = "CPT"):
    """Lit un fichier GEF. Lève ValueError si le format n'est pas reconnu."""
    try:
        txt = contenu.decode("utf-8", errors="replace")
    except Exception:
        txt = contenu.decode("latin-1", errors="replace")
    if "#EOH" not in txt.upper():
        raise ValueError("Fichier GEF invalide : marqueur #EOH absent.")

    haut, bas = re.split(r"#EOH\s*=?\s*", txt, maxsplit=1, flags=re.IGNORECASE)

    sep_col, sep_rec, nom, nappe = None, None, None, None
    vides = {}
    colonnes = {}
    for ligne in haut.splitlines():
        m = re.match(r"\s*#\s*([A-Za-z]+)\s*=\s*(.*)", ligne)
        if not m:
            continue
        mot, val = m.group(1).upper(), m.group(2).strip()
        if mot == "COLUMNSEPARATOR":
            sep_col = val[:1] or None
        elif mot == "RECORDSEPARATOR":
            sep_rec = val[:1] or None
        elif mot == "COLUMNINFO":
            p = [x.strip() for x in val.split(",")]
            if len(p) >= 4:
                try:
                    colonnes[int(p[0])] = {"unite": p[1], "nom": p[2], "code": int(float(p[3]))}
                except ValueError:
                    pass
        elif mot == "COLUMNVOID":
            p = [x.strip() for x in val.split(",")]
            if len(p) >= 2:
                try:
                    vides[int(p[0])] = float(p[1])
                except ValueError:
                    pass
        elif mot in ("TESTID", "PROJECTID") and not nom:
            nom = val.split(",")[-1].strip() or None
        elif mot == "MEASUREMENTVAR":
            p = [x.strip() for x in val.split(",")]
            # var 14 : niveau de la nappe (convention courante)
            if len(p) >= 2 and p[0] == "14":
                try:
                    nappe = abs(float(p[1]))
                except ValueError:
                    pass

    # index de colonne (1-based) -> grandeur
    idx = {}
    for i, info in colonnes.items():
        q = _GEF_QTY.get(info["code"])
        if q:
            idx[q] = i
    if "qc" not in idx:
        raise ValueError("Fichier GEF sans colonne de résistance de cône (code 2).")
    i_z = idx.get("z_corr") or idx.get("z_brut")
    if not i_z:
        raise ValueError("Fichier GEF sans colonne de profondeur (code 1 ou 11).")

    corps = bas
    if sep_rec:
        corps = corps.replace(sep_rec, "\n")
    points = []
    for ligne in corps.splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        champs = re.split(r"[;,\s]+", ligne) if not sep_col else \
            [c for c in ligne.split(sep_col) if c != ""]
        if len(champs) < max(i_z, idx["qc"]):
            continue
        try:
            z = float(champs[i_z - 1])
            qc = float(champs[idx["qc"] - 1])
        except (ValueError, IndexError):
            continue
        if vides.get(i_z) is not None and abs(z - vides[i_z]) < 1e-9:
            continue
        if vides.get(idx["qc"]) is not None and abs(qc - vides[idx["qc"]]) < 1e-9:
            continue
        fs = None
        if "fs" in idx:
            try:
                v = float(champs[idx["fs"] - 1])
                if vides.get(idx["fs"]) is None or abs(v - vides[idx["fs"]]) > 1e-9:
                    fs = v * 1000.0        # MPa -> kPa
            except (ValueError, IndexError):
                fs = None
        if fs is None and "rf" in idx:
            try:
                rf = float(champs[idx["rf"] - 1])
                if rf > 0:
                    fs = rf / 100.0 * qc * 1000.0
            except (ValueError, IndexError):
                pass
        z = abs(z)
        if qc >= 0:
            points.append((z, qc, fs))

    if len(points) < 5:
        raise ValueError("Fichier GEF lu mais moins de 5 points exploitables.")

    return _sondage(nom or nom_fichier, points,
                    f"GEF — {len(points)} points au pas réel", nappe=nappe)


# =============================================================
#  2. CSV / TXT
# =============================================================
_ENTETES = {
    "z": ("profondeur", "depth", "diepte", "z", "sondeerlengte", "prof"),
    "qc": ("qc", "conusweerstand", "cone", "resistance de pointe",
           "résistance de pointe", "qcone", "qt"),
    "fs": ("fs", "wrijving", "frottement", "friction", "flocal"),
    "rf": ("rf", "friction ratio", "rapport de frottement", "wrijvingsgetal"),
}


def _devine_colonnes(entete):
    """Associe chaque grandeur à un indice de colonne d'après l'en-tête."""
    trouve = {}
    for i, cell in enumerate(entete):
        c = str(cell).strip().lower()
        c = re.sub(r"[\[\(].*?[\]\)]", "", c).strip()      # retire (m), [MPa]
        for grandeur, cles in _ENTETES.items():
            if grandeur in trouve:
                continue
            if any(c == k or c.startswith(k) for k in cles):
                trouve[grandeur] = i
    return trouve


def parse_csv(contenu: bytes, nom_fichier: str = "CPT", mapping=None,
              qc_unite="MPa", fs_unite="kPa"):
    """
    Lit un CSV/TXT de sondage. `mapping` permet de forcer les colonnes :
    {"z": 0, "qc": 1, "fs": 2}. Sinon détection par l'en-tête.
    """
    txt = contenu.decode("utf-8", errors="replace")
    if not txt.strip():
        raise ValueError("Fichier vide.")

    # Détection du séparateur PILOTÉE PAR LE RÉSULTAT plutôt que par un
    # comptage de caractères : dans un CSV francophone, les virgules
    # décimales sont plus nombreuses que les vrais séparateurs ';' et
    # faussent toute heuristique de fréquence.
    def _decoupe(sep):
        ls = [l for l in csv.reader(io.StringIO(txt), delimiter=sep)
              if any(str(c).strip() for c in l)]
        return ls

    lignes, dialecte, debut = None, None, 0
    if mapping is None:
        meilleur = None
        for sep in (";", "\t", ",", "|"):
            ls = _decoupe(sep)
            if not ls or max(len(l) for l in ls) < 2:
                continue
            for i, l in enumerate(ls[:15]):
                m = _devine_colonnes(l)
                if "z" in m and "qc" in m:
                    score = len(m)          # plus de colonnes reconnues = mieux
                    if meilleur is None or score > meilleur[0]:
                        meilleur = (score, ls, sep, m, i + 1)
                    break
        if meilleur is None:
            raise ValueError(
                "Colonnes non reconnues. Attendu un en-tête contenant au moins "
                "une colonne de profondeur et une colonne qc "
                "(ex. « Profondeur (m) ; qc (MPa) ; fs (kPa) »).")
        _, lignes, dialecte, mapping, debut = meilleur
    else:
        # mapping imposé : on prend le séparateur qui produit assez de colonnes
        besoin = max(mapping.values()) + 1
        for sep in (";", "\t", ",", "|"):
            ls = _decoupe(sep)
            if ls and max(len(l) for l in ls) >= besoin:
                lignes, dialecte = ls, sep
                break
        if lignes is None:
            raise ValueError("Séparateur de colonnes non identifié.")
        # en-tête éventuel à sauter
        try:
            float(str(lignes[0][mapping["z"]]).replace(",", "."))
        except (ValueError, IndexError):
            debut = 1

    fqc = 1.0 if qc_unite == "MPa" else 0.001          # kPa -> MPa
    ffs = 1.0 if fs_unite == "kPa" else 1000.0         # MPa -> kPa

    points, ignorees = [], 0
    for l in lignes[debut:]:
        try:
            z = abs(float(str(l[mapping["z"]]).replace(",", ".")))
            qc = float(str(l[mapping["qc"]]).replace(",", ".")) * fqc
        except (ValueError, IndexError):
            ignorees += 1
            continue
        fs = None
        if "fs" in mapping:
            try:
                fs = float(str(l[mapping["fs"]]).replace(",", ".")) * ffs
            except (ValueError, IndexError):
                fs = None
        if fs is None and "rf" in mapping:
            try:
                rf = float(str(l[mapping["rf"]]).replace(",", "."))
                if rf > 0:
                    fs = rf / 100.0 * qc * 1000.0
            except (ValueError, IndexError):
                pass
        if qc >= 0:
            points.append((z, qc, fs))

    if len(points) < 5:
        raise ValueError("Moins de 5 lignes numériques exploitables.")
    avert = [f"{ignorees} ligne(s) non numérique(s) ignorée(s)."] if ignorees else []
    avert += _controler_profondeurs(points)
    return _sondage(nom_fichier, points, f"CSV — {len(points)} points", avert=avert)


def _controler_profondeurs(points):
    """
    Une colonne « Z » est aussi souvent une COTE (altitude, +12,50 m) qu'une
    profondeur. Prise pour une profondeur, elle décale tout le profil et
    fausse le calcul d'un ordre de grandeur, sans rien signaler. On teste
    donc ce que la série ressemble vraiment à une profondeur de sondage.
    """
    if not points:
        return []
    zs = sorted(p[0] for p in points)
    a = []
    if zs[0] > 3.0:
        a.append(
            f"La colonne de profondeur commence à {zs[0]:.2f} m : s'agit-il d'une "
            "COTE (altitude) et non d'une profondeur sous le terrain naturel ? "
            "Dans ce cas le profil est décalé et le calcul sera faux — convertis "
            "la colonne avant d'importer.")
    if zs[-1] > 100.0:
        a.append(f"Profondeur maximale lue : {zs[-1]:.1f} m — valeur inhabituelle "
                 "pour un CPT, vérifie l'unité de la colonne (cm au lieu de m ?).")
    ecarts = [b - a2 for a2, b in zip(zs, zs[1:]) if b > a2]
    if ecarts:
        med = sorted(ecarts)[len(ecarts) // 2]
        if med > 1.0:
            a.append(f"Pas de mesure médian de {med:.2f} m : très grossier pour un CPT "
                     "(le pas normalisé est de 2 cm). Vérifie la colonne choisie.")
    return a


# =============================================================
#  3. PDF À COURBES VECTORIELLES
#
#  Principe, et pourquoi il est exact :
#  dans un PDF vectoriel, la courbe qc n'est pas une image : c'est une
#  polyligne dont les sommets sont les points de mesure, aux coordonnées
#  près. Il suffit donc de (a) retrouver cette polyligne, (b) retrouver
#  le repère du graphique, (c) appliquer la transformation affine.
#
#  Le recalage se fait sur les TRAITS DE GRILLE, pas sur le centre des
#  étiquettes : une étiquette est dessinée avec un décalage typographique
#  de quelques dixièmes de point, ce qui suffit à décaler tout le profil.
#  Mesuré : recalage sur le texte -> 6,5 % d'écart moyen ;
#           recalage sur la grille -> 0,0005 %.
# =============================================================
_TOL = 2.0          # tolérance géométrique (points PDF)
_MIN_PTS_COURBE = 20


def _primitives(page):
    """Segments courts (grille, cadre) et polylignes longues (courbes)."""
    segments, polylignes = [], []
    for d in page.get_drawings():
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts.append((it[1].x, it[1].y)); pts.append((it[2].x, it[2].y))
            elif it[0] == "c":
                pts.append((it[1].x, it[1].y)); pts.append((it[4].x, it[4].y))
            elif it[0] == "re":
                r = it[1]
                pts += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
        if not pts:
            continue
        clean = [pts[0]]
        for q in pts[1:]:
            if abs(q[0] - clean[-1][0]) > 1e-6 or abs(q[1] - clean[-1][1]) > 1e-6:
                clean.append(q)
        rec = {"pts": clean, "n": len(clean), "couleur": d.get("color"),
               "xmin": min(x for x, _ in clean), "xmax": max(x for x, _ in clean),
               "ymin": min(y for _, y in clean), "ymax": max(y for _, y in clean)}
        (polylignes if len(clean) >= _MIN_PTS_COURBE else segments).append(rec)
    polylignes.sort(key=lambda p: -p["n"])
    return segments, polylignes


def _cadre(segments, polylignes):
    """Le plus grand rectangle plausible englobant les courbes."""
    cands = [s for s in segments
             if (s["xmax"] - s["xmin"]) > 40 and (s["ymax"] - s["ymin"]) > 80]
    if cands:
        c = max(cands, key=lambda s: (s["xmax"] - s["xmin"]) * (s["ymax"] - s["ymin"]))
        return c["xmin"], c["xmax"], c["ymin"], c["ymax"]
    if polylignes:                     # repli : l'emprise des courbes
        c = polylignes[0]
        return c["xmin"], c["xmax"], c["ymin"], c["ymax"]
    raise ValueError("Aucun cadre de graphique détecté.")


def _grilles(segments, X0, X1, Y0, Y1):
    v, h = set(), set()
    for s in segments:
        dx, dy = s["xmax"] - s["xmin"], s["ymax"] - s["ymin"]
        dedans = (X0 - _TOL <= s["xmin"] and s["xmax"] <= X1 + _TOL and
                  Y0 - _TOL <= s["ymin"] and s["ymax"] <= Y1 + _TOL)
        if not dedans:
            continue
        if dx < _TOL and dy > (Y1 - Y0) * 0.8:
            v.add(round((s["xmin"] + s["xmax"]) / 2, 3))
        elif dy < _TOL and dx > (X1 - X0) * 0.8:
            h.add(round((s["ymin"] + s["ymax"]) / 2, 3))
    return sorted(v), sorted(h)


def _etiquettes(page):
    out = []
    for w in page.get_text("words"):
        t = w[4].strip().replace(",", ".")
        if not re.fullmatch(r"-?\d+(\.\d+)?", t):
            continue
        out.append({"v": float(t), "cx": (w[0] + w[2]) / 2, "cy": (w[1] + w[3]) / 2})
    return out


def _apparier(traits, etqs, axe, X0, X1, Y0, Y1, marge=28.0):
    paires = []
    for t in traits:
        best, bd = None, 1e9
        for n in etqs:
            if axe == "x":
                dehors = not (n["cy"] < Y0 or n["cy"] > Y1)
                if dehors or abs(n["cy"] - (Y0 if n["cy"] < Y0 else Y1)) > marge:
                    continue
                d = abs(n["cx"] - t)
            else:
                dehors = not (n["cx"] < X0 or n["cx"] > X1)
                if dehors or abs(n["cx"] - (X0 if n["cx"] < X0 else X1)) > marge:
                    continue
                d = abs(n["cy"] - t)
            if d < marge and d < bd:
                best, bd = n, d
        if best is not None:
            paires.append((t, best["v"]))
    # une même valeur ne peut pas être portée par deux traits
    vus, propre = set(), []
    for pos, val in paires:
        if val not in vus:
            vus.add(val); propre.append((pos, val))
    return propre


def _affine(paires):
    """Moindres carrés pixel -> valeur. Retourne (a, b, R²)."""
    n = len(paires)
    if n < 2:
        raise ValueError("Recalage impossible : moins de deux repères d'axe.")
    sx = sum(p[0] for p in paires); sy = sum(p[1] for p in paires)
    sxx = sum(p[0] ** 2 for p in paires); sxy = sum(p[0] * p[1] for p in paires)
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        raise ValueError("Recalage impossible : repères d'axe confondus.")
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    moy = sy / n
    st = sum((p[1] - moy) ** 2 for p in paires)
    sr = sum((p[1] - (a * p[0] + b)) ** 2 for p in paires)
    return a, b, (1.0 - sr / st if st > 1e-12 else 1.0)


def analyser_pdf(contenu: bytes, page_idx: int = 0):
    """
    Analyse une page de PDF et retourne ce qui a été détecté, SANS encore
    décider : courbes candidates et recalage proposé. L'appelant (interface)
    affiche le tout et laisse l'ingénieur confirmer ou corriger.
    """
    if not _HAS_PDF:
        raise RuntimeError(
            "La lecture de PDF nécessite PyMuPDF (pip install pymupdf).")
    doc = _fitz.open(stream=contenu, filetype="pdf")
    if page_idx >= len(doc):
        raise ValueError(f"Le PDF ne contient que {len(doc)} page(s).")
    page = doc[page_idx]

    segments, polylignes = _primitives(page)
    if not polylignes:
        raise ValueError(
            "Aucune courbe vectorielle sur cette page. Si le rapport est un "
            "scan ou une photo, les données ne sont pas lisibles : demande le "
            "fichier GEF ou le PDF d'origine au bureau d'essais.")

    X0, X1, Y0, Y1 = _cadre(segments, polylignes)
    gv, gh = _grilles(segments, X0, X1, Y0, Y1)
    etqs = _etiquettes(page)

    calib = {"x0": X0, "x1": X1, "y0": Y0, "y1": Y1,
             "n_grille_v": len(gv), "n_grille_h": len(gh)}
    avert = []

    px = _apparier(gv, etqs, "x", X0, X1, Y0, Y1)
    py = _apparier(gh, etqs, "y", X0, X1, Y0, Y1)

    try:
        ax, bx, r2x = _affine(px)
        calib.update({"ax": ax, "bx": bx, "r2x": r2x, "auto_x": True,
                      "reperes_x": px})
    except ValueError:
        calib.update({"ax": None, "bx": None, "r2x": 0.0, "auto_x": False,
                      "reperes_x": px})
        avert.append("Axe des qc non recalé automatiquement : à saisir à la main.")
    try:
        ay, by, r2y = _affine(py)
        calib.update({"ay": ay, "by": by, "r2y": r2y, "auto_y": True,
                      "reperes_y": py})
    except ValueError:
        calib.update({"ay": None, "by": None, "r2y": 0.0, "auto_y": False,
                      "reperes_y": py})
        avert.append("Axe des profondeurs non recalé automatiquement : à saisir à la main.")

    for cle, lbl in (("r2x", "qc"), ("r2y", "profondeur")):
        if calib.get(cle) is not None and 0 < calib[cle] < 0.9995:
            avert.append(
                f"Repères de l'axe {lbl} mal alignés (R² = {calib[cle]:.4f}) : "
                "vérifie le recalage avant d'importer.")

    courbes = []
    for i, pl in enumerate(polylignes[:6]):
        courbes.append({
            "idx": i, "n": pl["n"], "couleur": pl["couleur"],
            "etendue_x": (pl["xmin"], pl["xmax"]),
            "etendue_y": (pl["ymin"], pl["ymax"]),
            "amplitude_x": pl["xmax"] - pl["xmin"],
            "pts": pl["pts"],
        })
    # la courbe qc est la plus « large » horizontalement (elle balaie l'axe)
    courbes.sort(key=lambda c: -c["amplitude_x"])
    calib["n_pages"] = len(doc)
    doc.close()
    return {"courbes": courbes, "calibration": calib, "avertissements": avert}


# Plage physique du rapport de frottement Rf = fs/qc. En dehors, ce n'est
# pas un sol : c'est un recalage faux. Sert de garde-fou automatique.
RF_MIN_PLAUSIBLE = 0.05      # %
RF_MAX_PLAUSIBLE = 12.0      # %


def extraire_courbe(analyse, idx_qc=0, idx_fs=None, calib=None, nom="CPT",
                    z_min=None, z_max=None, calib_fs=None, facteur_fs=1.0):
    """
    Applique le recalage aux polylignes choisies et produit un sondage.

    `calib`     : recalage de l'axe des qc (ax, bx) et des profondeurs (ay, by).
    `calib_fs`  : recalage PROPRE à la courbe de frottement. Dans un rapport
                  réel, fs n'est presque jamais tracé à la même échelle que
                  qc — soit sur un axe séparé, soit sur le même axe avec un
                  facteur (« fs × 10 »). Utiliser la calibration de qc pour
                  fs fausse Rf, donc l'indice Ic, donc tout le classement.
    `facteur_fs`: raccourci quand fs partage l'axe de qc avec un facteur
                  d'agrandissement (valeur lue sur le graphique divisée par
                  ce facteur).

    Le rapport de frottement obtenu est contrôlé : hors de la plage
    physique, un avertissement explicite est renvoyé plutôt qu'un profil
    silencieusement faux.
    """
    calib = calib or analyse["calibration"]
    for k in ("ax", "bx", "ay", "by"):
        if calib.get(k) is None:
            raise ValueError(
                "Recalage incomplet : renseigne les valeurs des axes avant d'importer.")
    ax, bx, ay, by = calib["ax"], calib["bx"], calib["ay"], calib["by"]
    cfs = calib_fs or calib
    ax_fs = cfs.get("ax", ax)
    bx_fs = cfs.get("bx", bx)
    try:
        facteur_fs = float(facteur_fs) or 1.0
    except Exception:
        facteur_fs = 1.0

    courbes = analyse["courbes"]
    if idx_qc >= len(courbes):
        raise ValueError("Courbe qc introuvable.")

    def _serie(c, a, b):
        return sorted(((ay * y + by, a * x + b) for (x, y) in c["pts"]),
                      key=lambda t: t[0])

    sq = _serie(courbes[idx_qc], ax, bx)
    sf = (_serie(courbes[idx_fs], ax_fs, bx_fs)
          if (idx_fs is not None and 0 <= idx_fs < len(courbes)) else None)

    def interp(serie, z):
        if not serie:
            return None
        if z <= serie[0][0]:
            return serie[0][1]
        if z >= serie[-1][0]:
            return serie[-1][1]
        lo, hi = 0, len(serie) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if serie[mid][0] <= z:
                lo = mid
            else:
                hi = mid
        z0, q0 = serie[lo]; z1, q1 = serie[hi]
        return q0 if z1 == z0 else q0 + (q1 - q0) * (z - z0) / (z1 - z0)

    points, avert = [], list(analyse.get("avertissements", []))
    neg = 0
    for (z, q) in sq:
        if z_min is not None and z < z_min:
            continue
        if z_max is not None and z > z_max:
            continue
        if q < 0:
            neg += 1
            q = 0.0
        fs = None
        if sf:
            v = interp(sf, z)
            # la courbe de frottement est lue en MPa sur SON axe, puis
            # ramenée à l'échelle réelle et convertie en kPa
            fs = None if v is None else max(0.0, v) / facteur_fs * 1000.0
        points.append((max(0.0, z), q, fs))
    if neg:
        avert.append(f"{neg} valeur(s) de qc négative(s) ramenée(s) à zéro "
                     "(dépassement de cadre à la lecture).")
    if len(points) < 5:
        raise ValueError("Moins de 5 points exploitables après recalage.")

    # ---- garde-fou : le rapport de frottement doit rester physique ----
    rfs = [p[2] / (p[1] * 1000.0) * 100.0
           for p in points if p[2] is not None and p[1] > 0.05]
    rf_med = None
    if rfs:
        rfs_tri = sorted(rfs)
        rf_med = rfs_tri[len(rfs_tri) // 2]
        if not (RF_MIN_PLAUSIBLE <= rf_med <= RF_MAX_PLAUSIBLE):
            avert.append(
                f"Rapport de frottement médian Rf = {rf_med:.1f} % — hors de la plage "
                f"physique ({RF_MIN_PLAUSIBLE:g} à {RF_MAX_PLAUSIBLE:g} %). "
                "La courbe de frottement n'est presque jamais tracée à la même échelle "
                "que qc : indique son échelle propre (ou son facteur d'agrandissement), "
                "sinon la classification du sol sera fausse.")

    s = _sondage(nom, points,
                 f"PDF vectoriel — {len(points)} points recalés sur la grille",
                 avert=avert, calib=calib)
    s["rf_median"] = rf_med
    return s


# =============================================================
#  4. PDF À TABLEAU DE VALEURS
# =============================================================
def extraire_tableau_pdf(contenu: bytes, page_idx: int = 0, nom="CPT"):
    """Cherche un tableau de valeurs (profondeur / qc / fs) dans le texte."""
    if not _HAS_PDF:
        raise RuntimeError("La lecture de PDF nécessite PyMuPDF (pip install pymupdf).")
    doc = _fitz.open(stream=contenu, filetype="pdf")
    if page_idx >= len(doc):
        raise ValueError(f"Le PDF ne contient que {len(doc)} page(s).")
    txt = doc[page_idx].get_text("text")
    doc.close()

    points = []
    for ligne in txt.splitlines():
        champs = re.findall(r"-?\d+(?:[.,]\d+)?", ligne)
        if len(champs) < 2:
            continue
        try:
            vals = [float(c.replace(",", ".")) for c in champs[:4]]
        except ValueError:
            continue
        z, qc = abs(vals[0]), vals[1]
        if not (0 <= z <= 100) or not (0 <= qc <= 100):
            continue
        fs = None
        if len(vals) >= 3 and 0 <= vals[2] <= 5:
            fs = vals[2] * 1000.0
        points.append((z, qc, fs))

    if len(points) < 10:
        raise ValueError(
            "Aucun tableau de valeurs exploitable sur cette page "
            f"({len(points)} ligne(s) trouvée(s)).")
    # un tableau réel a des profondeurs croissantes et régulières
    zs = [p[0] for p in sorted(points)]
    pas = [b - a for a, b in zip(zs, zs[1:]) if b > a]
    if pas:
        med = sorted(pas)[len(pas) // 2]
        if med <= 0 or med > 2.0:
            raise ValueError("Les profondeurs lues ne forment pas une série régulière.")
    return _sondage(nom, points, f"PDF (tableau) — {len(points)} lignes")


# =============================================================
#  5. AIGUILLAGE ET EXPORT
# =============================================================
def importer(contenu: bytes, nom_fichier: str, **kw):
    """Choisit l'analyseur d'après l'extension. Pour un PDF vectoriel,
    utilise plutôt analyser_pdf() + extraire_courbe() (l'ingénieur doit
    confirmer le recalage)."""
    bas = (nom_fichier or "").lower()
    base = re.sub(r"\.[^.]+$", "", nom_fichier or "CPT") or "CPT"
    if bas.endswith(".gef"):
        return parse_gef(contenu, base)
    if bas.endswith((".csv", ".txt", ".asc")):
        return parse_csv(contenu, base, **kw)
    if bas.endswith(".pdf"):
        try:
            return extraire_tableau_pdf(contenu, nom=base)
        except Exception:
            an = analyser_pdf(contenu)
            return extraire_courbe(an, nom=base)
    raise ValueError(f"Extension non reconnue : {nom_fichier}")


def vers_csv(sondages) -> str:
    """Export CSV de un ou plusieurs sondages (données brutes)."""
    if isinstance(sondages, dict):
        sondages = [sondages]
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(["sondage", "profondeur_m", "qc_MPa", "fs_kPa", "Rf_pct"])
    for s in sondages:
        for (z, qc, fs) in s["points"]:
            rf = ""
            if fs is not None and qc > 0:
                rf = f"{fs / (qc * 1000.0) * 100.0:.3f}"
            w.writerow([s["nom"], f"{z:.3f}", f"{qc:.4f}",
                        "" if fs is None else f"{fs:.2f}", rf])
    return buf.getvalue()


def couches_vers_csv(nom, couches) -> str:
    """Export CSV du profil de couches interprété."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(["sondage", "de_m", "a_m", "epaisseur_m", "type",
                "Ic", "qc_moy_MPa", "M_bas_MPa", "M_haut_MPa"])
    for c in couches:
        w.writerow([nom, f"{c['z0']:.2f}", f"{c['z1']:.2f}", f"{c['h']:.2f}",
                    c.get("sbt", ""),
                    "" if c.get("Ic") is None else f"{c['Ic']:.2f}",
                    f"{c.get('qc', 0):.2f}",
                    f"{c.get('M_bas', 0):.1f}", f"{c.get('M_haut', 0):.1f}"])
    return buf.getvalue()


def reechantillonner(sondage, pas=0.02):
    """Ramène un sondage à un pas régulier (utile après lecture de courbe)."""
    pts = sondage["points"]
    if not pts:
        return sondage
    z0, z1 = pts[0][0], pts[-1][0]
    n = max(2, int(round((z1 - z0) / pas)) + 1)
    out = []
    j = 0
    for i in range(n):
        z = z0 + i * pas
        while j + 1 < len(pts) and pts[j + 1][0] < z:
            j += 1
        if j + 1 >= len(pts):
            out.append((z, pts[-1][1], pts[-1][2]))
            continue
        za, qa, fa = pts[j]; zb, qb, fb = pts[j + 1]
        t = 0.0 if zb == za else (z - za) / (zb - za)
        q = qa + (qb - qa) * t
        f = None if (fa is None or fb is None) else fa + (fb - fa) * t
        out.append((z, q, f))
    s = dict(sondage)
    s["points"] = out
    return s
