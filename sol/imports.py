# -*- coding: utf-8 -*-
"""Import de sondages CPT — EXTRAIT de modules/sol_import.py v1.0, sans la lecture des PDF.

Le GEF, le CSV/TXT et les exports CSV sont le code du module, octet pour octet (tests/test_sol_moteur.py compare chaque
fonction sur des dizaines de fichiers et rougit au premier écart). La lecture des PDF (courbes vectorielles ou tableau)
a besoin de PyMuPDF, absent du navigateur : elle n'est PAS extraite ; ``importer`` refuse un .pdf avec un message qui
renvoie vers le GEF, le CSV ou l'écran Streamlit. Bibliothèque standard seule.
"""
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
#  3 et 4. PDF — NON EXTRAITS (PyMuPDF absent du navigateur)
#  Les bornes physiques du rapport de frottement restent : elles
#  servent aussi au contrôle d'un import CSV.
# =============================================================
# Plage physique du rapport de frottement Rf = fs/qc. En dehors, ce n'est
# pas un sol : c'est un recalage faux. Sert de garde-fou automatique.
RF_MIN_PLAUSIBLE = 0.05      # %
RF_MAX_PLAUSIBLE = 12.0      # %


# =============================================================
#  5. AIGUILLAGE ET EXPORT
# =============================================================
PDF_REFUSE = ("Un rapport PDF ne se lit pas ici (la lecture des courbes vectorielles a besoin de PyMuPDF) : "
              "importer le fichier GEF ou CSV du bureau d'essais, ou lire le PDF dans l'écran Streamlit.")


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
        # le module lit le PDF avec PyMuPDF (tableau, puis courbes vectorielles) : absent du navigateur
        raise ValueError(PDF_REFUSE)
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
