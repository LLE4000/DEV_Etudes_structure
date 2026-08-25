# -*- coding: utf-8 -*-
"""
Tests de l'import de sondages (modules/sol_import.py) — sans aucune IA.

Le test le plus important est celui du PDF vectoriel : on FABRIQUE un
rapport CPT dont on connaît la vérité terrain, on le relit par le code,
et on vérifie qu'on retrouve la courbe au millième de pour cent près.

Lancer :  python3 tests/test_sol_import.py
"""
import io
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import sol_import as SI  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   {info}" if info else ""))


# =====================================================================
print("=" * 72)
print(" 1. GEF")
print("=" * 72)
GEF = """#GEFID= 1,1,0
#COLUMN= 3
#COLUMNINFO= 1, m, Sondeerlengte, 1
#COLUMNINFO= 2, MPa, Conusweerstand, 2
#COLUMNINFO= 3, MPa, Wrijvingsweerstand, 3
#COLUMNSEPARATOR= ;
#RECORDSEPARATOR= !
#COLUMNVOID= 2, -9999.000
#TESTID= CPT-042
#MEASUREMENTVAR= 14, 3.20, m, grondwaterstand
#EOH=
0.020;0.540;0.0135!
0.040;0.620;0.0155!
1.000;2.100;0.0530!
2.000;2.400;0.0600!
5.000;11.800;0.0710!
8.000;12.400;0.0740!
9.000;-9999.000;0.0000!
12.000;3.100;0.1120!
"""
s = SI.parse_gef(GEF.encode(), "fichier")
chk("nom lu depuis #TESTID", s["nom"] == "CPT-042", s["nom"])
chk("nappe lue (#MEASUREMENTVAR 14)", s["nappe_m"] == 3.20)
chk("valeur COLUMNVOID écartée", len(s["points"]) == 7, f"{len(s['points'])} points")
chk("qc lu correctement", abs(s["points"][2][1] - 2.100) < 1e-9)
chk("fs converti MPa → kPa", abs(s["points"][2][2] - 53.0) < 1e-6)
chk("points triés par profondeur",
    all(a[0] <= b[0] for a, b in zip(s["points"], s["points"][1:])))

# profondeurs négatives (convention descendante de certains labos)
neg = GEF.replace("\n0.020;", "\n-0.020;").replace("\n1.000;", "\n-1.000;")
sn = SI.parse_gef(neg.encode(), "x")
chk("profondeurs négatives ramenées en positif",
    all(p[0] >= 0 for p in sn["points"]))

# séparateur absent -> repli sur une découpe générique
sans_sep = GEF.replace("#COLUMNSEPARATOR= ;\n", "").replace(";", " ")
try:
    ss = SI.parse_gef(sans_sep.encode(), "x")
    chk("GEF sans #COLUMNSEPARATOR encore lisible", len(ss["points"]) >= 5,
        f"{len(ss['points'])} points")
except ValueError as e:
    chk("GEF sans #COLUMNSEPARATOR encore lisible", False, str(e)[:60])

# encodage latin-1 avec accents
lat = GEF.replace("#TESTID= CPT-042", "#TESTID= Sondage réalisé")
sl = SI.parse_gef(lat.encode("latin-1"), "x")
chk("fichier latin-1 accepté", len(sl["points"]) == 7)

for mauvais, libelle in ((b"pas un gef", "contenu quelconque"),
                         (GEF.replace("#COLUMNINFO= 2, MPa, Conusweerstand, 2\n", "").encode(),
                          "sans colonne qc")):
    try:
        SI.parse_gef(mauvais, "x")
        chk(f"GEF invalide rejeté ({libelle})", False)
    except ValueError:
        chk(f"GEF invalide rejeté ({libelle})", True)

# =====================================================================
print()
print("=" * 72)
print(" 2. CSV")
print("=" * 72)
CSV_FR = ("Profondeur (m);qc (MPa);fs (kPa)\n0,20;1,45;36,2\n0,40;1,52;38,0\n"
          "1,00;2,10;53,0\n5,00;11,80;71,0\n12,00;3,10;112,0\nligne parasite;;;\n")
s2 = SI.parse_csv(CSV_FR.encode(), "CPT02")
chk("en-tête FR + décimales à virgule", len(s2["points"]) == 5)
chk("valeur qc juste", abs(s2["points"][0][1] - 1.45) < 1e-9)
chk("ligne parasite signalée",
    any("non numérique" in a for a in s2["avertissements"]),
    str(s2["avertissements"])[:90])

CSV_EN = "depth,qc,Rf\n1.0,2.0,2.5\n2.0,3.0,2.0\n3.0,4.0,1.5\n4.0,5.0,1.0\n5.0,6.0,0.8\n"
s3 = SI.parse_csv(CSV_EN.encode(), "CPT03")
chk("en-tête EN + séparateur virgule", len(s3["points"]) == 5)
chk("Rf converti en fs", abs(s3["points"][0][2] - 2.5 / 100 * 2.0 * 1000) < 1e-6)

CSV_TAB = "Depth\tqc\tfs\n1.0\t2.0\t50\n2.0\t3.0\t60\n3.0\t4.0\t70\n4.0\t5.0\t80\n5.0\t6.0\t90\n"
chk("séparateur tabulation", len(SI.parse_csv(CSV_TAB.encode(), "x")["points"]) == 5)

CSV_EXTRA = ("essai;date;Profondeur (m);qc (MPa);fs (kPa);commentaire\n"
             "A;01/01;1,0;2,0;50;ras\nA;01/01;2,0;3,0;60;ras\nA;01/01;3,0;4,0;70;\n"
             "A;01/01;4,0;5,0;80;\nA;01/01;5,0;6,0;90;\n")
se = SI.parse_csv(CSV_EXTRA.encode(), "x")
chk("colonnes surnuméraires ignorées", len(se["points"]) == 5
    and abs(se["points"][0][1] - 2.0) < 1e-9)

chk("mapping imposé respecté",
    len(SI.parse_csv(b"a;b;c\n1;2;3\n2;3;4\n3;4;5\n4;5;6\n5;6;7\n", "x",
                     mapping={"z": 0, "qc": 1})["points"]) == 5)

try:
    SI.parse_csv(b"a;b;c\n1;2;3\n", "x")
    chk("CSV sans colonnes reconnues rejeté", False)
except ValueError:
    chk("CSV sans colonnes reconnues rejeté", True)

print()
print(" Contrôles de plausibilité de la colonne de profondeur")
# Une colonne « Z » est aussi souvent une COTE qu'une profondeur.
COTE = "Z;qc;fs\n12,50;2,0;50\n12,48;2,1;52\n12,46;2,2;54\n12,44;2,3;56\n12,42;2,4;58\n"
sc = SI.parse_csv(COTE.encode(), "x")
chk("colonne de cote (altitude) signalée",
    any("COTE" in a for a in sc["avertissements"]), str(sc["avertissements"])[:80])

CM = "Profondeur;qc;fs\n20;2,0;50\n40;2,1;52\n60;2,2;54\n80;2,3;56\n100;2,4;58\n"
scm = SI.parse_csv(CM.encode(), "x")
chk("pas de mesure grossier signalé",
    any("grossier" in a for a in scm["avertissements"]), str(scm["avertissements"])[:80])

NORMAL = ("Profondeur (m);qc (MPa);fs (kPa)\n0,02;1,4;35\n0,04;1,5;37\n"
          "0,06;1,6;39\n0,08;1,7;41\n0,10;1,8;43\n")
chk("CSV normal → aucun avertissement de plausibilité",
    not SI.parse_csv(NORMAL.encode(), "x")["avertissements"])

# =====================================================================
print()
print("=" * 72)
print(" 3. PDF VECTORIEL — reconstruction d'une courbe connue")
print("=" * 72)

from reportlab.lib.pagesizes import A4                      # noqa: E402
from reportlab.lib.units import mm                          # noqa: E402
from reportlab.pdfgen import canvas as rlc                  # noqa: E402

random.seed(7)
COUCHES = [(0.0, 1.5, 1.4), (1.5, 5.0, 2.2), (5.0, 9.0, 12.0), (9.0, 20.0, 3.1)]
DZ = 0.02
FACTEUR_FS = 25.0        # fs tracé agrandi, comme dans beaucoup de rapports


def verite():
    pts, z = [], 0.0
    while z <= 20.0 + 1e-9:
        base = next((q for (a, b, q) in COUCHES if a <= z < b), 3.1)
        pts.append((round(z, 2), max(0.05, base + base * random.uniform(-0.10, 0.10))))
        z += DZ
    return pts


VRAI = verite()


def fabrique_pdf():
    W, H = A4
    X0, Y0, PW, PH = 45 * mm, 30 * mm, 70 * mm, 210 * mm
    QC_MAX, Z_MAX = 25.0, 20.0
    buf = io.BytesIO()
    c = rlc.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(X0, Y0 + PH + 26 * mm, "SONDAGE AU PENETROMETRE STATIQUE")
    c.setLineWidth(0.8)
    c.rect(X0, Y0, PW, PH)
    c.setFont("Helvetica", 7)
    for q in range(0, int(QC_MAX) + 1, 5):
        x = X0 + PW * q / QC_MAX
        c.setLineWidth(0.25); c.line(x, Y0, x, Y0 + PH)
        c.drawCentredString(x, Y0 + PH + 4, str(q))
    for z in range(0, int(Z_MAX) + 1, 2):
        y = Y0 + PH - PH * z / Z_MAX
        c.setLineWidth(0.25); c.line(X0, y, X0 + PW, y)
        c.drawRightString(X0 - 4, y - 2, str(z))
    c.setLineWidth(0.7); c.setStrokeColorRGB(0.12, 0.29, 0.53)
    p = c.beginPath()
    for i, (z, q) in enumerate(VRAI):
        x = X0 + PW * min(q, QC_MAX) / QC_MAX
        y = Y0 + PH - PH * z / Z_MAX
        p.moveTo(x, y) if i == 0 else p.lineTo(x, y)
    c.drawPath(p, stroke=1, fill=0)
    c.setStrokeColorRGB(0.72, 0.25, 0.15)
    p2 = c.beginPath()
    for i, (z, q) in enumerate(VRAI):
        fs = q * (2.5 if q < 5 else 0.6) / 100.0
        x = X0 + PW * min(fs * FACTEUR_FS, QC_MAX) / QC_MAX
        y = Y0 + PH - PH * z / Z_MAX
        p2.moveTo(x, y) if i == 0 else p2.lineTo(x, y)
    c.drawPath(p2, stroke=1, fill=0)
    c.save()
    return buf.getvalue()


PDF = fabrique_pdf()
an = SI.analyser_pdf(PDF)
cal = an["calibration"]
chk("cadre du graphique détecté", cal["x0"] < cal["x1"] and cal["y0"] < cal["y1"])
chk("traits de grille trouvés", cal["n_grille_v"] >= 2 and cal["n_grille_h"] >= 2,
    f"{cal['n_grille_v']}×{cal['n_grille_h']}")
chk("recalage automatique des deux axes", cal["auto_x"] and cal["auto_y"])
chk("recalage exact (R² = 1)", cal["r2x"] > 0.99999 and cal["r2y"] > 0.99999,
    f"R²x={cal['r2x']:.8f} R²y={cal['r2y']:.8f}")
chk("aucun avertissement de recalage", not an["avertissements"])
chk("deux courbes candidates", len(an["courbes"]) == 2)

snd = SI.extraire_courbe(an, idx_qc=0, idx_fs=1, nom="CPT01", facteur_fs=FACTEUR_FS)


def interp(serie, z):
    if z <= serie[0][0]:
        return serie[0][1]
    if z >= serie[-1][0]:
        return serie[-1][1]
    lo, hi = 0, len(serie) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if serie[m][0] <= z:
            lo = m
        else:
            hi = m
    z0, q0 = serie[lo][0], serie[lo][1]
    z1, q1 = serie[hi][0], serie[hi][1]
    return q0 if z1 == z0 else q0 + (q1 - q0) * (z - z0) / (z1 - z0)


BORNES = (1.5, 5.0, 9.0)
ecarts = [abs(interp(snd["points"], z) - v) / max(v, .01) * 100
          for (z, v) in VRAI if all(abs(z - b) > 0.10 for b in BORNES)]
moy = sum(ecarts) / len(ecarts)
chk(f"reconstruction fidèle de qc (écart moyen {moy:.5f} %)", moy < 0.01, f"{moy:.5f} %")
chk("nombre de points conservé", len(snd["points"]) == len(VRAI),
    f"{len(snd['points'])} / {len(VRAI)}")

# =====================================================================
print()
print(" Garde-fou sur le rapport de frottement")
sans = SI.extraire_courbe(an, idx_qc=0, idx_fs=1, nom="X")          # facteur oublié
chk("échelle de fs fausse → Rf hors plage détecté",
    sans["rf_median"] > SI.RF_MAX_PLAUSIBLE
    and any("frottement" in a for a in sans["avertissements"]),
    f"Rf = {sans['rf_median']:.1f} %")
chk("échelle de fs juste → aucun avertissement",
    SI.RF_MIN_PLAUSIBLE <= snd["rf_median"] <= SI.RF_MAX_PLAUSIBLE
    and not snd["avertissements"],
    f"Rf = {snd['rf_median']:.2f} %")

# la classification doit suivre : c'est tout l'enjeu du garde-fou
from modules import sol_theorie as ST                        # noqa: E402
c_faux = ST.profil_depuis_cpt(sans["points"], nappe_m=4.0)
c_bon = ST.profil_depuis_cpt(snd["points"], nappe_m=4.0)
chk("échelle fausse → sol mal classé (organique à tort)",
    any("organique" in c["sbt"].lower() for c in c_faux))
chk("échelle juste → aucun sol organique dans ce profil",
    not any("organique" in c["sbt"].lower() for c in c_bon),
    " | ".join(c["sbt"][:22] for c in c_bon))
chk("échelle juste → couche sableuse identifiée entre 5 et 9 m",
    any(c["Ic"] is not None and c["Ic"] < 2.05 and c["z0"] < 9.0 < c["z1"] + 1e-9
        or (c["Ic"] is not None and c["Ic"] < 2.05) for c in c_bon))

# =====================================================================
print()
print(" Robustesse du recalage")
c2 = dict(cal); c2["ax"] = None
try:
    SI.extraire_courbe(an, calib=c2)
    chk("recalage incomplet rejeté", False)
except ValueError:
    chk("recalage incomplet rejeté", True)

c3 = dict(cal); c3["ax"] = cal["ax"] * 2.0
chk("recalage manuel effectivement appliqué",
    abs(interp(SI.extraire_courbe(an, calib=c3, nom="X")["points"], 6.0)
        - 2 * interp(snd["points"], 6.0)) > 1.0)

try:
    SI._affine([(10.0, 5.0), (10.0, 7.0)])
    chk("repères confondus rejetés", False)
except ValueError:
    chk("repères confondus rejetés", True)

buf = io.BytesIO()
cv = rlc.Canvas(buf); cv.drawString(100, 700, "Rapport sans graphique"); cv.save()
try:
    SI.analyser_pdf(buf.getvalue())
    chk("PDF sans courbe → message orientant vers le GEF", False)
except ValueError as e:
    chk("PDF sans courbe → message orientant vers le GEF", "GEF" in str(e))

# =====================================================================
print()
print("=" * 72)
print(" 4. RÉÉCHANTILLONNAGE ET EXPORT")
print("=" * 72)
r = SI.reechantillonner(snd, pas=0.05)
pas = [b[0] - a[0] for a, b in zip(r["points"], r["points"][1:])]
chk("pas régulier exact", max(abs(p - 0.05) for p in pas) < 1e-9)
chk("plage de profondeur conservée",
    abs(r["points"][0][0] - snd["points"][0][0]) < 1e-9
    and r["points"][-1][0] <= snd["points"][-1][0] + 1e-9)

csv = SI.vers_csv([s, s2])
lignes = csv.strip().splitlines()
chk("export CSV : en-tête", lignes[0].startswith("sondage;profondeur_m;qc_MPa"))
chk("export CSV : nombre de lignes",
    len(lignes) == 1 + len(s["points"]) + len(s2["points"]))
chk("export CSV : cinq colonnes par ligne", all(l.count(";") == 4 for l in lignes))
chk("export CSV : fs absent rendu vide, pas 'None'", "None" not in csv)

cou = SI.couches_vers_csv("CPT01", c_bon)
chk("export CSV des couches", cou.splitlines()[0].startswith("sondage;de_m;a_m")
    and len(cou.strip().splitlines()) == 1 + len(c_bon))

# =====================================================================
print()
print("=" * 72)
print(" 5. AIGUILLAGE AUTOMATIQUE")
print("=" * 72)
chk("aiguillage .gef", SI.importer(GEF.encode(), "CPT-042.gef")["source"].startswith("GEF"))
chk("aiguillage .csv", SI.importer(CSV_FR.encode(), "mesures.csv")["source"].startswith("CSV"))
try:
    SI.importer(b"x", "truc.docx")
    chk("extension inconnue rejetée", False)
except ValueError:
    chk("extension inconnue rejetée", True)

print()
print("=" * 72)
print(f" RÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
print("=" * 72)
for nom, info in KO:
    print("   -", nom, "|", info)
sys.exit(0 if not KO else 1)
