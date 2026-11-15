# -*- coding: utf-8 -*-
"""Écrire et lire un CLASSEUR EXCEL (.xlsx) avec la bibliothèque standard seule — le navigateur n'a pas openpyxl.

Ce qu'il faut aux fichiers échangés avec les loueurs et les chantiers (cahier § 7, § 8, § 11.5) : des styles (police,
fond, bordure fine, alignement, format des nombres, cellule déverrouillée), des cellules fusionnées, des colonnes et des
lignes masquées, des largeurs et des hauteurs, les volets figés, des listes de validation (sans bloquer : un
avertissement), une mise en forme conditionnelle (une case modifiée devient verte), la protection des feuilles SANS mot
de passe, la mise en page (A4, ajustée à la largeur, titres répétés, pied de page), une feuille « très cachée », des
formules (calculées par Excel à l'ouverture).

    c = Classeur(titre)
    f = c.feuille("NOM")
    f.ecrire(ligne, colonne, valeur, style)      # valeur : None, str, int, float, bool, date ; « =… » : une formule
    octets = c.octets()                           # reproductibles (dates du zip fixes)

    lire(octets) -> {"feuilles": [{"nom", "etat", "cellules": {(ligne, colonne): valeur}}]}

Un STYLE est un dictionnaire : police (taille, gras, italique, couleur), fond, bordure, h (left/center/right), v
(top/center/bottom), retour (à la ligne), retrait, format (des nombres), libre (déverrouillée)."""
import datetime
import io
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

DATE_ZIP = (2026, 1, 1, 0, 0, 0)
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/package/2006/relationships"
EPOQUE = datetime.datetime(1899, 12, 30)
POLICE = "Calibri"


def col(n: int) -> str:
    """Le nom de la colonne n° n (1 → A, 27 → AA)."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def num_col(s: str) -> int:
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - 64)
    return n


def ref(ligne: int, colonne: int) -> str:
    return f"{col(colonne)}{ligne}"


def _x(t) -> str:
    return escape(str(t), {'"': "&quot;"})


def serie(d) -> float:
    """Le numéro de série Excel d'une date (système 1900)."""
    if not isinstance(d, datetime.datetime):
        d = datetime.datetime(d.year, d.month, d.day)
    delta = d - EPOQUE
    return delta.days + delta.seconds / 86400.0


def de_serie(x: float) -> datetime.date:
    return (EPOQUE + datetime.timedelta(days=float(x))).date()


def _argb(c: Optional[str]) -> Optional[str]:
    if not c:
        return None
    c = c.lstrip("#").upper()
    return c if len(c) == 8 else "FF" + c


# ------------------------------------------------------------------ les styles
class _Styles:
    def __init__(self):
        self.polices: List[Tuple] = [(POLICE, 10, False, False, None)]
        self.fonds: List[Optional[str]] = [None, "gray125"]
        self.bordures: List[Optional[str]] = [None]
        self.formats: Dict[str, int] = {}
        self.xfs: List[Tuple] = [(0, 0, 0, 0, (None, None, False, 0, None), False)]
        self.index: Dict[Tuple, int] = {self.xfs[0]: 0}
        self.dxfs: List[Tuple] = []

    def _i(self, lst, v):
        if v not in lst:
            lst.append(v)
        return lst.index(v)

    def xf(self, s: Optional[Dict]) -> int:
        if not s:
            return 0
        p = (s.get("nom", POLICE), s.get("taille", 10), bool(s.get("gras")), bool(s.get("italique")), _argb(s.get("couleur")))
        f = self._i(self.polices, p)
        fd = self._i(self.fonds, _argb(s.get("fond"))) if s.get("fond") else 0
        b = 0
        if s.get("bordure") or s.get("bas"):
            b = self._i(self.bordures, (_argb(s.get("bordure")), _argb(s.get("bas")), s.get("bas_epais", False)))
        nf = 0
        fmt = s.get("format")
        if fmt:
            if fmt not in self.formats:
                self.formats[fmt] = 164 + len(self.formats)
            nf = self.formats[fmt]
        al = (s.get("h"), s.get("v", "center"), bool(s.get("retour")), int(s.get("retrait") or 0), s.get("rotation"))
        cle = (f, fd, b, nf, al, bool(s.get("libre")))
        if cle not in self.index:
            self.index[cle] = len(self.xfs)
            self.xfs.append(cle)
        return self.index[cle]

    def dxf(self, fond: Optional[str], couleur: Optional[str]) -> int:
        cle = (_argb(fond), _argb(couleur))
        if cle not in self.dxfs:
            self.dxfs.append(cle)
        return self.dxfs.index(cle)

    def xml(self) -> str:
        nfs = "".join(f'<numFmt numFmtId="{i}" formatCode="{_x(f)}"/>' for f, i in self.formats.items())
        polices = "".join(
            f'<font>{"<b/>" if g else ""}{"<i/>" if it else ""}<sz val="{t}"/>{f"<color rgb={chr(34)}{c}{chr(34)}/>" if c else ""}'
            f'<name val="{_x(n)}"/><family val="2"/></font>' for n, t, g, it, c in self.polices)
        fonds = ""
        for f in self.fonds:
            if f is None:
                fonds += '<fill><patternFill patternType="none"/></fill>'
            elif f == "gray125":
                fonds += '<fill><patternFill patternType="gray125"/></fill>'
            else:
                fonds += f'<fill><patternFill patternType="solid"><fgColor rgb="{f}"/><bgColor indexed="64"/></patternFill></fill>'
        bords = ""
        for b in self.bordures:
            if b is None:
                bords += "<border><left/><right/><top/><bottom/><diagonal/></border>"
                continue
            tout, bas, epais = b

            def cote(t, c, style="thin"):
                return f'<{t} style="{style}"><color rgb="{c}"/></{t}>' if c else f"<{t}/>"
            bords += ("<border>" + cote("left", tout) + cote("right", tout) + cote("top", tout)
                      + cote("bottom", bas or tout, "medium" if (bas and epais) else "thin") + "<diagonal/></border>")
        xfs = ""
        for f, fd, b, nf, al, libre in self.xfs:
            h, v, retour, retrait, rot = al
            a = ""
            if h:
                a += f' horizontal="{h}"'
            if v:
                a += f' vertical="{v}"'
            if retour:
                a += ' wrapText="1"'
            if retrait:
                a += f' indent="{retrait}"'
            if rot:
                a += f' textRotation="{rot}"'
            prot = '<protection locked="0"/>' if libre else ""
            xfs += (f'<xf numFmtId="{nf}" fontId="{f}" fillId="{fd}" borderId="{b}" xfId="0" applyNumberFormat="1" applyFont="1" '
                    f'applyFill="1" applyBorder="1" applyAlignment="1"{" applyProtection=" + chr(34) + "1" + chr(34) if libre else ""}>'
                    f'<alignment{a}/>{prot}</xf>')
        dxfs = "".join(f'<dxf>{f"<font><color rgb={chr(34)}{c}{chr(34)}/></font>" if c else ""}'
                       f'{f"<fill><patternFill patternType={chr(34)}solid{chr(34)}><bgColor rgb={chr(34)}{fd}{chr(34)}/></patternFill></fill>" if fd else ""}</dxf>'
                       for fd, c in self.dxfs)
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<styleSheet xmlns="{NS}">'
                f'<numFmts count="{len(self.formats)}">{nfs}</numFmts>'
                f'<fonts count="{len(self.polices)}">{polices}</fonts><fills count="{len(self.fonds)}">{fonds}</fills>'
                f'<borders count="{len(self.bordures)}">{bords}</borders>'
                f'<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                f'<cellXfs count="{len(self.xfs)}">{xfs}</cellXfs>'
                f'<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                f'<dxfs count="{len(self.dxfs)}">{dxfs}</dxfs><tableStyles count="0"/></styleSheet>')


# ------------------------------------------------------------------ une feuille
class Feuille:
    def __init__(self, classeur: "Classeur", nom: str):
        if len(nom) > 31 or any(c in nom for c in "[]:*?/\\"):
            raise ValueError(f"nom de feuille invalide : {nom}")
        self.c, self.nom = classeur, nom
        self.cellules: Dict[Tuple[int, int], Tuple] = {}
        self.largeurs: Dict[int, float] = {}
        self.colonnes_cachees: set = set()
        self.hauteurs: Dict[int, float] = {}
        self.lignes_cachees: set = set()
        self.fusions: List[str] = []
        self.validations: List[Dict] = []
        self.conditions: List[Tuple[str, str, int]] = []
        self.fige: Optional[str] = None
        self.quadrillage = True
        self.protegee = False
        self.paysage = True
        self.ajuster_hauteur = 0
        self.titres: Optional[str] = None
        self.pied = "Page &P / &N"
        self.etat = "visible"
        self.zoom = 100

    def ecrire(self, ligne: int, colonne: int, valeur=None, style: Optional[Dict] = None):
        if isinstance(valeur, str) and not valeur:
            valeur = None
        self.cellules[(ligne, colonne)] = (valeur, self.c.styles.xf(style))

    def style(self, ligne: int, colonne: int, style: Dict):
        v = self.cellules.get((ligne, colonne), (None, 0))[0]
        self.ecrire(ligne, colonne, v, style)

    def valeur(self, ligne: int, colonne: int):
        return self.cellules.get((ligne, colonne), (None, 0))[0]

    def fusionner(self, l1: int, c1: int, l2: int, c2: int):
        self.fusions.append(f"{ref(l1, c1)}:{ref(l2, c2)}")

    def largeur(self, colonne: int, w: float):
        self.largeurs[colonne] = w

    def cacher_colonne(self, colonne: int):
        self.colonnes_cachees.add(colonne)

    def hauteur(self, ligne: int, h: float):
        self.hauteurs[ligne] = h

    def cacher_ligne(self, ligne: int):
        self.lignes_cachees.add(ligne)

    def validation(self, plage: str, sorte: str, formule: str, titre: str, message: str, operateur: Optional[str] = None):
        """Une liste de validation qui AVERTIT sans bloquer (errorStyle « warning »)."""
        self.validations.append({"plage": plage, "sorte": sorte, "formule": formule, "titre": titre, "message": message,
                                 "operateur": operateur})

    def condition(self, plage: str, formule: str, fond: str, couleur: Optional[str] = None):
        self.conditions.append((plage, formule, self.c.styles.dxf(fond, couleur)))

    # ---------------------------------------------------------------- XML
    def _cellule(self, l: int, c: int, v, s: int) -> str:
        r = ref(l, c)
        st = f' s="{s}"' if s else ""
        if v is None:
            return f'<c r="{r}"{st}/>' if s else ""
        if isinstance(v, bool):
            return f'<c r="{r}"{st} t="b"><v>{int(v)}</v></c>'
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError("nombre non fini")
            return f'<c r="{r}"{st}><v>{v if isinstance(v, int) else repr(float(v))}</v></c>'
        if isinstance(v, (datetime.date, datetime.datetime)):
            return f'<c r="{r}"{st}><v>{repr(serie(v))}</v></c>'
        t = str(v)
        if t.startswith("=") and len(t) > 1:
            return f'<c r="{r}"{st}><f>{_x(t[1:])}</f></c>'
        esp = ' xml:space="preserve"' if t != t.strip() or "\n" in t else ""
        return f'<c r="{r}"{st} t="inlineStr"><is><t{esp}>{_x(t)}</t></is></c>'

    def xml(self) -> str:
        lignes: Dict[int, List] = {}
        for (l, c), (v, s) in self.cellules.items():
            lignes.setdefault(l, []).append((c, v, s))
        for l in list(self.hauteurs) + list(self.lignes_cachees):
            lignes.setdefault(l, [])
        corps = ""
        for l in sorted(lignes):
            attrs = ""
            if l in self.hauteurs:
                attrs += f' ht="{self.hauteurs[l]}" customHeight="1"'
            if l in self.lignes_cachees:
                attrs += ' hidden="1"'
            cs = "".join(self._cellule(l, c, v, s) for c, v, s in sorted(lignes[l], key=lambda x: x[0]))
            corps += f'<row r="{l}"{attrs}>{cs}</row>'
        cols = ""
        toutes = sorted(set(self.largeurs) | self.colonnes_cachees)
        for c in toutes:
            w = self.largeurs.get(c, 9.0)
            cols += f'<col min="{c}" max="{c}" width="{w}" customWidth="1"{" hidden=" + chr(34) + "1" + chr(34) if c in self.colonnes_cachees else ""}/>'
        vue = f'<sheetView workbookViewId="0"{" showGridLines=" + chr(34) + "0" + chr(34) if not self.quadrillage else ""} zoomScale="{self.zoom}"'
        if self.fige:
            m = re.match(r"([A-Z]+)(\d+)", self.fige)
            cx, ly = num_col(m.group(1)) - 1, int(m.group(2)) - 1
            pane = ("bottomRight" if cx and ly else "bottomLeft" if ly else "topRight")
            vue += (f'><pane{" xSplit=" + chr(34) + str(cx) + chr(34) if cx else ""}{" ySplit=" + chr(34) + str(ly) + chr(34) if ly else ""} '
                    f'topLeftCell="{self.fige}" activePane="{pane}" state="frozen"/><selection pane="{pane}" activeCell="{self.fige}" '
                    f'sqref="{self.fige}"/></sheetView>')
        else:
            vue += "/>"
        prot = ('<sheetProtection sheet="1" objects="1" scenarios="1" formatCells="0" formatColumns="0" formatRows="0" '
                'selectLockedCells="0" selectUnlockedCells="0"/>') if self.protegee else ""
        fus = f'<mergeCells count="{len(self.fusions)}">{"".join(f"<mergeCell ref={chr(34)}{m}{chr(34)}/>" for m in self.fusions)}</mergeCells>' if self.fusions else ""
        cond, prio = "", 1
        for plage, formule, dxf in self.conditions:
            cond += (f'<conditionalFormatting sqref="{plage}"><cfRule type="expression" dxfId="{dxf}" priority="{prio}">'
                     f'<formula>{_x(formule)}</formula></cfRule></conditionalFormatting>')
            prio += 1
        val = ""
        if self.validations:
            val = f'<dataValidations count="{len(self.validations)}">'
            for d in self.validations:
                op = f' operator="{d["operateur"]}"' if d["operateur"] else ""
                val += (f'<dataValidation type="{d["sorte"]}" errorStyle="warning"{op} allowBlank="1" showErrorMessage="1" '
                        f'errorTitle="{_x(d["titre"])}" error="{_x(d["message"])}" sqref="{d["plage"]}"><formula1>{_x(d["formule"])}</formula1>'
                        f'</dataValidation>')
            val += "</dataValidations>"
        mise = (f'<pageMargins left="0.4" right="0.4" top="0.5" bottom="0.5" header="0.3" footer="0.3"/>'
                f'<pageSetup paperSize="9" orientation="{"landscape" if self.paysage else "portrait"}" fitToWidth="1" '
                f'fitToHeight="{self.ajuster_hauteur}"/>')
        pied = f'<headerFooter><oddFooter>&amp;C&amp;"{POLICE}"&amp;8{_x(self.pied)}</oddFooter></headerFooter>' if self.pied else ""
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet xmlns="{NS}" xmlns:r="{NS_R}">'
                f'<sheetPr><pageSetUpPr fitToPage="1"/></sheetPr><sheetViews>{vue}</sheetViews>'
                f'<sheetFormatPr defaultRowHeight="15"/>{"<cols>" + cols + "</cols>" if cols else ""}<sheetData>{corps}</sheetData>'
                f'{prot}{fus}{cond}{val}{mise}{pied}</worksheet>')


# ------------------------------------------------------------------ le classeur
class Classeur:
    def __init__(self, titre: str = ""):
        self.titre = titre
        self.styles = _Styles()
        self.feuilles: List[Feuille] = []

    def feuille(self, nom: str) -> Feuille:
        if any(f.nom == nom for f in self.feuilles):
            raise ValueError(f"feuille en double : {nom}")
        f = Feuille(self, nom)
        self.feuilles.append(f)
        return f

    def octets(self) -> bytes:
        n = len(self.feuilles)
        if not n:
            raise ValueError("un classeur sans feuille")
        sheets = "".join(f'<sheet name="{_x(f.nom)}" sheetId="{i + 1}"{" state=" + chr(34) + f.etat + chr(34) if f.etat != "visible" else ""} r:id="rId{i + 1}"/>'
                         for i, f in enumerate(self.feuilles))
        noms = "".join(f'<definedName name="_xlnm.Print_Titles" localSheetId="{i}">&apos;{_x(f.nom).replace(chr(39), "&apos;&apos;")}&apos;!{f.titres}</definedName>'
                       for i, f in enumerate(self.feuilles) if f.titres)
        actif = next((i for i, f in enumerate(self.feuilles) if f.etat == "visible"), 0)
        wb = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<workbook xmlns="{NS}" xmlns:r="{NS_R}">'
              f'<bookViews><workbookView activeTab="{actif}"/></bookViews><sheets>{sheets}</sheets>'
              f'{"<definedNames>" + noms + "</definedNames>" if noms else ""}<calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>')
        rels = "".join(f'<Relationship Id="rId{i + 1}" Type="{NS_R}/worksheet" Target="worksheets/sheet{i + 1}.xml"/>' for i in range(n))
        rels += f'<Relationship Id="rId{n + 1}" Type="{NS_R}/styles" Target="styles.xml"/>'
        types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="xml" ContentType="application/xml"/>'
                 '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                 + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                           f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(n))
                 + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                 '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                 '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                 '</Types>')
        racine = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{NS_P}">'
                  f'<Relationship Id="rId1" Type="{NS_R}/officeDocument" Target="xl/workbook.xml"/>'
                  f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                  f'<Relationship Id="rId3" Type="{NS_R}/extended-properties" Target="docProps/app.xml"/></Relationships>')
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<cp:coreProperties '
                'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>' + _x(self.titre) + '</dc:title></cp:coreProperties>')
        app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties '
               'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Stab Méthodes</Application></Properties>')
        tampon = io.BytesIO()
        with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED) as z:
            def ajouter(nom, texte):
                zi = zipfile.ZipInfo(nom, DATE_ZIP)
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, texte.encode("utf-8"))
            ajouter("[Content_Types].xml", types)
            ajouter("_rels/.rels", racine)
            ajouter("docProps/core.xml", core)
            ajouter("docProps/app.xml", app)
            ajouter("xl/workbook.xml", wb)
            ajouter("xl/_rels/workbook.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="{NS_P}">{rels}</Relationships>')
            ajouter("xl/styles.xml", self.styles.xml())
            for i, f in enumerate(self.feuilles):
                ajouter(f"xl/worksheets/sheet{i + 1}.xml", f.xml())
        return tampon.getvalue()


# ------------------------------------------------------------------ la lecture
class ClasseurIllisible(ValueError):
    """Le fichier n'est pas un classeur Excel lisible."""


def _q(t: str) -> str:
    return f"{{{NS}}}{t}"


def _texte_si(e) -> str:
    """Le texte d'une chaîne partagée ou en ligne : ses <t>, runs compris (sans les annotations phonétiques)."""
    out = []
    for n in e.iter():
        if n.tag == _q("rPh"):
            continue
        if n.tag == _q("t") and n.text:
            out.append(n.text)
    return "".join(out)


def _cible(base: str, cible: str) -> str:
    if cible.startswith("/"):
        return cible.lstrip("/")
    parts = (base.rsplit("/", 1)[0] + "/" + cible).split("/")
    out = []
    for p in parts:
        if p == "..":
            out and out.pop()
        elif p and p != ".":
            out.append(p)
    return "/".join(out)


def lire(octets: bytes) -> Dict:
    """Les feuilles d'un classeur : {"feuilles": [{"nom", "etat", "cellules": {(ligne, colonne): valeur}}]}. Les valeurs :
    nombre (int si entier), texte, booléen ; une formule rend sa dernière valeur calculée si le fichier la porte, sinon
    son texte (« =… »). Les dates restent des numéros de série (l'appelant sait quelles cases sont des dates : de_serie)."""
    try:
        z = zipfile.ZipFile(io.BytesIO(octets))
        noms = set(z.namelist())
        wb = ET.fromstring(z.read("xl/workbook.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        raise ClasseurIllisible("ce fichier n'est pas un classeur Excel (.xlsx)")
    rels = {}
    if "xl/_rels/workbook.xml.rels" in noms:
        for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
            rels[r.get("Id")] = _cible("xl/workbook.xml", r.get("Target"))
    partagees = []
    sst = next((p for p in rels.values() if p.endswith("sharedStrings.xml")), "xl/sharedStrings.xml")
    if sst in noms:
        partagees = [_texte_si(si) for si in ET.fromstring(z.read(sst)).findall(_q("si"))]
    feuilles = []
    for s in wb.find(_q("sheets")) or []:
        rid = s.get(f"{{{NS_R}}}id")
        chemin = rels.get(rid)
        if not chemin or chemin not in noms:
            continue
        ws = ET.fromstring(z.read(chemin))
        cellules = {}
        for c in ws.iter(_q("c")):
            r = c.get("r")
            m = re.match(r"([A-Z]+)(\d+)$", r or "")
            if not m:
                continue
            l, cc = int(m.group(2)), num_col(m.group(1))
            t = c.get("t")
            v = c.find(_q("v"))
            f = c.find(_q("f"))
            if t == "inlineStr":
                e = c.find(_q("is"))
                val = _texte_si(e) if e is not None else ""
            elif t == "s":
                try:
                    val = partagees[int(v.text)]
                except (TypeError, ValueError, IndexError, AttributeError):
                    val = None
            elif t in ("str", "e"):
                val = v.text if v is not None else (("=" + f.text) if f is not None and f.text else None)
            elif t == "b":
                val = (v is not None and v.text == "1")
            else:
                if v is not None and v.text is not None:
                    try:
                        x = float(v.text)
                        val = int(x) if x.is_integer() and "." not in v.text and "E" not in v.text.upper() else x
                    except ValueError:
                        val = v.text
                elif f is not None and f.text:
                    val = "=" + f.text
                else:
                    val = None
            if val is not None and val != "":
                cellules[(l, cc)] = val
        feuilles.append({"nom": s.get("name"), "etat": s.get("state") or "visible", "cellules": cellules})
    return {"feuilles": feuilles}
