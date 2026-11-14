# -*- coding: utf-8 -*-
"""L'écriture d'un CLASSEUR Excel (.xlsx) — ce qu'il faut à un fichier SAF, avec la bibliothèque standard seule : des
feuilles de VALEURS (jamais de formule), des nombres en cellules numériques, des dates en cellules date, et chaque plage
déclarée en TABLEAU Excel nommé.

    ecrire(feuilles) -> les octets du classeur, reproductibles (mêmes feuilles → mêmes octets)

    feuilles : [{"nom": str (31 caractères au plus), "lignes": [[cellule, …], …], "tableau": None | "tblNom"}]
    une cellule : None (vide), str, int, float, bool, datetime.datetime ou datetime.date
    un tableau couvre toute la feuille : sa première ligne porte les en-têtes (des textes, tous différents), et il lui
    faut au moins une ligne de données.
"""
import datetime
import io
import math
import zipfile
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

DATE_ZIP = (2026, 1, 1, 0, 0, 0)
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_P = "http://schemas.openxmlformats.org/package/2006/relationships"
_EPOQUE = datetime.datetime(1899, 12, 30)
_INTERDITS = set('[]:*?/\\')


def colonne(n: int) -> str:
    """Le nom de la colonne n° n (1 → A, 27 → AA)."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _attr(t: str) -> str:
    return escape(t, {'"': "&quot;"})


def _nombre(v) -> str:
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        raise ValueError("un nombre non fini ne s'écrit pas dans un classeur")
    return repr(float(v))


def _serie(d) -> float:
    """Le numéro de série Excel d'une date (système 1900)."""
    if not isinstance(d, datetime.datetime):
        d = datetime.datetime(d.year, d.month, d.day)
    delta = d - _EPOQUE
    return delta.days + (delta.seconds + delta.microseconds / 1e6) / 86400.0


def ecrire(feuilles: List[Dict]) -> bytes:
    """Les octets du classeur (voir l'en-tête du fichier). ValueError pour une feuille ou un tableau mal formé."""
    noms = [f["nom"] for f in feuilles]
    if not feuilles or len(set(noms)) != len(noms):
        raise ValueError("un classeur a des feuilles, de noms différents")
    textes: Dict[str, int] = {}
    xml_feuilles, tableaux = [], []
    for i, f in enumerate(feuilles, 1):
        nom = f["nom"]
        if not nom or len(nom) > 31 or _INTERDITS & set(nom):
            raise ValueError(f"nom de feuille refusé : {nom!r}")
        lignes = f["lignes"]
        largeur = max((len(li) for li in lignes), default=0)
        corps = []
        for r, li in enumerate(lignes, 1):
            cellules = []
            for c, v in enumerate(li, 1):
                if v is None or v == "":
                    continue
                ref = f"{colonne(c)}{r}"
                if isinstance(v, bool):
                    cellules.append(f'<c r="{ref}" t="b"><v>{int(v)}</v></c>')
                elif isinstance(v, (int, float)):
                    cellules.append(f'<c r="{ref}"><v>{_nombre(v)}</v></c>')
                elif isinstance(v, (datetime.datetime, datetime.date)):
                    cellules.append(f'<c r="{ref}" s="1"><v>{_nombre(_serie(v))}</v></c>')
                else:
                    cellules.append(f'<c r="{ref}" t="s"><v>{textes.setdefault(str(v), len(textes))}</v></c>')
            corps.append(f'<row r="{r}">{"".join(cellules)}</row>')
        plage = f"A1:{colonne(max(largeur, 1))}{max(len(lignes), 1)}"
        parties = ""
        if f.get("tableau"):
            entetes = lignes[0] if lignes else []
            if len(lignes) < 2 or len(entetes) != largeur or any(not isinstance(e, str) or not e for e in entetes) \
                    or len(set(entetes)) != len(entetes):
                raise ValueError(f"le tableau « {f['tableau']} » veut des en-têtes (textes différents) et une ligne de données")
            tableaux.append({"feuille": i, "nom": f["tableau"], "plage": plage, "entetes": entetes})
            parties = '<tableParts count="1"><tablePart r:id="rId1"/></tableParts>'
        xml_feuilles.append(f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet xmlns="{_NS}" xmlns:r="{_NS_R}">'
                            f'<dimension ref="{plage}"/><sheetData>{"".join(corps)}</sheetData>{parties}</worksheet>')
    if len({t["nom"] for t in tableaux}) != len(tableaux):
        raise ValueError("deux tableaux portent le même nom")
    n = len(feuilles)
    types = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
             '<Default Extension="xml" ContentType="application/xml"/>',
             '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
             '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
             '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>']
    types += [f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
              for i in range(1, n + 1)]
    types += [f'<Override PartName="/xl/tables/table{j}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"/>'
              for j in range(1, len(tableaux) + 1)]
    entete = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    fichiers = [
        ("[Content_Types].xml", entete + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + "".join(types) + "</Types>"),
        ("_rels/.rels", entete + f'<Relationships xmlns="{_NS_P}"><Relationship Id="rId1" Type="{_NS_R}/officeDocument" Target="xl/workbook.xml"/></Relationships>'),
        ("xl/workbook.xml", entete + f'<workbook xmlns="{_NS}" xmlns:r="{_NS_R}"><sheets>'
         + "".join(f'<sheet name="{_attr(nom)}" sheetId="{i}" r:id="rId{i}"/>' for i, nom in enumerate(noms, 1)) + "</sheets></workbook>"),
        ("xl/_rels/workbook.xml.rels", entete + f'<Relationships xmlns="{_NS_P}">'
         + "".join(f'<Relationship Id="rId{i}" Type="{_NS_R}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, n + 1))
         + f'<Relationship Id="rId{n + 1}" Type="{_NS_R}/styles" Target="styles.xml"/>'
         + f'<Relationship Id="rId{n + 2}" Type="{_NS_R}/sharedStrings" Target="sharedStrings.xml"/></Relationships>'),
        # deux styles de cellule : 0 = général, 1 = date et heure (format n° 22)
        ("xl/styles.xml", entete + f'<styleSheet xmlns="{_NS}"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
         '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
         '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
         '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
         '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
         '<xf numFmtId="22" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>'
         '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'),
        ("xl/sharedStrings.xml", entete + f'<sst xmlns="{_NS}" count="{len(textes)}" uniqueCount="{len(textes)}">'
         + "".join(f'<si><t xml:space="preserve">{escape(t)}</t></si>' for t in textes) + "</sst>"),
    ]
    for i, x in enumerate(xml_feuilles, 1):
        fichiers.append((f"xl/worksheets/sheet{i}.xml", x))
    for j, t in enumerate(tableaux, 1):
        fichiers.append((f"xl/worksheets/_rels/sheet{t['feuille']}.xml.rels", entete + f'<Relationships xmlns="{_NS_P}">'
                         f'<Relationship Id="rId1" Type="{_NS_R}/table" Target="../tables/table{j}.xml"/></Relationships>'))
        fichiers.append((f"xl/tables/table{j}.xml", entete + f'<table xmlns="{_NS}" id="{j}" name="{_attr(t["nom"])}" displayName="{_attr(t["nom"])}" '
                         f'ref="{t["plage"]}" totalsRowShown="0"><autoFilter ref="{t["plage"]}"/><tableColumns count="{len(t["entetes"])}">'
                         + "".join(f'<tableColumn id="{c}" name="{_attr(e)}"/>' for c, e in enumerate(t["entetes"], 1))
                         + "</tableColumns></table>"))
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED) as z:
        for nom, contenu in fichiers:
            info = zipfile.ZipInfo(nom, DATE_ZIP)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, contenu.encode("utf-8"))
    return tampon.getvalue()
