# -*- coding: utf-8 -*-
"""Le site servi en local COMME sur Cloudflare Pages : les fichiers de site/web, avec les en-têtes de ``_headers``
(la politique de sécurité comprise) et les adresses sans « .html ». verifier_site.js teste ainsi le site sous la
même politique et les mêmes adresses qu'en ligne — un serveur qui les ignorerait laisserait passer un script que
Cloudflare bloquera, ou un lien qui y perdrait sa note.

    python3 site/web/outils/serveur_local.py [port] [--dossier D]   (8770 par défaut, adresse 127.0.0.1 ; site/web par
                                                                    défaut, ou la copie qu'écrit construire.py --dossier)

Du format de Cloudflare, on lit ce que le site emploie : une adresse par ligne (``*`` pour « n'importe quoi »), puis
ses en-têtes indentés ; un en-tête donné par plusieurs règles est joint par une virgule, comme chez Cloudflare.
"""
import functools
import http.server
import os
import re
import sys

WEB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def regles(chemin: str = os.path.join(WEB, "_headers")):
    """[(motif, [(nom, valeur), …]), …] dans l'ordre du fichier."""
    out = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            if not ligne.strip() or ligne.lstrip().startswith("#"):
                continue
            if ligne[0] in " \t":
                nom, _, valeur = ligne.strip().partition(":")
                out[-1][1].append((nom.strip(), valeur.strip()))
            else:
                out.append((ligne.strip(), []))
    return out


def entetes(url: str, rg=None) -> dict:
    """Les en-têtes d'une adresse (« /js/etabli.js ») : ceux de toutes les règles qui la couvrent."""
    res = {}
    for motif, hs in (rg if rg is not None else regles()):
        if re.fullmatch(re.escape(motif).replace(r"\*", ".*"), url):
            for n, v in hs:
                res[n] = f"{res[n]}, {v}" if n in res else v
    return res


class Gestionnaire(http.server.SimpleHTTPRequestHandler):
    REGLES = regles()
    RACINE_SITE = WEB

    def send_head(self):
        """Les adresses de Cloudflare Pages : « /poutre » sert poutre.html ; « /poutre.html » est redirigé vers
        « /poutre » (et « /index.html » vers « / »). Ici la redirection JETTE la requête (« ?note=… ») : plus sévère que
        Cloudflare, elle fait échouer le parcours si un lien vise encore une page en « .html » avec sa note."""
        chemin, _, requete = self.path.partition("?")
        if chemin.endswith(".html"):
            self.send_response(308)
            self.send_header("Location", chemin[: -len("index.html")] if chemin.endswith("/index.html") else chemin[: -len(".html")])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if chemin != "/" and "." not in os.path.basename(chemin) and os.path.isfile(os.path.join(self.RACINE_SITE, chemin.lstrip("/") + ".html")):
            self.path = chemin + ".html" + (f"?{requete}" if requete else "")
        return super().send_head()

    def end_headers(self):
        for n, v in entetes(self.path.split("?")[0].split("#")[0], self.REGLES).items():
            self.send_header(n, v)
        super().end_headers()

    def log_message(self, *a):  # le parcours n'a pas besoin du journal des requêtes
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=8770)
    ap.add_argument("--dossier", default=WEB)
    a = ap.parse_args()
    dossier = os.path.abspath(a.dossier)
    Gestionnaire.REGLES = regles(os.path.join(dossier, "_headers"))
    Gestionnaire.RACINE_SITE = dossier
    serveur = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Gestionnaire, directory=dossier))
    print(f"{dossier} sur http://127.0.0.1:{a.port} (en-têtes de _headers appliqués)")
    serveur.serve_forever()
