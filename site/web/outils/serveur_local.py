# -*- coding: utf-8 -*-
"""Le site servi en local COMME sur Cloudflare Pages : les fichiers de site/web, avec les en-têtes de ``_headers``
(la politique de sécurité comprise). verifier_site.js teste ainsi le site sous la même politique qu'en ligne — un
serveur qui l'ignorerait laisserait passer un script que Cloudflare bloquera.

    python3 site/web/outils/serveur_local.py [port]        (8770 par défaut, adresse 127.0.0.1)

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

    def end_headers(self):
        for n, v in entetes(self.path.split("?")[0].split("#")[0], self.REGLES).items():
            self.send_header(n, v)
        super().end_headers()

    def log_message(self, *a):  # le parcours n'a pas besoin du journal des requêtes
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    serveur = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Gestionnaire, directory=WEB))
    print(f"site/web sur http://127.0.0.1:{port} (en-têtes de _headers appliqués)")
    serveur.serve_forever()
