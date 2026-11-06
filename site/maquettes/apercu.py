# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » : l'APERÇU CLIQUABLE en une seule page (accueil, projets, établi du garde-corps, équipe),
à ouvrir au téléphone comme au PC. Les quatre vues viennent des MÊMES fonctions que les pages statiques (construire.py) :
les liens entre pages deviennent des ancres (#accueil, #projets, #garde-corps, #equipe), et le thème suit celui de l'appareil
(le bouton soleil / lune le bascule).

Lancement : python3 site/maquettes/apercu.py <fichier_sortie.html>
"""
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import construire as C  # noqa: E402

VUES = (("accueil", C.accueil), ("projets", C.projets), ("garde-corps", C.module), ("equipe", C.equipe))


def css_theme_systeme(css: str) -> str:
    """La charte porte le thème sombre sous [data-theme="dark"]. L'aperçu le donne aussi à l'appareil réglé en sombre,
    sauf choix explicite du clair : chaque règle sombre est reproduite sous @media (prefers-color-scheme: dark). Ces
    règles sont DÉRIVÉES du fichier à chaque construction (aucune copie tenue à la main) ; le compte est vérifié."""
    blocs = re.findall(r'^(\[data-theme="dark"\][^{]*\{[^}]*\})', css, flags=re.M)
    attendu = len(re.findall(r'^\[data-theme="dark"\]', css, flags=re.M))
    assert blocs and len(blocs) == attendu, (len(blocs), attendu)
    reprises = [b.replace('[data-theme="dark"]', ':root:not([data-theme="light"])', 1) for b in blocs]
    return css + "\n/* le thème de l'appareil (aperçu) */\n@media (prefers-color-scheme: dark) {\n" + "\n".join(reprises) + "\n}\n"


def construire_apercu() -> str:
    css = css_theme_systeme(open(os.path.join(ICI, "charte.css"), encoding="utf-8").read())
    vues, scripts = "", ""
    for i, (cle, fn) in enumerate(VUES):
        _titre, corps, script = fn()
        vues += f'<div class="vpage" data-page="{cle}"{"" if i == 0 else " hidden"}>{corps}</div>'
        scripts += script
    routeur = """
document.documentElement.lang='fr';
function MONTRER(p){
  var ok=false;
  document.querySelectorAll('.vpage').forEach(function(v){var on=v.dataset.page===p;v.hidden=!on;ok=ok||on;});
  if(!ok){document.querySelector('.vpage').hidden=false;}
  window.scrollTo(0,0);
  if(window.RECENTRER) window.RECENTRER();
}
window.ALLER=function(p){if(location.hash==='#'+p) MONTRER(p); else location.hash=p;};
window.addEventListener('hashchange',function(){MONTRER(location.hash.slice(1)||'accueil');});
"""
    depart = "\nMONTRER(location.hash.slice(1)||'accueil');\n"
    return ('<title>Maquettes Stab Méthodes</title>\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap">\n'
            f'<style>{css}</style>\n{vues}\n<script>{routeur}{C.SCRIPT_COMMUN}{scripts}{depart}</script>\n')


def main():
    sortie = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ICI, "apercu.html")
    with open(sortie, "w", encoding="utf-8") as f:
        f.write(construire_apercu())
    print(sortie, os.path.getsize(sortie) // 1024, "ko")


if __name__ == "__main__":
    main()
