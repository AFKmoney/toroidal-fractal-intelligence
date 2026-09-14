# Rapport d'exécution — ATOM-native 500 transitions

Date : 2026-09-13  
Machine : CPU de l'environnement Arena, sans GPU utilisé  
Chemin : `Atomizer → AtomCompiler → cœur toroidal existant → AtomSurfaceHead`

## Commande exacte

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data /tmp/wiki.train.raw \
  --steps 500 \
  --episode-length 64 \
  --output-dir checkpoints/atom_native_500
```

Le fichier `/tmp/wiki.train.raw` est le même extrait WikiText-2 raw local que
celui utilisé précédemment : il ne s'agit pas du corpus WikiText-2 complet.
Aucun téléchargement Hugging Face n'a été utilisé pendant ce run.

## Refonte et manipulations réalisées

1. Lecture de `ATOM_RULES.md` et de `ARCHITECTURE.md`.
2. Vérification du décalage entre le chemin documenté `T(x,c,S) → A` et le
   chemin concret GPT-2 `ID → embedding → mapping`.
3. Création de `src/io/atomizer.py` : flux UTF-8, paquets atomiques, contexte
   glissant, frontières déterministes et reconstruction lossless.
4. Correction du découpage à la limite de span pour ne jamais couper un point
   de code UTF-8.
5. Création de `src/atom_native.py` : compilateur observations → huit
   propriétés d'atome, réutilisation du cœur toroidal sans modification de
   `src/toroidal/`, et tête de surface byte/longueur.
6. Création de `test/test_atom_native.py`.
7. Création de `tools/run_atom_native.py`.
8. Smoke run de 10 transitions, terminé avec succès.
9. Run complet de 500 transitions, terminé avec succès.
10. Sauvegarde, reload, puis génération sur les deux prompts demandés.

## Atomisation du corpus

| Mesure | Résultat |
|---|---:|
| Caractères Unicode | 3 070 |
| Bytes UTF-8 | 3 091 |
| Paquets atomiques | 568 |
| Paquets d'entraînement | 504 |
| Paquets de validation | 64 |
| Taille moyenne d'un paquet | 5,4419 bytes |
| Taille minimale | 1 byte |
| Taille maximale | 16 bytes |
| Dimension d'observation | 320 |
| Reconstruction exacte | oui |

Le corpus a été séparé en 504 paquets d'entraînement et 64 paquets de
validation. Chaque transition d'entraînement prédit le paquet atomique
suivant, et non un ID GPT-2.

## Configuration du modèle

```text
d_model          = 8
n_modes          = 8
n_atoms_max      = 128
max_span_bytes   = 16
surface alphabet = 256 bytes
learning rate    = 1e-3
optimizer        = AdamW
épisodes         = 64 paquets
transitions      = 500
seed             = 20260913
```

L'optimizer contient 51 642 paramètres entraînables. Le checkpoint complet
contient 93 646 paramètres, dont l'ancien encodeur/chemin de compatibilité est
gelé et n'est pas utilisé pour l'entrée atom-native.

## Résultats d'entraînement

| Mesure | Résultat |
|---|---:|
| Transitions | 500 |
| Temps CPU | 2,6565 s |
| Débit | 188,47 paquets/s |
| Loss totale, transition 1 | 8,18672 |
| Loss totale, transition 500 | 5,32815 |
| Perplexité totale finale | 206,06 |
| Byte loss, transition 1 | 5,62346 |
| Byte loss, transition 500 | 3,41579 |
| Byte perplexity initiale | 276,85 |
| Byte perplexity finale | 30,44 |
| Loss de validation | 6,76933 |
| Byte loss de validation | 4,28386 |
| Byte perplexity de validation | 72,52 |
| Loss/gradients/paramètres NaN ou Inf | aucun |

La courbe est disponible dans
`checkpoints/atom_native_500/loss_trajectory.json`, avec un point tous les
25 steps. Elle est non monotone, notamment lors des resets d'épisode, mais la
loss d'entraînement et la loss byte diminuent nettement.

## Checkpoint et état après reload

Fichier :

```text
checkpoints/atom_native_500/atom_native_500.pt
```

Le reload a réussi avant la génération. État rechargé :

| Élément | État |
|---|---:|
| Atomes explicites | 52 |
| Champ `alpha` | `(8, 8)` |
| Norme du champ | 7,10765 |
| Temps toroidal | 0,52 |
| Historique d'énergie | 20 entrées |
| Abstractions actives | 0 |
| Consolidations | 0 |
| Norme de persistance | 0 |
| Paramètres partagés finis | oui |
| Champ fini | oui |
| Tous les paramètres finis | oui |

L'absence d'abstraction et de consolidation dans ce smoke run n'est pas une
erreur : les seuils du cœur n'ont pas été modifiés et ce petit entraînement ne
les a pas franchis.

Les paramètres partagés rechargés étaient :

```text
coupling_scale = 1.4767365
energy_decay   = 1.4591408
phase_sync     = -0.0143648
```

Ils restent finis, mais ces valeurs montrent aussi qu'une étude de stabilité
plus longue est nécessaire avant de tirer une conclusion physique sur les
paramètres.

## Génération après reload

Paramètres utilisés exactement :

```text
temperature = 0.8
top_k       = 5
max_length  = 20
```

### Prompt : `The future of AI is`

```text
tnflinhhite imnnthh 
```

### Prompt : `Valkyria Chronicles III is`

```text
tnfli(hhite imnnthh 
```

Les deux générations sont encore faibles et partiellement incohérentes, ce
qui est attendu après 500 transitions sur 3 070 caractères avec `d_model=8`.
Le résultat important de ce run est la validation de la nouvelle interface
atom-native, pas encore la qualité linguistique finale.

Pour les deux prompts, la sortie avant reload et après reload est identique
avec la même seed (`reload_match=true`).

## Tests exécutés

### Tests atom-native

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s test -p 'test_atom_native.py' -v
```

Résultat : **6 tests réussis**.

### Invariants du cœur toroidal

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import runpy
ns = runpy.run_path('test/core_invariants.py')
for name in sorted(ns):
    if name.startswith('test_'):
        ns[name]()
        print(name, 'ok')
PY
```

Résultat : **3 invariants réussis**.

`pytest` n'était pas installé dans `.venv`; les tests ont donc été exécutés
avec le runner standard `unittest` et les fonctions d'invariant directement.
La compilation Python de `src/` et du nouveau test a également réussi.

## Conclusion de l'état actuel

La refonte réalise maintenant un chemin réel :

```text
observation brute → paquet atomique contextuel → ToroidalAtom
```

Elle supprime la dépendance au vocabulaire GPT-2 pour l'entrée et réduit la
sortie de surface à un alphabet byte borné. Le run CPU montre un débit de
188,22 transitions atomiques par seconde dans la configuration smoke.

Ce résultat démontre le fonctionnement, la réversibilité des données, le
backward, le checkpoint et le reload. Il ne démontre pas encore que le modèle
est linguistiquement efficace à grande échelle. Les prochaines mesures utiles
seraient une validation séparée plus grande, plusieurs seeds, une étude de
`max_span_bytes`, puis un run avec davantage de données et une dimension
atomique supérieure.
