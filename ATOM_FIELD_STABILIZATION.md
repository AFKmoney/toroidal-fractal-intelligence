# Stabilisation de l'amplitude du champ ATOM

Date : 2026-09-13  
Objectif : empêcher le champ toroidal de diverger pendant des flux
atom-native longs, tout en conservant la dynamique et le cœur existants.

## Diagnostic avant stabilisation

Le run atom-native non borné de 5 000 transitions avait appris, mais son champ
pouvait prendre des amplitudes gigantesques :

```text
norme alpha maximale observée : 20 335,45
energy_decay final           : 2,8547
coupling_scale final         : 3,8886
```

Le paramètre nommé `energy_decay` était libre de dépasser `1`. Dans la
formule actuelle du cœur :

```text
decay = (energy_decay - 1) * alpha
```

une valeur supérieure à `1` transforme le terme de décroissance en terme de
croissance. Le résultat était donc une perte qui pouvait baisser tandis que
l'amplitude physique du champ explosait.

La perte n'était pas devenue NaN dans le smoke run, mais cette trajectoire
n'était pas acceptable pour un flux continu beaucoup plus long.

## Contraintes conservées

La stabilisation a été ajoutée uniquement dans l'adaptateur atom-native :

- aucun fichier de `src/toroidal/` n'a été modifié;
- aucune équation RK4 du cœur n'a été modifiée;
- aucune interaction toroidale n'a été remplacée;
- aucun Transformer ou mécanisme d'attention n'a été ajouté;
- le format des huit propriétés d'atome reste inchangé;
- l'atomizer et le chemin `AtomPacket → ToroidalAtom` restent les mêmes.

Il s'agit d'une politique de sécurité d'état et d'une projection des
paramètres d'entraînement, pas d'une nouvelle architecture séquentielle.

## Modification 1 — limite RMS du champ

`src/atom_native.py` contient maintenant `FieldAmplitudeController`.

Après l'évolution RK4 et l'interaction locale, mais avant l'agrégation, la
consolidation et la copie dans l'état persistant, il calcule :

```text
rms = sqrt(mean(alpha²) + epsilon)
scale = min(1, max_rms / (rms + epsilon))
alpha_stable = scale * alpha
```

La direction du champ n'est pas modifiée. Les petits champs ne sont pas
réduits. Seuls les champs dont le RMS dépasse la limite sont rescalés.

Configuration du run stabilisé :

```text
max_rms = 2.0
```

Avec `n_modes=8` et `d_model=8`, le champ possède 64 valeurs. La norme
Euclidienne maximale correspondante est donc approximativement :

```text
2.0 * sqrt(64) = 16.0
```

Le contrôleur mesure et journalise :

- RMS avant limite;
- RMS après limite;
- facteur de rescaling;
- norme complète du champ.

## Modification 2 — projection des paramètres partagés

Après chaque `optimizer.step()`, l'expérience applique :

```text
0.90  ≤ energy_decay   ≤ 0.999
0.0   ≤ coupling_scale ≤ 2.0
-1.0  ≤ phase_sync     ≤ 1.0
```

Le point critique est `energy_decay ≤ 0.999`, qui empêche le terme de
croissance de devenir positif dans cette expérience.

Cette projection est exécutée par
`AtomNativeModel.stabilize_dynamics_parameters()` dans le script
`tools/run_atom_native.py`. Elle ne change pas le fichier du cœur dynamique.

## Commande du run stabilisé

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data /tmp/wiki.train.raw \
  --steps 5000 \
  --episode-length 64 \
  --field-max-rms 2.0 \
  --energy-decay-min 0.90 \
  --energy-decay-max 0.999 \
  --output-dir checkpoints/atom_native_stable_5000
```

Toujours sur CPU, sans GPU.

## Comparaison avant / après

| Mesure | Non borné | Stabilisé |
|---|---:|---:|
| Transitions | 5 000 | 5 000 |
| Temps | 27,929 s | 27,088 s |
| Débit | 179,03/s | 184,58/s |
| Loss finale | 5,73878 | 5,69998 |
| Meilleure loss | 3,36482 | 3,61978 |
| Validation loss | 5,39475 | 5,33986 |
| Validation byte perplexity | 20,6052 | 19,5584 |
| NaN/Inf | aucun | aucun |
| Norme max du champ | 20 335,45 | 16,00 |
| RMS après limite max | non borné | 2,0000 |
| Steps limités par le contrôleur | non applicable | 569 / 5 000 |
| Fraction limitée | non applicable | 11,38 % |

La meilleure loss brute du run non borné est légèrement plus basse, mais elle
était obtenue avec des amplitudes du champ extrêmes. Le run stabilisé échange
un peu de minimum d'entraînement contre une trajectoire bornée et une
validation légèrement meilleure.

## Résultats stabilisés détaillés

```text
loss step 1              : 8,18672
loss step 5000           : 5,69998
byte loss step 1         : 5,62346
byte loss step 5000      : 3,42673
validation loss          : 5,33986
validation byte loss     : 2,97341
validation byte PPL      : 19,5584
field RMS avant max      : 2,17457
field RMS après max      : 1,9999994
facteur de scale min     : 0,91972
norme alpha max          : 15,99999
NaN/Inf                  : aucun
```

Le limiteur n'est pas actif au début. Il commence à intervenir lorsque le
champ atteint le seuil, puis ramène le RMS à 2.0. Les paramètres finaux après
projection sont :

```text
coupling_scale = 2,0000
energy_decay   = 0,9990
phase_sync     = -0,2206
```

`coupling_scale` atteint sa borne supérieure. Cela indique qu'il faudra
probablement étudier cette borne et le learning rate dans un futur run plutôt
que de considérer ces valeurs comme définitives.

## État après checkpoint/reload

Checkpoint :

```text
checkpoints/atom_native_stable_5000/atom_native_500.pt
```

État rechargé :

```text
atomes explicites : 8
alpha              : (8, 8)
norme alpha        : 0,99615
t toroidal         : 0,08
champ fini         : oui
paramètres finis   : oui
```

La sauvegarde est faite à la fin d'un nouvel épisode; les 8 atomes correspondent
à la position de l'état au moment exact du checkpoint, et non au nombre maximal
atteint dans les épisodes précédents.

## Génération stabilisée après reload

Paramètres :

```text
temperature = 0.8
top_k       = 5
max_length  = 20
```

Prompt `The future of AI is` :

```text
t.h i asi,nrtn ne aa
```

Prompt `Valkyria Chronicles III is` :

```text
thle to oe,nrin ne a
```

Les sorties avant et après reload sont identiques avec la même seed. Elles
restent faibles linguistiquement; la stabilisation traite l'amplitude, pas la
capacité linguistique du modèle.

## Tests ajoutés et exécutés

Tests atom-native :

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s test -p 'test_atom_native.py' -v
```

Résultat : **8 tests réussis**.

Nouveaux tests spécifiques :

- limite RMS active seulement au-dessus du seuil;
- projection de `energy_decay`, `coupling_scale` et `phase_sync`;
- forward/backward/checkpoint avec l'adaptateur stabilisé.

Invariants du cœur :

```text
3 tests réussis
```

Compilation Python : réussie. `git diff --check` : réussi.

## Fichiers produits

```text
checkpoints/atom_native_stable_5000/atom_native_500.pt
checkpoints/atom_native_stable_5000/loss_trajectory.json
checkpoints/atom_native_stable_5000/run_metrics.json
checkpoints/atom_native_stable_5000/atomization_stats.json
```

Le fichier `run_metrics.json` contient aussi le nombre exact de transitions
limitées, les RMS avant/après et les paramètres partagés finaux.

## Conclusion

La stabilisation fonctionne dans la configuration testée :

```text
champ non borné : norme jusqu'à 20 335
champ stabilisé : norme ≤ 16
```

L'apprentissage continue, la validation s'améliore légèrement et le débit CPU
reste du même ordre. Le système est donc plus sûr pour prolonger les flux,
mais ce n'est pas encore une preuve de stabilité infinie. La prochaine étape
serait de tester plusieurs seeds et plusieurs limites RMS sur un corpus plus
large, puis de vérifier si `coupling_scale` continue de saturer à 2.0.
