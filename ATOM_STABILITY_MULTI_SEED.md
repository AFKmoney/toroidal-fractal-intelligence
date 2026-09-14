# Stability Benchmark multi-seeds — configuration atom-native stabilisée

Date : 2026-09-13  
Objectif : mesurer la reproductibilité de la configuration stabilisée, sans
modifier le code, les équations, les bornes, le tokenizer ou `src/toroidal/`.

## Statut du code

Aucun fichier de code n'a été modifié pour ce benchmark. Les cinq runs ont
utilisé le runner déjà validé :

```text
tools/run_atom_native.py
```

Les seules différences entre les runs sont les seeds et les répertoires de
sortie.

## Configuration strictement identique

```text
corpus                 = /tmp/wiki.train.raw
corpus                  = extrait WikiText-2 raw local, 3 070 caractères
atomizer                = atomizer-v1-byte-span
max_span_bytes          = 16
d_model                 = 8
n_modes                 = 8
n_atoms_max             = 128
episode_length          = 64 transitions
transitions par seed    = 5 000
learning rate           = 1e-3
optimizer               = AdamW
field_max_rms           = 2.0
energy_decay            = [0.90, 0.999]
coupling_scale          = [0.0, 2.0]
phase_sync              = [-1.0, 1.0]
temperature/top_k       = non utilisés pour l'entraînement
```

Seeds exécutées :

```text
20260913, 20260914, 20260915, 20260916, 20260917
```

Commande de référence, répétée avec chaque seed :

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data /tmp/wiki.train.raw \
  --steps 5000 \
  --episode-length 64 \
  --field-max-rms 2.0 \
  --energy-decay-min 0.90 \
  --energy-decay-max 0.999 \
  --seed SEED \
  --output-dir checkpoints/atom_native_stability_seed_SEED
```

Le temps CPU ci-dessous est le temps utilisateur mesuré par le shell. Le
runner enregistre aussi son propre temps écoulé dans chaque `run_metrics.json`.

## Résultats par seed

Les moyennes « début » et « fin » sont calculées sur les points de loss
journalisés tous les 25 steps :

- début : steps 1 à 500;
- fin : steps 4 501 à 5 000.

| Seed | Loss initiale | Loss finale | Loss moy. début | Loss moy. fin | Val byte loss | Val byte PPL | Max RMS | Max norme champ | Limité | Limité % | Coupling | Decay | Phase | NaN/Inf | CPU user s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 20260913 | 8,18672 | 5,69998 | 7,68883 | 5,10295 | 2,97341 | 19,55843 | 2,00000 | 16,0000 | 569 | 11,38 % | 2,0000 | 0,9990 | -0,2206 | non | 31,243 |
| 20260914 | 8,29519 | 5,55559 | 7,68118 | 5,10077 | 2,96041 | 19,30580 | 2,00000 | 16,0000 | 611 | 12,22 % | 2,0000 | 0,9990 | -0,2143 | non | 29,700 |
| 20260915 | 8,48926 | 5,55208 | 7,63457 | 5,04403 | 2,97649 | 19,61881 | 2,00000 | 16,0000 | 8 | 0,16 % | 2,0000 | 0,9961 | -0,0963 | non | 29,627 |
| 20260916 | 8,34029 | 5,73214 | 7,65370 | 5,01540 | 2,97379 | 19,56591 | 2,00000 | 16,0000 | 21 | 0,42 % | 2,0000 | 0,9966 | -0,1294 | non | 29,787 |
| 20260917 | 8,48051 | 5,84928 | 7,63384 | 5,11866 | 2,96121 | 19,32124 | 2,00000 | 16,0000 | 606 | 12,12 % | 2,0000 | 0,9990 | -0,2251 | non | 29,419 |

Observations directes :

- les cinq seeds diminuent leur loss moyenne entre le début et la fin;
- les cinq seeds restent finies;
- les cinq seeds respectent RMS `≤ 2.0` et norme du champ `≤ 16.0`;
- la validation byte perplexity reste dans une plage étroite `19,31–19,62`;
- `coupling_scale` atteint `2.0` pour chaque seed;
- `energy_decay` atteint `0.999` pour trois seeds, mais pas pour les seeds
  `20260915` et `20260916`;
- le nombre de transitions limitées varie fortement selon la seed.

## Tableau demandé

| seed | final loss | val PPL | max RMS | limited % | coupling | decay | phase |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260913 | 5,69998 | 19,55843 | 2,00000 | 11,38 % | 2,0000 | 0,9990 | -0,2206 |
| 20260914 | 5,55559 | 19,30580 | 2,00000 | 12,22 % | 2,0000 | 0,9990 | -0,2143 |
| 20260915 | 5,55208 | 19,61881 | 2,00000 | 0,16 % | 2,0000 | 0,9961 | -0,0963 |
| 20260916 | 5,73214 | 19,56591 | 2,00000 | 0,42 % | 2,0000 | 0,9966 | -0,1294 |
| 20260917 | 5,84928 | 19,32124 | 2,00000 | 12,12 % | 2,0000 | 0,9990 | -0,2251 |

## Statistiques multi-seeds

Écart-type utilisé : écart-type population (`ddof=0`) des cinq seeds.

| Métrique | Moyenne | Écart-type | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Loss initiale | 8,35839 | 0,11475 | 8,18672 | 8,48926 |
| Loss finale | 5,67781 | 0,11278 | 5,55208 | 5,84928 |
| Loss moyenne début | 7,65842 | 0,02297 | 7,63384 | 7,68883 |
| Loss moyenne fin | 5,07636 | 0,03963 | 5,01540 | 5,11866 |
| Validation byte loss | 2,96906 | 0,00683 | 2,96041 | 2,97649 |
| Validation byte perplexity | 19,47404 | 0,13279 | 19,30580 | 19,61881 |
| RMS maximum | 2,00000 | 0,00000006 | 1,99999928 | 1,99999940 |
| Norme max du champ | 15,99999 | 0,00000047 | 15,99999237 | 15,99999332 |
| Transitions limitées | 363,0 | 284,95 | 8 | 611 |
| Fraction limitée | 7,26 % | 5,70 % | 0,16 % | 12,22 % |
| `coupling_scale` final | 2,00000 | 0,00000 | 2,00000 | 2,00000 |
| `energy_decay` final | 0,99795 | 0,00130 | 0,99614 | 0,99900 |
| `phase_sync` final | -0,17714 | 0,05361 | -0,22506 | -0,09634 |
| Temps CPU utilisateur | 29,955 s | 0,655 s | 29,419 s | 31,243 s |

Tous les runs ont signalé `NaN/Inf = non`.

## Atteinte des bornes

```text
coupling_scale = 2.0 : 5 / 5 seeds
energy_decay   = 0.999 : 3 / 5 seeds
les deux simultanément : 3 / 5 seeds
```

La saturation de `coupling_scale` est donc reproductible sur les cinq seeds.
La saturation de `energy_decay` est reproductible sur trois seeds, mais pas sur
les deux autres :

```text
seed 20260915 : energy_decay = 0,996145
seed 20260916 : energy_decay = 0,996587
```

Aucune borne n'a été changée pour expliquer cette différence.

## Verdict

# STABILITY BENCHMARK : PASS

Critères satisfaits sur les 5 seeds :

- 5 000 transitions complétées à chaque fois;
- aucune loss, gradient ou paramètre NaN/Inf;
- champ borné par la limite RMS actuelle;
- norme du champ bornée à environ 16;
- loss moyenne début → fin en baisse pour chaque seed;
- validation byte perplexity stable entre 19,31 et 19,62;
- `coupling_scale=2.0` atteint 5/5 fois.

Nuance importante : le verdict PASS concerne la **stabilité numérique et la
robustesse de l'apprentissage**. La saturation simultanée des deux paramètres
n'est pas entièrement reproductible : `energy_decay=0.999` est atteint par
3/5 seeds seulement. Les bornes restent inchangées comme demandé.

## Artefacts par seed

Chaque répertoire contient :

```text
checkpoints/atom_native_stability_seed_SEED/atom_native_500.pt
checkpoints/atom_native_stability_seed_SEED/loss_trajectory.json
checkpoints/atom_native_stability_seed_SEED/run_metrics.json
checkpoints/atom_native_stability_seed_SEED/atomization_stats.json
checkpoints/atom_native_stability_seed_SEED/console.log
checkpoints/atom_native_stability_seed_SEED/external_time.txt
```

Aucune modification de code n'a été effectuée pendant le benchmark.
