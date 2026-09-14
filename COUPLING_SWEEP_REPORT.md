# Coupling-scale bound sweep — atom-native stability configuration

Date : 2026-09-13  
Objectif : mesurer l'effet de la borne expérimentale de
`coupling_scale`, sans modifier le code de production.

## Règles respectées

- aucun fichier de `src/toroidal/` modifié;
- aucune équation modifiée;
- aucun changement du tokenizer;
- aucun changement de corpus;
- aucun changement d'architecture;
- aucun changement de learning rate, épisode, `d_model`, `n_modes` ou autre
  hyperparamètre;
- `field_max_rms=2.0` inchangé;
- bornes `energy_decay=[0.90, 0.999]` inchangées;
- bornes `phase_sync=[-1.0, 1.0]` inchangées;
- seules les bornes expérimentales de `coupling_scale` ont varié;
- 5 seeds par configuration;
- 2 000 transitions par seed.

La modification de la borne a été appliquée uniquement par monkey-patch en
mémoire dans un harness temporaire `/tmp/coupling_sweep.py`. Aucun fichier de
production n'a été modifié et le harness temporaire n'est pas ajouté au dépôt.

## Configuration fixe

```text
corpus             = /tmp/wiki.train.raw
atomizer           = atomizer-v1-byte-span
max_span_bytes     = 16
d_model            = 8
n_modes            = 8
n_atoms_max        = 128
episode_length     = 64
transitions        = 2 000 par seed
learning rate      = 1e-3
field_max_rms      = 2.0
energy_decay       = [0.90, 0.999]
phase_sync         = [-1.0, 1.0]
seeds              = 20260913–20260917
```

Configurations testées :

```text
coupling_scale_max = 1.0
coupling_scale_max = 1.5
coupling_scale_max = 2.0
coupling_scale_max = 2.5
```

## Tableau comparatif : moyenne ± écart-type

L'écart-type est l'écart-type population calculé sur les cinq seeds de chaque
configuration.

| coupling max | Loss finale | Validation byte loss | Validation byte PPL | RMS maximum | Norme max champ | Fraction limitée | Coupling final | Decay final | Phase finale | NaN/Inf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1,0 | 6,58089 ± 0,07744 | 3,33874 ± 0,01092 | 28,18540 ± 0,30780 | 1,99999931 ± 0,00000009 | 15,99999256 ± 0,00000071 | 0,63 % ± 0,16 % | 1,0000 ± 0,0000 | 0,9990 ± 0,0000 | -0,12190 ± 0,00604 | 0/5 |
| 1,5 | 6,50871 ± 0,07110 | 3,28995 ± 0,01134 | 26,84321 ± 0,30526 | 1,99999926 ± 0,00000009 | 15,99999199 ± 0,00000047 | 0,65 % ± 0,19 % | 1,5000 ± 0,0000 | 0,9990 ± 0,0000 | -0,12049 ± 0,01322 | 0/5 |
| 2,0 | 6,43872 ± 0,06414 | 3,25097 ± 0,01251 | 25,81749 ± 0,32395 | 1,99999928 ± 0,00000000 | 15,99999237 ± 0,00000000 | 1,88 % ± 1,04 % | 2,0000 ± 0,0000 | 0,9990 ± 0,0000 | -0,13146 ± 0,01932 | 0/5 |
| 2,5 | 6,36373 ± 0,05814 | 3,21347 ± 0,01272 | 24,86726 ± 0,31695 | 1,99999931 ± 0,00000005 | 15,99999256 ± 0,00000038 | 3,66 % ± 1,73 % | 2,5000 ± 0,0000 | 0,9990 ± 0,0000 | -0,14728 ± 0,02526 | 0/5 |

## Détails par seed

| coupling max | seed | Loss finale | Val byte PPL | RMS max | Norme max | Limité % | Coupling | Decay | Phase | NaN/Inf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1,0 | 20260913 | 6,44910 | 28,61219 | 2,00000 | 16,0000 | 0,70 % | 1,0000 | 0,9990 | -0,1244 | non |
| 1,0 | 20260914 | 6,54671 | 27,78306 | 2,00000 | 16,0000 | 0,70 % | 1,0000 | 0,9990 | -0,1202 | non |
| 1,0 | 20260915 | 6,66122 | 28,24225 | 2,00000 | 16,0000 | 0,50 % | 1,0000 | 0,9990 | -0,1167 | non |
| 1,0 | 20260916 | 6,59794 | 28,39290 | 2,00000 | 16,0000 | 0,40 % | 1,0000 | 0,9990 | -0,1159 | non |
| 1,0 | 20260917 | 6,64949 | 27,89662 | 2,00000 | 16,0000 | 0,85 % | 1,0000 | 0,9990 | -0,1324 | non |
| 1,5 | 20260913 | 6,38851 | 27,35116 | 2,00000 | 16,0000 | 0,55 % | 1,5000 | 0,9990 | -0,1223 | non |
| 1,5 | 20260914 | 6,47667 | 26,55246 | 2,00000 | 16,0000 | 0,60 % | 1,5000 | 0,9990 | -0,1166 | non |
| 1,5 | 20260915 | 6,58268 | 26,90117 | 2,00000 | 16,0000 | 0,40 % | 1,5000 | 0,9990 | -0,0974 | non |
| 1,5 | 20260916 | 6,52266 | 26,90830 | 2,00000 | 16,0000 | 0,95 % | 1,5000 | 0,9990 | -0,1344 | non |
| 1,5 | 20260917 | 6,57305 | 26,50295 | 2,00000 | 16,0000 | 0,75 % | 1,5000 | 0,9990 | -0,1318 | non |
| 2,0 | 20260913 | 6,33020 | 26,36657 | 2,00000 | 16,0000 | 2,05 % | 2,0000 | 0,9990 | -0,1322 | non |
| 2,0 | 20260914 | 6,41008 | 25,59837 | 2,00000 | 16,0000 | 2,65 % | 2,0000 | 0,9990 | -0,1378 | non |
| 2,0 | 20260915 | 6,50545 | 25,88609 | 2,00000 | 16,0000 | 0,40 % | 2,0000 | 0,9990 | -0,0974 | non |
| 2,0 | 20260916 | 6,45115 | 25,83163 | 2,00000 | 16,0000 | 1,05 % | 2,0000 | 0,9990 | -0,1327 | non |
| 2,0 | 20260917 | 6,49671 | 25,40480 | 2,00000 | 16,0000 | 3,25 % | 2,0000 | 0,9990 | -0,1572 | non |
| 2,5 | 20260913 | 6,26438 | 25,38549 | 2,00000 | 16,0000 | 4,60 % | 2,5000 | 0,9990 | -0,1586 | non |
| 2,5 | 20260914 | 6,34000 | 24,66445 | 2,00000 | 16,0000 | 5,45 % | 2,5000 | 0,9990 | -0,1539 | non |
| 2,5 | 20260915 | 6,42149 | 24,97426 | 2,00000 | 16,0000 | 0,40 % | 2,5000 | 0,9990 | -0,0973 | non |
| 2,5 | 20260916 | 6,37404 | 24,87045 | 2,00000 | 16,0000 | 4,15 % | 2,5000 | 0,9990 | -0,1652 | non |
| 2,5 | 20260917 | 6,41874 | 24,44167 | 2,00000 | 16,0000 | 3,70 % | 2,5000 | 0,9990 | -0,1615 | non |

## Observations factuelles

1. Les quatre configurations restent numériquement stables : aucun NaN/Inf,
   RMS maximum égal à 2.0 et norme maximale du champ égale à environ 16.0.
2. La moyenne de la loss finale diminue à chaque borne testée :

```text
1.0 → 6,58089
1.5 → 6,50871
2.0 → 6,43872
2.5 → 6,36373
```

3. La validation byte PPL suit la même tendance :

```text
1.0 → 28,18540
1.5 → 26,84321
2.0 → 25,81749
2.5 → 24,86726
```

4. La fraction de transitions limitées augmente avec la borne :

```text
1.0 → 0,63 %
1.5 → 0,65 %
2.0 → 1,88 %
2.5 → 3,66 %
```

5. Pour les 20 runs, `energy_decay` termine à `0.9990`.
6. Pour chaque configuration, `coupling_scale` termine à sa borne respective
   dans les 5/5 seeds.
7. Dans cette expérience de 2 000 transitions, `2.5` n'a pas déclenché de
   divergence numérique et obtient les meilleurs scores moyens parmi les
   quatre valeurs testées.

Ces résultats ne justifient pas de changer automatiquement la borne de
production : ils décrivent uniquement l'effet mesuré dans ce sweep court,
sur ce corpus et cette configuration.

## Artefacts

Chaque run possède son propre checkpoint, sa trajectoire et ses métriques :

```text
checkpoints/coupling_sweep_2000_cmax_1p0_seed_20260913/
checkpoints/coupling_sweep_2000_cmax_1p0_seed_20260914/
...
checkpoints/coupling_sweep_2000_cmax_2p5_seed_20260917/
```

Aucun code de production n'a été modifié après l'expérience.
