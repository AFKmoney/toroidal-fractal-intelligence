# Rapport — entraînement atom-native prolongé à 5 000 transitions

Date : 2026-09-13  
Machine : CPU, aucun GPU utilisé  
Corpus : `/tmp/wiki.train.raw`, extrait WikiText-2 raw local

## Commande

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data /tmp/wiki.train.raw \
  --steps 5000 \
  --episode-length 64 \
  --output-dir checkpoints/atom_native_long_5000
```

Aucun code du cœur toroidal n'a été modifié pour ce run.

## Configuration

```text
d_model          = 8
n_modes          = 8
n_atoms_max      = 128
max_span_bytes   = 16
learning rate    = 1e-3
optimizer        = AdamW
épisode          = 64 paquets
seed             = 20260913
```

Le corpus produit 568 paquets atomiques. Le run réutilise 504 paquets pour
l'entraînement et conserve 64 paquets pour la validation.

## Résultats principaux

| Mesure | Résultat |
|---|---:|
| Transitions atomiques | 5 000 |
| Temps CPU | 27,929 s |
| Débit moyen | 179,03 paquets/s |
| Loss au step 1 | 8,18672 |
| Loss au step 5 000 | 5,73878 |
| Meilleure loss observée | 3,36482 au step 4 350 |
| Byte loss au step 1 | 5,62346 |
| Byte loss au step 5 000 | 3,42673 |
| Meilleure byte loss | 1,72950 au step 950 |
| Loss validation | 5,39475 |
| Byte perplexity validation | 20,61 |
| NaN/Inf | aucun |

La loss instantanée n'est pas monotone. Elle recommence régulièrement à un
niveau élevé après chaque reset d'épisode de 64 paquets, puis redescend dans
l'épisode. C'est pourquoi le dernier point n'est pas le meilleur point global.

## Évolution par fenêtres de 500 transitions

Les valeurs ci-dessous sont les moyennes des points enregistrés tous les
25 steps :

| Fenêtre | Loss moyenne | Byte loss moyenne | Meilleure loss |
|---|---:|---:|---:|
| 1–500 | 7,6657 | 4,9791 | 5,3282 |
| 501–1 000 | 5,9436 | 3,7052 | 3,5748 |
| 1 001–1 500 | 5,9360 | 3,4227 | 3,5632 |
| 1 501–2 000 | 6,0331 | 3,4996 | 4,5210 |
| 2 001–2 500 | 5,7105 | 3,3413 | 4,4648 |
| 2 501–3 000 | 5,7543 | 3,3780 | 4,4186 |
| 3 001–3 500 | 5,5717 | 3,0940 | 3,9200 |
| 3 501–4 000 | 5,6599 | 3,0724 | 3,7088 |
| 4 001–4 500 | 5,7830 | 3,2145 | 3,3648 |
| 4 501–5 000 | 5,0644 | 2,8355 | 3,6036 |

La comparaison la plus importante est donc :

```text
loss moyenne fenêtre 1       : 7,6657
loss moyenne fenêtre 4 501–5 000 : 5,0644

byte loss moyenne fenêtre 1       : 4,9791
byte loss moyenne fenêtre 4 501–5 000 : 2,8355
```

La loss descend donc sur la durée en moyenne, même si elle oscille fortement.
La validation progresse également par rapport au run de 500 transitions :

```text
validation byte perplexity à 500 transitions  : 72,52
validation byte perplexity à 5 000 transitions: 20,61
```

## État du champ et stabilité

Le run reste sans NaN ni Inf, mais la norme du champ augmente fortement dans
certains épisodes :

- step 700 : norme `72,09`;
- step 950 : norme `2 088,97`;
- step 1 150 : norme `17 633,48`;
- step 1 600 : norme `20 335,45`.

Les valeurs restent finies grâce au clipping des gradients et au reset
épisodique, mais ce comportement indique une **instabilité d'amplitude** du
champ dans cette configuration. Il ne faut pas interpréter la baisse de loss
comme une validation de stabilité physique à long terme.

Après reload du checkpoint final :

```text
checkpoint : checkpoints/atom_native_long_5000/atom_native_500.pt
atomes     : 8
alpha      : (8, 8)
norme alpha : 1,48747
t toroidal : 0,08
champ fini : oui
paramètres finis : oui
```

Le checkpoint est sauvegardé à la fin d'un nouvel épisode; il contient donc
8 atomes au moment du reload, même si des épisodes précédents ont atteint
jusqu'à 64 atomes.

## Génération après reload

Paramètres exacts :

```text
temperature = 0.8
top_k       = 5
max_length  = 20
```

### `The future of AI is`

```text
thle ttero  aari aae
```

### `Valkyria Chronicles III is`

```text
thle tt e teii dths 
```

Les sorties avant et après reload sont identiques avec la même seed. Elles sont
un peu plus textuelles que celles du run de 500 transitions, mais restent loin
d'une génération linguistique correcte.

## Fichiers produits

```text
checkpoints/atom_native_long_5000/atom_native_500.pt
checkpoints/atom_native_long_5000/loss_trajectory.json
checkpoints/atom_native_long_5000/run_metrics.json
checkpoints/atom_native_long_5000/atomization_stats.json
```

## Conclusion

Oui, l'entraînement prolongé fait descendre la loss moyenne et la loss de
validation :

```text
byte validation perplexity : 72,52 → 20,61
```

Le modèle apprend davantage sur le petit corpus. Cependant, l'expérience
révèle aussi que le champ toroidal peut prendre des amplitudes très élevées.
La prochaine priorité scientifique n'est donc pas simplement d'ajouter des
steps, mais de mesurer et corriger la stabilité d'amplitude avant de conclure
qu'ATOM est efficace sur des flux beaucoup plus longs.
