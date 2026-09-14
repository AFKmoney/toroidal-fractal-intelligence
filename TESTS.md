# Tests ATOM

## Chemin canonique

Les tests doivent respecter `ATOM_RULES.md`: un token par tick, état conservé,
loss next-token et aucune séquence aplatie.

```bash
python -m pytest test/core_invariants.py
```

## Test d'intégration minimal

Le test d'intégration doit vérifier au minimum:

- création d'un atome par token;
- évolution du champ toroidal;
- exécution RK4 et interaction locale;
- logits de vocabulaire finis;
- plusieurs ticks consécutifs;
- perte next-token scalaire et finie;
- backward et `optimizer.step()`;
- sauvegarde, reload et continuation avec le même état.

## Règle pour les benchmarks

Les anciens scripts de benchmark, de scaling et de debug ont été retirés car
ils aplatissaient les séquences ou ajoutaient une baseline Transformer. Ils ne
sont pas une référence pour ATOM. Toute nouvelle évaluation doit appeler le
trainer séquentiel de `src/training/trainer.py`.
