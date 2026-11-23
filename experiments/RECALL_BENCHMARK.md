# Recall benchmark (PR 3)

- seed 20261123, n=200/corpus, semantic=False, gate threshold 0.6

| corpus | tier | recall@1 | recall@3 | recall@5 | gate pass/total | conf med |
|---|---|---|---|---|---|---|
| en-facts | exact | 0.0 | 0.0 | 0.0 | 0/200 | 0.0 |
| en-facts | fuzzy | 0.97 | 0.97 | 0.97 | 194/200 | 0.727 |
| en-facts | semantic | 0.0 | 0.0 | 0.0 | 0/200 | 0.0 |
| es-facts | exact | 0.0 | 0.0 | 0.0 | 0/200 | 0.0 |
| es-facts | fuzzy | 1.0 | 1.0 | 1.0 | 197/200 | 0.721 |
| es-facts | semantic | 0.0 | 0.0 | 0.0 | 0/200 | 0.0 |
| code-ids | exact | 0.0 | 0.0 | 0.0 | 0/200 | 0.0 |
| code-ids | fuzzy | 1.0 | 1.0 | 1.0 | 200/200 | 0.727 |
| code-ids | semantic | 0.0 | 0.0 | 0.0 | 0/200 | 0.0 |

## Triage — misses by query class

| query class | misses | fails by tier |
|---|---|---|
| factual | 5 | {'exact': 5, 'fuzzy': 5, 'semantic': 5} |
| procedural | 3 | {'exact': 3, 'fuzzy': 3, 'semantic': 3} |

## Maintainer real-query cases
- [HIT] factual: 'who maintains the billing service'
- [HIT] factual: 'qué motor de base de datos usamos'
- [HIT] code: 'PaymentService retry policy'
- [MISS] procedural: 'como revertimos un despliegue malo'
