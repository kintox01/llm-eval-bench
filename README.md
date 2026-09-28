# llm-eval-bench

J'ai fait ce projet pour comparer des LLM (Claude, GPT) avec Jev, le modèle de TypeSafe AI,
sur une tâche de classification avec une sortie typée.

Dans l'idée, le but c'est d'implémenter Jev dans un projet pour le tester face à des modèles
plus coûteux. Je voulais savoir quel modèle utiliser pour trier des tickets de support, pas
juste en regardant qui a le plus de bonnes réponses, mais aussi le coût, le temps de réponse
et, pour Jev, si la confiance qu'il annonce est fiable.

## La tâche

Pour chaque ticket, le modèle doit renvoyer ça :

```python
class TicketLabel(BaseModel):
    category: Literal["billing", "bug", "account", "feature_request", "other"]
    urgency: Literal["low", "medium", "high"]
    refund_requested: bool
```

J'ai écrit et annoté 30 tickets à la main (`data/tickets.jsonl`), avec quelques cas pièges
exprès (ironie, ticket qui est à la fois un bug et une demande de remboursement, etc.). Ils ont
une explication dans le champ `note`.

Les LLM passent par Pydantic AI avec `TicketLabel` comme `output_type`, du coup si un modèle
répond n'importe quoi, c'est détecté direct. Jev, lui, ne prend pas de prompt : je lui envoie
le ticket et trois questions typées (`Choice` pour la catégorie, `Score` pour l'urgence, `Noul`
pour le remboursement). Les définitions des labels sont écrites une seule fois dans `task.py` et
servent aux deux, comme ça la comparaison est juste.

## Installation

```bash
uv sync
cp .env.example .env   # puis mettre ses clés API dedans
uv run pytest          # les tests tournent sans clé et sans réseau
```

Faut au moins une clé parmi `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` et `TYPESAFE_API_KEY`,
selon les modèles qu'on veut lancer.

## Les commandes

### `evalbench run` : lancer les évals

```bash
uv run evalbench run configs/tickets.yaml
```

| Option | À quoi ça sert |
|---|---|
| `CONFIG_PATH` (obligatoire) | le YAML avec le dataset et les modèles à tester |
| `--only NOM` | lance seulement ce run ; on peut le mettre plusieurs fois (`--only jev --only sonnet-v2`) |
| `--limit N` | prend seulement les N premiers tickets, pratique pour tester sans payer grand-chose |
| `--concurrency N` | nombre d'appels en parallèle (5 par défaut dans la config) ; je mets 2 quand ma connexion est instable |
| `--db CHEMIN` | fichier SQLite où stocker les résultats (par défaut `results/evalbench.sqlite`) |

Exemples :

```bash
uv run evalbench run configs/tickets.yaml --only haiku-v1 --limit 5      # test rapide
uv run evalbench run configs/tickets.yaml --only jev --concurrency 2
```

Tous les clients API sont créés avant le premier appel : si une clé manque, ça s'arrête direct au
lieu de planter au milieu après avoir déjà payé les premiers runs.

### `evalbench report` : voir les résultats

```bash
uv run evalbench report                    # dernier run de chaque config
uv run evalbench report <run_id> <run_id>  # des runs précis
```

| Option | À quoi ça sert |
|---|---|
| `RUN_IDS` (facultatif) | les runs à afficher ; sans rien, le dernier run de chaque config |
| `--db CHEMIN` | fichier SQLite à lire |

### `evalbench errors` : voir les tickets ratés

```bash
uv run evalbench errors <run_id>
```

Affiche chaque ticket raté avec la bonne réponse et celle du modèle, les champs faux en rouge.

| Option | À quoi ça sert |
|---|---|
| `RUN_ID` (obligatoire) | le run à inspecter |
| `--db CHEMIN` | fichier SQLite à lire |

### `evalbench compare` : comparer deux runs

```bash
uv run evalbench compare <run_a> <run_b>
```

Liste les tickets que le run B a corrigés et ceux qu'il a cassés par rapport au run A. Utile
pour savoir si un nouveau prompt est vraiment meilleur.

| Option | À quoi ça sert |
|---|---|
| `BASELINE` (obligatoire) | le run de référence |
| `CANDIDATE` (obligatoire) | le run à comparer |
| `--fail-on-regression` | renvoie le code 1 si au moins un ticket a régressé, pour bloquer une PR en CI |
| `--no-fail-on-regression` | comportement par défaut, affiche juste le résultat |
| `--db CHEMIN` | fichier SQLite à lire |

### Le dashboard

```bash
uv run streamlit run dashboard/app.py
```

Tableau des résultats, graphique coût/qualité, matrice de confusion, courbe de calibration de
Jev et liste des erreurs.

## Ajouter un modèle ou un prompt

Il suffit d'ajouter une entrée dans `configs/tickets.yaml` :

```yaml
- name: sonnet-v3
  model: anthropic:claude-sonnet-5
  prompt: prompts/tickets/v3.md
```

Le préfixe du modèle choisit le runner : `typesafe:` pour Jev, sinon n'importe quel modèle
supporté par Pydantic AI. Faut aussi ajouter le prix dans `pricing.py`, sinon le coût
s'affiche `?` (je préfère ça plutôt qu'un coût à 0 qui serait faux).

## Les métriques

- exactitude par champ, et exact match quand les 3 champs sont justes (un appel qui plante
  compte comme faux)
- taux de sorties invalides et taux d'erreurs API, séparés vu que c'est pas le même problème
- latence p50 et p95
- coût pour 1000 tickets
- ECE (calibration), seulement pour Jev vu que c'est le seul qui donne une confiance

## Résultats

Run du 28/09/2026, 30 tickets, `--concurrency 2`. Pas encore testé avec OpenAI.

| run | category | urgency | refund | exact | p50 | p95 | $ / 1000 tickets | ECE |
|---|---|---|---|---|---|---|---|---|
| jev | 100 % | 93 % | 100 % | **93 %** | 0,34 s | 1,11 s | 0,023 | 0,017 |
| sonnet-v2 (Sonnet 5) | 100 % | 90 % | 100 % | 90 % | 2,04 s | 3,63 s | 3,427 | - |
| haiku-v2 (Haiku 4.5) | 97 % | 73 % | 100 % | 70 % | 1,31 s | 5,04 s | 1,576 | - |
| haiku-v1 (Haiku 4.5) | 93 % | 73 % | 100 % | 67 % | 1,17 s | 4,11 s | 1,361 | - |

Ce que j'en retiens :

- Jev et Sonnet 5 sont devant, à un ticket d'écart, donc impossible de vraiment les départager
  avec 30 exemples. Par contre Jev coûte à peu près 150 fois moins cher et répond 6 fois plus
  vite.
- La confiance de Jev est bien calibrée (ECE de 0,017). En gros, on pourrait envoyer à un
  humain les tickets où il a peu confiance et automatiser le reste.
- Haiku plafonne vers 70 %, et mon prompt few-shot (v2) ne change pas grand-chose.
- Quasi toutes les erreurs portent sur l'urgence. La catégorie et le remboursement sont quasi
  parfaits pour tout le monde.
- Les LLM ne sont pas déterministes : haiku-v1 est passé de 100 % à 93 % en catégorie entre
  deux runs sans que je touche à rien.

Quelques notes sur la méthode :

- J'ai corrigé t15 et t21 après un premier run (c'est expliqué dans leur `note`). Pour t15, mon
  annotation ne respectait pas ma propre définition de l'urgence, et t21 était trop ambigu,
  du coup je l'ai réécrit. J'ai fait gaffe à corriger en me basant sur les définitions, pas
  sur ce que les modèles avaient répondu.
- Mon premier run tournait sur un partage de connexion 4G et Jev a eu 43 % de timeouts.
  Relancé avec `--concurrency 2`, plus aucune erreur.

## Limites et ce que je veux ajouter

- 30 exemples c'est peu, les petits écarts ne veulent rien dire (environ ±7 points)
- j'ai tout annoté seul, donc l'urgence reste un peu subjective
- à faire : un cache des réponses, plusieurs runs par config pour mesurer la variance, et une
  deuxième tâche pour montrer que le banc marche sur autre chose que des tickets
