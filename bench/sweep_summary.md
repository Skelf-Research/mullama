# Mullama vs ollama — sweep results

Source: `/Volumes/Github/mullama/bench/sweep_results.jsonl` (40 runs across 5 models).

Bench: `concurrent_sessions.py`, 4 turns × 48 max_tokens per turn,
temperature=0. Sessions vary by row.


### qwen2.5-0.5b

| sessions | mode | engine | wall (s) | agg tok/s | baseline tok/s | scaling | latency inflation |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | stateless | mullama | 5.37 | 142.9 | 73.2 | 1.95× | 2.04× |
| 4 | stateless | ollama | 9.29 | 82.7 | 61.4 | 1.35× | 2.77× |
| | | **mullama / ollama** | **1.73× faster** | **1.73× more** | | | |
| 4 | session | mullama | 3.83 | 200.6 | 81.8 | 2.45× | 1.63× |
| 4 | session | ollama | 9.64 | 79.7 | 60.9 | 1.31× | 2.82× |
| | | **mullama / ollama** | **2.52× faster** | **2.52× more** | | | |
| 8 | stateless | mullama | 8.88 | 173.0 | 76.3 | 2.27× | 3.16× |
| 8 | stateless | ollama | 18.92 | 81.2 | 63.2 | 1.28× | 5.53× |
| | | **mullama / ollama** | **2.13× faster** | **2.13× more** | | | |
| 8 | session | mullama | 8.38 | 183.2 | 81.4 | 2.25× | 3.00× |
| 8 | session | ollama | 18.46 | 83.2 | 62.2 | 1.34× | 5.34× |
| | | **mullama / ollama** | **2.20× faster** | **2.20× more** | | | |

### llama3.2-1b

| sessions | mode | engine | wall (s) | agg tok/s | baseline tok/s | scaling | latency inflation |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | stateless | mullama | 11.30 | 68.0 | 50.7 | 1.34× | 2.98× |
| 4 | stateless | ollama | 13.78 | 55.8 | 44.0 | 1.27× | 2.97× |
| | | **mullama / ollama** | **1.22× faster** | **1.22× more** | | | |
| 4 | session | mullama | 8.27 | 92.9 | 59.4 | 1.57× | 2.55× |
| 4 | session | ollama | 13.51 | 56.9 | 45.2 | 1.26× | 2.99× |
| | | **mullama / ollama** | **1.63× faster** | **1.63× more** | | | |
| 8 | stateless | mullama | 19.11 | 80.4 | 51.7 | 1.55× | 5.13× |
| 8 | stateless | ollama | 27.17 | 56.5 | 46.1 | 1.23× | 5.88× |
| | | **mullama / ollama** | **1.42× faster** | **1.42× more** | | | |
| 8 | session | mullama | 15.11 | 101.6 | 57.7 | 1.76× | 4.06× |
| 8 | session | ollama | 27.48 | 55.9 | 44.6 | 1.25× | 5.75× |
| | | **mullama / ollama** | **1.82× faster** | **1.82× more** | | | |

### qwen2.5-1.5b

| sessions | mode | engine | wall (s) | agg tok/s | baseline tok/s | scaling | latency inflation |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | stateless | mullama | 15.23 | 50.4 | 39.8 | 1.27× | 3.15× |
| 4 | stateless | ollama | 18.44 | 41.6 | 34.7 | 1.20× | 3.11× |
| | | **mullama / ollama** | **1.21× faster** | **1.21× more** | | | |
| 4 | session | mullama | 15.04 | 51.1 | 44.0 | 1.16× | 3.45× |
| 4 | session | ollama | 18.48 | 41.6 | 34.6 | 1.20× | 3.11× |
| | | **mullama / ollama** | **1.23× faster** | **1.23× more** | | | |
| 8 | stateless | mullama | 22.71 | 67.6 | 38.7 | 1.75× | 4.30× |
| 8 | stateless | ollama | 36.28 | 42.3 | 34.9 | 1.21× | 5.95× |
| | | **mullama / ollama** | **1.60× faster** | **1.60× more** | | | |
| 8 | session | mullama | 20.92 | 73.4 | 43.9 | 1.67× | 4.04× |
| 8 | session | ollama | 36.44 | 42.1 | 34.6 | 1.22× | 5.93× |
| | | **mullama / ollama** | **1.74× faster** | **1.74× more** | | | |

### llama3.2-3b

| sessions | mode | engine | wall (s) | agg tok/s | baseline tok/s | scaling | latency inflation |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | stateless | mullama | 30.90 | 24.9 | 21.3 | 1.17× | 3.42× |
| 4 | stateless | ollama | 30.89 | 24.9 | 21.4 | 1.16× | 3.23× |
| | | **mullama / ollama** | **1.00× faster** | **1.00× more** | | | |
| 4 | session | mullama | 26.81 | 28.6 | 24.7 | 1.16× | 3.44× |
| 4 | session | ollama | 31.05 | 24.7 | 21.8 | 1.14× | 3.30× |
| | | **mullama / ollama** | **1.16× faster** | **1.16× more** | | | |
| 8 | stateless | mullama | 46.15 | 33.3 | 21.5 | 1.55× | 5.16× |
| 8 | stateless | ollama | 61.17 | 25.1 | 21.7 | 1.16× | 6.27× |
| | | **mullama / ollama** | **1.33× faster** | **1.33× more** | | | |
| 8 | session | mullama | 31.19 | 49.2 | 24.8 | 1.98× | 4.04× |
| 8 | session | ollama | 66.10 | 23.2 | 21.7 | 1.07× | 6.83× |
| | | **mullama / ollama** | **2.12× faster** | **2.12× more** | | | |

### qwen2.5-7b

| sessions | mode | engine | wall (s) | agg tok/s | baseline tok/s | scaling | latency inflation |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | stateless | mullama | 51.88 | 14.8 | 10.2 | 1.45× | 2.76× |
| 4 | stateless | ollama | 64.10 | 12.0 | 11.3 | 1.06× | 3.55× |
| | | **mullama / ollama** | **1.24× faster** | **1.24× more** | | | |
| 4 | session | mullama | 48.18 | 15.9 | 12.1 | 1.32× | 3.03× |
| 4 | session | ollama | 64.09 | 12.0 | 11.3 | 1.06× | 3.57× |
| | | **mullama / ollama** | **1.33× faster** | **1.33× more** | | | |
| 8 | stateless | mullama | 98.90 | 15.5 | 10.1 | 1.53× | 4.46× |
| 8 | stateless | ollama | 210.50 | 7.3 | 11.3 | 0.64× | 11.75× |
| | | **mullama / ollama** | **2.13× faster** | **2.13× more** | | | |
| 8 | session | mullama | 80.50 | 19.1 | 12.2 | 1.56× | 4.38× |
| 8 | session | ollama | 182.86 | 8.4 | 10.8 | 0.77× | 9.70× |
| | | **mullama / ollama** | **2.27× faster** | **2.27× more** | | | |

## Headline (best concurrency advantage per model)

| model | sessions | mode | mullama wall | ollama wall | speedup |
|---|---:|---|---:|---:|---:|
| qwen2.5-0.5b | 4 | session | 3.83s | 9.64s | **2.52×** |
| llama3.2-1b | 8 | session | 15.11s | 27.48s | **1.82×** |
| qwen2.5-1.5b | 8 | session | 20.92s | 36.44s | **1.74×** |
| llama3.2-3b | 8 | session | 31.19s | 66.10s | **2.12×** |
| qwen2.5-7b | 8 | session | 80.50s | 182.86s | **2.27×** |
