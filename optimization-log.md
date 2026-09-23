# Optimization Log

A record of all my runs so I can keep track of whether or not my optimizations are useful.

## Specs

**CPU:** 13th Gen Intel Core i5-13420H

**RAM:** 64 GB DDR4-3200 SODIMM

**OS:** Windows 11 Pro 25H2 (Build 26200.9168)

## Runs

Using the following command:

```ps
./bench_avg.ps1 -n 40
```

The recorded score represents a median of all the scores across the 40 runs.

To ensure consistent results, I used Quick CPU with the `High performance` power plan. A clock speed of 3.97 GHz was observed during runs.

| Commit hash | Changes | Score |
| - | - | - |
| `d0605ad1` | Initial implementation | 0.707 |
| `288f1808` | Separate boundary logic into separate loops | 0.980 |
| `fc9c7642` | Flatten grid to 1D vector | 1.014 |
| `d1d3dc5f` | Use SIMD in loops | 1.010 |
| `eefbd6eb` | Multithread main stencil loop | 1.977 |
| `43783a1f` | Replace std::vector with array | 1.919 |
| this commit | Divide main loop into blocks | 1.812 |
