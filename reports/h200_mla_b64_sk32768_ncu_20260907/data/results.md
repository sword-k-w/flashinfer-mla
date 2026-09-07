# H200 B=64、Sk=32768：NCU 补测结果

Baseline = SM90 attention-only；实验组 = owner-local schedule + compact KV（无审计）。Prefill 为 Sq=128 dense。
每项 3 次独立进程采集；LTC 与 memory 分开运行。LTC 单位为 requests，不换算为 bytes。

| 配置 | baseline LTC 中位数 [min, max] | 实验组 LTC 中位数 [min, max] | LTC 减少 | baseline L2 sector hit rate 3 次 (%) | 实验组 L2 3 次 (%) |
| --- | ---: | ---: | ---: | --- | --- |
| decode Sq=1 | 19,035,409 [19,031,056, 19,043,519] | 255,981 [255,639, 256,443] | 98.66% | 1.10, 1.14, 1.19 | 1.12, 1.12, 1.18 |
| decode Sq=4 | 36,292,519 [36,241,351, 36,337,062] | 551,569 [551,525, 551,660] | 98.48% | 64.58, 65.56, 65.81 | 77.47†, 74.77, 70.74 |
| prefill Sq=128 | 210,968,563 [195,530,361, 213,517,716] | 17,956,238 [17,955,151, 17,956,426] | 91.49% | 82.76†, 112.97†, 89.08† | 92.71, 88.86†, 91.15† |

L2 为 NCU 原始 `lts__t_sector_hit_rate.pct`。† 表示超出 [0,100]% 或 hit+miss 与 total 偏差超过 5%；不截断、不修复、不据此计算有效命中率提升。5% 沿用项目 QA 阈值，并非 NVIDIA 精度规范。

## Sector 原始计数

δ = 100 × ((hit+miss)/total − 1)。

| 原始 CSV | hit rate (%) | total sectors | hit sectors | miss sectors | δ (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [decode_sq1/memory/baseline_01.csv](decode_sq1/memory/baseline_01.csv) | 1.10 | 115352514 | 1272530 | 113791741 | -0.25 |
| [decode_sq1/memory/compact_01.csv](decode_sq1/memory/compact_01.csv) | 1.12 | 77388281 | 869663 | 76394502 | -0.16 |
| [decode_sq1/memory/compact_02.csv](decode_sq1/memory/compact_02.csv) | 1.12 | 77403325 | 865080 | 76393176 | -0.19 |
| [decode_sq1/memory/baseline_02.csv](decode_sq1/memory/baseline_02.csv) | 1.14 | 115330392 | 1311865 | 113792058 | -0.20 |
| [decode_sq1/memory/baseline_03.csv](decode_sq1/memory/baseline_03.csv) | 1.19 | 115347620 | 1375185 | 113791272 | -0.16 |
| [decode_sq1/memory/compact_03.csv](decode_sq1/memory/compact_03.csv) | 1.18 | 77414473 | 914893 | 76393197 | -0.14 |
| [decode_sq4/memory/baseline_01.csv](decode_sq4/memory/baseline_01.csv) | 64.58 | 461813129 | 298218490 | 153555379 | -2.17 |
| [decode_sq4/memory/compact_01.csv](decode_sq4/memory/compact_01.csv) | 77.47 | 302862822 | 234629481 | 84370743 | +5.33 |
| [decode_sq4/memory/compact_02.csv](decode_sq4/memory/compact_02.csv) | 74.77 | 307649824 | 230015866 | 84373491 | +2.19 |
| [decode_sq4/memory/baseline_02.csv](decode_sq4/memory/baseline_02.csv) | 65.56 | 453913164 | 297572645 | 153913625 | -0.53 |
| [decode_sq4/memory/baseline_03.csv](decode_sq4/memory/baseline_03.csv) | 65.81 | 451843716 | 297378075 | 154907515 | +0.10 |
| [decode_sq4/memory/compact_03.csv](decode_sq4/memory/compact_03.csv) | 70.74 | 328535989 | 232412952 | 84368962 | -3.58 |
| [prefill_sq128/memory/baseline_01.csv](prefill_sq128/memory/baseline_01.csv) | 82.76 | 14111806025 | 11679115900 | 495260812 | -13.73 |
| [prefill_sq128/memory/compact_01.csv](prefill_sq128/memory/compact_01.csv) | 92.71 | 11140128902 | 10328336407 | 440334350 | -3.33 |
| [prefill_sq128/memory/compact_02.csv](prefill_sq128/memory/compact_02.csv) | 88.86 | 11158025120 | 9914765491 | 431890645 | -7.27 |
| [prefill_sq128/memory/baseline_02.csv](prefill_sq128/memory/baseline_02.csv) | 112.97 | 11694718597 | 13211403560 | 516605344 | +17.39 |
| [prefill_sq128/memory/baseline_03.csv](prefill_sq128/memory/baseline_03.csv) | 89.08 | 14290094316 | 12728993511 | 495348904 | -7.46 |
| [prefill_sq128/memory/compact_03.csv](prefill_sq128/memory/compact_03.csv) | 91.15 | 11432338941 | 10420159084 | 424961926 | -5.14 |

NCU duration 仅用于诊断，不作为无 profiler 时的性能测量。完整 requested metrics、软件版本、源码 SHA-256、逐次 argv/env 见 results.json 和 *.command.json。
