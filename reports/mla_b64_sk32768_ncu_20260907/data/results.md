# B=64、Sk=32768：NCU 补测结果

Baseline = standard；实验组 = localized。Prefill 为 Sq=128 dense。
每项 3 次独立进程采集；LTC 与 memory 分开运行。LTC 单位为 requests，不换算为 bytes。

| 配置 | baseline LTC 中位数 [min, max] | 实验组 LTC 中位数 [min, max] | LTC 减少 | baseline L2 sector hit rate 3 次 (%) | 实验组 L2 3 次 (%) |
| --- | ---: | ---: | ---: | --- | --- |
| decode Sq=1 | 18,925,190 [18,924,421, 18,933,975] | 353,313 [353,304, 353,330] | 98.13% | 37.07, 37.09, 37.07 | 46.96, 46.97, 46.97 |
| decode Sq=4 | 44,183,145 [44,073,434, 44,315,989] | 1,366,063 [1,366,052, 1,366,063] | 96.91% | 64.08, 64.66, 64.31 | 84.11, 83.75, 85.27 |
| prefill Sq=128 | 128,623,443 [128,253,791, 128,976,525] | 43,030,716 [43,030,278, 43,034,854] | 66.55% | 84.28†, 118.02†, 107.11† | 97.62, 100.82†, 100.02† |

L2 为 NCU 原始 `lts__t_sector_hit_rate.pct`。† 表示超出 [0,100]% 或 hit+miss 与 total 偏差超过 5%；不截断、不修复、不据此计算有效命中率提升。5% 沿用项目 QA 阈值，并非 NVIDIA 精度规范。

## Sector 原始计数

δ = 100 × ((hit+miss)/total − 1)。

| 原始 CSV | hit rate (%) | total sectors | hit sectors | miss sectors | δ (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [decode_sq1/memory/standard_01.csv](decode_sq1/memory/standard_01.csv) | 37.07 | 182311611 | 67582637 | 114233745 | -0.27 |
| [decode_sq1/memory/localized_01.csv](decode_sq1/memory/localized_01.csv) | 46.96 | 143963417 | 67599322 | 76321324 | -0.03 |
| [decode_sq1/memory/localized_02.csv](decode_sq1/memory/localized_02.csv) | 46.97 | 143918305 | 67599245 | 76328529 | +0.01 |
| [decode_sq1/memory/standard_02.csv](decode_sq1/memory/standard_02.csv) | 37.09 | 182261291 | 67596962 | 114238141 | -0.23 |
| [decode_sq1/memory/standard_03.csv](decode_sq1/memory/standard_03.csv) | 37.07 | 182271712 | 67567995 | 114253402 | -0.25 |
| [decode_sq1/memory/localized_03.csv](decode_sq1/memory/localized_03.csv) | 46.97 | 143920419 | 67598393 | 76319447 | -0.00 |
| [decode_sq4/memory/standard_01.csv](decode_sq4/memory/standard_01.csv) | 64.08 | 646889369 | 414544114 | 228489322 | -0.60 |
| [decode_sq4/memory/localized_01.csv](decode_sq4/memory/localized_01.csv) | 84.11 | 567420136 | 477282665 | 88813956 | -0.23 |
| [decode_sq4/memory/localized_02.csv](decode_sq4/memory/localized_02.csv) | 83.75 | 562773809 | 471334078 | 89286705 | -0.38 |
| [decode_sq4/memory/standard_02.csv](decode_sq4/memory/standard_02.csv) | 64.66 | 643429690 | 416041677 | 228340056 | +0.15 |
| [decode_sq4/memory/standard_03.csv](decode_sq4/memory/standard_03.csv) | 64.31 | 642926154 | 413439466 | 227134090 | -0.37 |
| [decode_sq4/memory/localized_03.csv](decode_sq4/memory/localized_03.csv) | 85.27 | 560396846 | 477849918 | 88838365 | +1.12 |
| [prefill_sq128/memory/standard_01.csv](prefill_sq128/memory/standard_01.csv) | 84.28 | 16100956824 | 13570334394 | 444218089 | -12.96 |
| [prefill_sq128/memory/localized_01.csv](prefill_sq128/memory/localized_01.csv) | 97.62 | 17939511568 | 17512610540 | 256694315 | -0.95 |
| [prefill_sq128/memory/localized_02.csv](prefill_sq128/memory/localized_02.csv) | 100.82 | 17467174222 | 17610803167 | 259085964 | +2.31 |
| [prefill_sq128/memory/standard_02.csv](prefill_sq128/memory/standard_02.csv) | 118.02 | 11447699513 | 13510348499 | 394403931 | +21.46 |
| [prefill_sq128/memory/standard_03.csv](prefill_sq128/memory/standard_03.csv) | 107.11 | 12569266727 | 13463164121 | 402417172 | +10.31 |
| [prefill_sq128/memory/localized_03.csv](prefill_sq128/memory/localized_03.csv) | 100.02 | 17532251408 | 17535061385 | 262355753 | +1.51 |

NCU duration 仅用于诊断，不作为无 profiler 时的性能测量。完整 requested metrics、软件版本、源码 SHA-256、逐次 argv/env 见 results.json 和 *.command.json。
