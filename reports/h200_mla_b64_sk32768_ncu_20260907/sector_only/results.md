# 四项 sector-only 复测

原始百分比全部保留。QA 要求百分比在 [0,100]% 内，且 hit+miss/total 偏差不超过 5%。
该复测仍为 4 replay passes，没有减少 pass 数。

| CSV | passes | sector hit rate (%) | total sectors | hit sectors | miss sectors | 偏差 (%) | QA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [sq4_baseline_01.csv](sq4_baseline_01.csv) | 4 | 64.05 | 453227998 | 290276840 | 153754645 | -2.03 | passed |
| [sq4_compact_01.csv](sq4_compact_01.csv) | 4 | 78.73 | 320510513 | 252346544 | 84368646 | +5.06 | flagged |
| [sq4_compact_02.csv](sq4_compact_02.csv) | 4 | 77.58 | 326071003 | 252965044 | 84366577 | +3.45 | passed |
| [sq4_baseline_02.csv](sq4_baseline_02.csv) | 4 | 65.57 | 452842215 | 296912862 | 154287837 | -0.36 | passed |
| [sq4_baseline_03.csv](sq4_baseline_03.csv) | 4 | 65.90 | 450884938 | 297148663 | 152896123 | -0.19 | passed |
| [sq4_compact_03.csv](sq4_compact_03.csv) | 4 | 71.77 | 321665039 | 230850620 | 84369102 | -2.00 | passed |
| [sq128_baseline_01.csv](sq128_baseline_01.csv) | 4 | 103.08 | 12857511413 | 13253134047 | 495502602 | +6.93 | flagged |
| [sq128_compact_01.csv](sq128_compact_01.csv) | 4 | 92.67 | 11711227581 | 10852282431 | 424952890 | -3.71 | passed |
| [sq128_compact_02.csv](sq128_compact_02.csv) | 4 | 85.40 | 11920612879 | 10179821786 | 436958480 | -10.94 | flagged |
| [sq128_baseline_02.csv](sq128_baseline_02.csv) | 4 | 103.19 | 12098121866 | 12483967747 | 495243371 | +7.28 | flagged |
| [sq128_baseline_03.csv](sq128_baseline_03.csv) | 4 | 105.92 | 12672405631 | 13422169224 | 495391435 | +9.83 | flagged |
| [sq128_compact_03.csv](sq128_compact_03.csv) | 4 | 93.93 | 11822179046 | 11105159975 | 424955266 | -2.47 | passed |
