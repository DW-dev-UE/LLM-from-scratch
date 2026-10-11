# Document-masking correction of attention FLOPs (exact count)

Generated 2026-10-11T14:51:43+00:00 on `bdb00e7715c5` by `paper/repro/doclen_flops.py` (seed 20261011, 20100 rows sampled uniformly over 8193-token windows, elapsed 13 s, 659 MB read, peak RSS 104 MB).

## Headline

- Mixture-weighted F(masked) = 20.8683 GFLOP/token, **F/F_paper = 0.96717** (95% bootstrap 0.96697 to 0.96737; normal approx 0.96697 to 0.96736).
- Attention GFLOP/token: masked mixture 0.7072, no-mask exact (full 8192-token document) 1.3666, paper convention 1.4156.
- Mean segment length 850.6 tokens, median 427.0 (9.63 segments per row); fraction of tokens with p < 1024: 0.6008; rows with a single segment: 0.1151.

## Reference conventions

| convention | mean global ctx | mean local ctx | attention GFLOP/token | F (GFLOP/token) | F / F_paper |
|---|---:|---:|---:|---:|---:|
| paper (apex3/flops.py) | 4096.0 | 1024.0 | 1.4156 | 21.5767 | 1.00000 |
| no-mask exact (one 8192-token document per row) | 4096.5 | 960.0625 | 1.3666 | 21.5277 | 0.99773 |
| masked, phase-1 mixture (this run) | 1434.1 | 633.9770 | 0.7072 | 20.8683 | 0.96717 |

## Per domain

| domain | weight | rows | mean global ctx | mean local ctx | attn GFLOP/tok | F GFLOP/tok | F/F_paper (95% normal) | segs/row | mean seg | median seg | token-wtd mean seg | frac p<1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 0.350 | 7000 | 1606.9 | 677.48 | 0.7671 | 20.9283 | 0.96995 (0.96958 to 0.97031) | 9.03 | 907.5 | 387.0 | 3213 | 0.5483 |
| web | 0.400 | 8000 | 1074.8 | 563.66 | 0.5980 | 20.7591 | 0.96211 (0.96182 to 0.96239) | 11.09 | 738.8 | 440.0 | 2149 | 0.6956 |
| textbook | 0.020 | 400 | 301.2 | 300.57 | 0.2771 | 20.4383 | 0.94724 (0.94709 to 0.94738) | 15.95 | 513.7 | 485.0 | 601 | 0.9940 |
| pdf | 0.050 | 1000 | 2794.1 | 855.76 | 1.0864 | 21.2476 | 0.98474 (0.98392 to 0.98557) | 3.15 | 2602.3 | 1247.0 | 5587 | 0.2980 |
| math | 0.060 | 1200 | 1421.2 | 646.33 | 0.7147 | 20.8758 | 0.96752 (0.96666 to 0.96837) | 7.36 | 1112.7 | 758.0 | 2841 | 0.6269 |
| papers | 0.060 | 1200 | 1833.3 | 753.61 | 0.8604 | 21.0215 | 0.97427 (0.97340 to 0.97513) | 8.36 | 980.2 | 215.0 | 3666 | 0.4167 |
| medical | 0.030 | 600 | 2370.2 | 796.48 | 0.9758 | 21.1369 | 0.97962 (0.97828 to 0.98095) | 6.70 | 1223.0 | 257.0 | 4739 | 0.3419 |
| qa | 0.020 | 400 | 504.6 | 398.57 | 0.3836 | 20.5448 | 0.95217 (0.95149 to 0.95285) | 15.85 | 516.8 | 383.0 | 1008 | 0.8850 |
| books | 0.010 | 300 | 1959.8 | 673.28 | 0.8181 | 20.9793 | 0.97231 (0.97001 to 0.97461) | 9.88 | 829.1 | 313.0 | 3919 | 0.5246 |
| **mixture** | 1.000 | 20100 | 1434.1 | 633.98 | 0.7072 | 20.8683 | 0.96717 (0.96697 to 0.96736) | 9.63 | 850.6 | 427.0 | | 0.6008 |

Segment statistics count every varlen piece as FA3 sees it, including the row-truncated first and last pieces; 'token-wtd mean seg' is the mean segment length experienced by a token (sum L^2 / sum L). Mixture median is a row-weight-corrected median over segments.

## Corrected MFU

Real training: corrected = reported x F_masked / F_paper (factor 0.96717; interval from the bootstrap CI of the factor).

| interval | instance | reported MFU % | corrected MFU % | 95% interval |
|---|---|---:|---:|---|
| 0-2B | A | 36.50 | 35.30 | 35.29 to 35.31 |
| 2-3B | A | 41.20 | 39.85 | 39.84 to 39.86 |
| 3-4B | A | 44.80 | 43.33 | 43.32 to 43.34 |
| 4-6B | A | 46.60 | 45.07 | 45.06 to 45.08 |
| 4-6B | B | 46.80 | 45.26 | 45.25 to 45.27 |
| 6-8B | A | 47.10 | 45.55 | 45.54 to 45.56 |
| 6-8B | B | 47.30 | 45.75 | 45.74 to 45.76 |
| 8-10B | A | 47.50 | 45.94 | 45.93 to 45.95 |
| 8-10B | B | 47.70 | 46.13 | 46.12 to 46.14 |
| 10-10.9B | A | 48.90 | 47.29 | 47.28 to 47.30 |
| 10-10.9B | B | 49.10 | 47.49 | 47.48 to 47.50 |

Benchmark (synthetic, one document per row, no masking): corrected = reported x F_nomask_exact / F_paper (factor 0.99773).

| stage | reported MFU % | corrected MFU % |
|---|---:|---:|
| stage 8 | 45.70 | 45.60 |
| stage 9 | 49.23 | 49.12 |
| stage 10 | 50.74 | 50.62 |

## Method and caveats

- Rows: `*.bin` = flat little-endian uint32 tokens; a domain stream is the sorted concatenation of its shards; window w = stream[w*8193 : w*8193+8193], input = first 8192 tokens (apex3/data.py TokenStream.read, PackedBatches.window, pack_rows). The loader walks windows in a seeded permutation, so uniform random windows reproduce its row distribution; shards are never padded and rows are not aligned to documents.
- Segments: positions restart at row start (even mid-document) and on the token after every EOS; cu_seqlens = indices with position 0 (positions_from_tokens, build_attn_meta). The EOS token is the last token of the segment it ends.
- ctx counts ideal causal keys; FA3 executes whole tiles, so executed FLOPs per token are somewhat higher than this count in segments shorter than a tile, and the correction describes algorithmic FLOPs, the same convention as the paper's formula.
- Mixture = phase-1 weights of server/train_gh200.sh; earlier training intervals are assumed to have used it. Intervals reflect row-sampling error only (rows within a domain resampled).
