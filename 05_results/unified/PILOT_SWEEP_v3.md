# Pilot sweep and fixed-candidate decomposition (population estimand, pilot charged in full, t bound)

Source: 05_results/menu_allocation/menu_alloc_*_v3b*[_ntr*|_candbest].csv (300 draws per budget).


## eps = 0.01

### 1. Candidate quality vs pilot size (candidate chosen on the pilot; identical across arms)

| Collection | n_train | P(regret <= eps) | median slack | slack q25 | pilot docs |
|---|---:|---:|---:|---:|---:|
| ANTIQUE | 10 | 0.73 | 0.010 | 0.010 | 315 |
| ANTIQUE | 20 | 0.86 | 0.010 | 0.010 | 635 |
| ANTIQUE | 40 | 0.92 | 0.010 | 0.010 | 1,268 |
| ANTIQUE | 80 | 0.99 | 0.010 | 0.010 | 2,530 |
| CAsT | 10 | 0.73 | 0.010 | 0.006 | 491 |
| CAsT | 20 | 0.85 | 0.010 | 0.009 | 975 |
| CAsT | 40 | 0.93 | 0.010 | 0.009 | 1,955 |
| CAsT | 80 | 1.00 | 0.010 | 0.009 | 3,906 |
| DBpedia | 10 | 0.79 | 0.010 | 0.010 | 513 |
| DBpedia | 20 | 0.84 | 0.010 | 0.010 | 1,029 |
| DBpedia | 40 | 0.89 | 0.010 | 0.010 | 2,059 |
| DBpedia | 80 | 0.94 | 0.010 | 0.010 | 4,115 |
| DL 21-23 | 10 | 0.72 | 0.010 | 0.010 | 692 |
| DL 21-23 | 20 | 0.87 | 0.010 | 0.010 | 1,386 |
| DL 21-23 | 40 | 0.95 | 0.010 | 0.010 | 2,773 |
| DL 21-23 | 80 | 1.00 | 0.010 | 0.010 | 5,555 |

### 2. Certification given a feasible candidate (max over budgets of ACT | regret <= eps)

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 0.65 | 0.74 | 0.83 | 0.96 | 0.68 |
| ANTIQUE | static_sum | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| ANTIQUE | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ANTIQUE | plugin_exact | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| CAsT | per_pair | 0.06 | 0.05 | 0.01 | 0.01 | 0.05 |
| CAsT | static_sum | 0.95 | 0.97 | 0.99 | 1.00 | 1.00 |
| CAsT | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAsT | plugin_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DBpedia | per_pair | 0.51 | 0.58 | 0.59 | 0.55 | 0.56 |
| DBpedia | static_sum | 0.81 | 0.83 | 0.87 | 0.87 | 0.90 |
| DBpedia | oracle_exact | 0.85 | 0.87 | 0.90 | 0.89 | 0.91 |
| DBpedia | plugin_exact | 0.79 | 0.86 | 0.88 | 0.88 | 0.83 |
| DL 21-23 | per_pair | 0.77 | 0.87 | 0.96 | 1.00 | 0.86 |
| DL 21-23 | static_sum | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| DL 21-23 | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DL 21-23 | plugin_exact | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |

### 3. Total unique documents (pilot included) to reach ACT 0.5

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | n.r. | 2,232 | 2,173 | 2,986 | 2,092 |
| ANTIQUE | static_sum | 989 | 1,067 | 1,588 | 2,769 | 1,013 |
| ANTIQUE | oracle_exact | 875 | 1,052 | 1,588 | 2,769 | 1,037 |
| ANTIQUE | plugin_exact | 989 | 1,075 | 1,588 | 2,770 | 995 |
| CAsT | per_pair | n.r. | n.r. | n.r. | n.r. | n.r. |
| CAsT | static_sum | 4,368 | 4,349 | 4,629 | 5,417 | 3,330 |
| CAsT | oracle_exact | 3,809 | 3,629 | 4,199 | 5,160 | 3,086 |
| CAsT | plugin_exact | 4,089 | 3,802 | 4,270 | 5,210 | 3,262 |
| DBpedia | per_pair | n.r. | n.r. | 5,854 | 8,038 | 4,123 |
| DBpedia | static_sum | 2,533 | 2,365 | 3,146 | 5,082 | 2,206 |
| DBpedia | oracle_exact | 2,781 | 2,349 | 3,113 | 5,013 | 2,106 |
| DBpedia | plugin_exact | 2,900 | 2,655 | 3,339 | 5,250 | 2,155 |
| DL 21-23 | per_pair | 4,513 | 4,229 | 4,684 | 6,615 | 3,701 |
| DL 21-23 | static_sum | 1,873 | 2,284 | 3,458 | 5,959 | 2,194 |
| DL 21-23 | oracle_exact | 1,730 | 2,157 | 3,325 | 5,902 | 2,081 |
| DL 21-23 | plugin_exact | 1,878 | 2,219 | 3,353 | 5,900 | 2,180 |

### 3b. Sampling documents only (pilot excluded) to reach ACT 0.5

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | n.r. | 1,597 | 904 | 456 | 1,462 |
| ANTIQUE | static_sum | 673 | 432 | 320 | 239 | 382 |
| ANTIQUE | oracle_exact | 560 | 417 | 320 | 239 | 406 |
| ANTIQUE | plugin_exact | 674 | 440 | 320 | 240 | 365 |
| CAsT | per_pair | n.r. | n.r. | n.r. | n.r. | n.r. |
| CAsT | static_sum | 3,877 | 3,375 | 2,675 | 1,511 | 2,357 |
| CAsT | oracle_exact | 3,319 | 2,655 | 2,245 | 1,254 | 2,114 |
| CAsT | plugin_exact | 3,598 | 2,828 | 2,316 | 1,305 | 2,290 |
| DBpedia | per_pair | n.r. | n.r. | 3,795 | 3,923 | 3,093 |
| DBpedia | static_sum | 2,021 | 1,336 | 1,087 | 967 | 1,176 |
| DBpedia | oracle_exact | 2,268 | 1,320 | 1,055 | 898 | 1,076 |
| DBpedia | plugin_exact | 2,387 | 1,626 | 1,281 | 1,135 | 1,125 |
| DL 21-23 | per_pair | 3,821 | 2,843 | 1,912 | 1,061 | 2,312 |
| DL 21-23 | static_sum | 1,181 | 897 | 686 | 405 | 805 |
| DL 21-23 | oracle_exact | 1,038 | 770 | 552 | 347 | 692 |
| DL 21-23 | plugin_exact | 1,186 | 833 | 580 | 345 | 791 |

Max wrong-certificate rate over all cells at eps=0.01: 0.003

## eps = 0.02

### 1. Candidate quality vs pilot size (candidate chosen on the pilot; identical across arms)

| Collection | n_train | P(regret <= eps) | median slack | slack q25 | pilot docs |
|---|---:|---:|---:|---:|---:|
| ANTIQUE | 10 | 0.73 | 0.020 | 0.020 | 315 |
| ANTIQUE | 20 | 0.85 | 0.020 | 0.020 | 634 |
| ANTIQUE | 40 | 0.93 | 0.020 | 0.020 | 1,265 |
| ANTIQUE | 80 | 0.99 | 0.020 | 0.020 | 2,529 |
| CAsT | 10 | 0.83 | 0.019 | 0.014 | 489 |
| CAsT | 20 | 0.92 | 0.020 | 0.017 | 976 |
| CAsT | 40 | 0.97 | 0.020 | 0.019 | 1,953 |
| CAsT | 80 | 1.00 | 0.020 | 0.019 | 3,900 |
| DBpedia | 10 | 0.87 | 0.020 | 0.020 | 510 |
| DBpedia | 20 | 0.92 | 0.020 | 0.020 | 1,031 |
| DBpedia | 40 | 0.96 | 0.020 | 0.020 | 2,053 |
| DBpedia | 80 | 1.00 | 0.020 | 0.020 | 4,112 |
| DL 21-23 | 10 | 0.71 | 0.020 | 0.020 | 697 |
| DL 21-23 | 20 | 0.87 | 0.020 | 0.020 | 1,386 |
| DL 21-23 | 40 | 0.98 | 0.020 | 0.020 | 2,774 |
| DL 21-23 | 80 | 1.00 | 0.020 | 0.020 | 5,534 |

### 2. Certification given a feasible candidate (max over budgets of ACT | regret <= eps)

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 0.72 | 0.81 | 0.89 | 0.98 | 0.78 |
| ANTIQUE | static_sum | 0.96 | 0.98 | 1.00 | 1.00 | 1.00 |
| ANTIQUE | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ANTIQUE | plugin_exact | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAsT | per_pair | 0.07 | 0.07 | 0.03 | 0.09 | 0.12 |
| CAsT | static_sum | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAsT | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CAsT | plugin_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DBpedia | per_pair | 0.59 | 0.58 | 0.63 | 0.65 | 0.66 |
| DBpedia | static_sum | 0.88 | 0.85 | 0.89 | 0.90 | 0.99 |
| DBpedia | oracle_exact | 0.92 | 0.89 | 0.93 | 0.95 | 0.99 |
| DBpedia | plugin_exact | 0.87 | 0.86 | 0.92 | 0.95 | 0.97 |
| DL 21-23 | per_pair | 0.80 | 0.93 | 0.98 | 1.00 | 0.96 |
| DL 21-23 | static_sum | 0.95 | 0.98 | 1.00 | 1.00 | 1.00 |
| DL 21-23 | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DL 21-23 | plugin_exact | 0.96 | 0.98 | 1.00 | 1.00 | 1.00 |

### 3. Total unique documents (pilot included) to reach ACT 0.5

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 2,269 | 1,784 | 2,049 | 2,900 | 1,657 |
| ANTIQUE | static_sum | 890 | 1,015 | 1,585 | 2,769 | 990 |
| ANTIQUE | oracle_exact | 849 | 994 | 1,584 | 2,770 | 991 |
| ANTIQUE | plugin_exact | 835 | 994 | 1,586 | 2,769 | 990 |
| CAsT | per_pair | n.r. | n.r. | n.r. | n.r. | n.r. |
| CAsT | static_sum | 3,050 | 3,181 | 3,618 | 4,779 | 2,523 |
| CAsT | oracle_exact | 2,665 | 2,749 | 3,407 | 4,680 | 2,349 |
| CAsT | plugin_exact | 2,735 | 2,799 | 3,377 | 4,665 | 2,467 |
| DBpedia | per_pair | 5,007 | 5,308 | 4,897 | 6,273 | 3,816 |
| DBpedia | static_sum | 1,605 | 2,186 | 2,797 | 4,749 | 1,784 |
| DBpedia | oracle_exact | 1,724 | 2,183 | 2,770 | 4,957 | 1,824 |
| DBpedia | plugin_exact | 1,773 | 2,465 | 2,920 | 4,918 | 1,828 |
| DL 21-23 | per_pair | 3,695 | 3,234 | 4,139 | 6,336 | 3,057 |
| DL 21-23 | static_sum | 1,753 | 2,102 | 3,283 | 5,859 | 2,059 |
| DL 21-23 | oracle_exact | 1,668 | 1,953 | 3,192 | 5,839 | 1,913 |
| DL 21-23 | plugin_exact | 1,759 | 2,051 | 3,250 | 5,839 | 1,997 |

### 3b. Sampling documents only (pilot excluded) to reach ACT 0.5

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 1,953 | 1,151 | 784 | 371 | 1,027 |
| ANTIQUE | static_sum | 574 | 381 | 320 | 240 | 360 |
| ANTIQUE | oracle_exact | 534 | 360 | 319 | 241 | 360 |
| ANTIQUE | plugin_exact | 520 | 360 | 321 | 240 | 360 |
| CAsT | per_pair | n.r. | n.r. | n.r. | n.r. | n.r. |
| CAsT | static_sum | 2,561 | 2,205 | 1,665 | 879 | 1,550 |
| CAsT | oracle_exact | 2,176 | 1,773 | 1,454 | 780 | 1,375 |
| CAsT | plugin_exact | 2,246 | 1,824 | 1,424 | 765 | 1,493 |
| DBpedia | per_pair | 4,498 | 4,277 | 2,844 | 2,160 | 2,788 |
| DBpedia | static_sum | 1,095 | 1,155 | 744 | 636 | 756 |
| DBpedia | oracle_exact | 1,215 | 1,152 | 717 | 845 | 796 |
| DBpedia | plugin_exact | 1,263 | 1,434 | 867 | 806 | 800 |
| DL 21-23 | per_pair | 2,998 | 1,848 | 1,365 | 802 | 1,668 |
| DL 21-23 | static_sum | 1,056 | 717 | 509 | 325 | 670 |
| DL 21-23 | oracle_exact | 972 | 568 | 417 | 305 | 525 |
| DL 21-23 | plugin_exact | 1,062 | 665 | 476 | 305 | 608 |

Max wrong-certificate rate over all cells at eps=0.02: 0.003

## eps = 0.03

### 1. Candidate quality vs pilot size (candidate chosen on the pilot; identical across arms)

| Collection | n_train | P(regret <= eps) | median slack | slack q25 | pilot docs |
|---|---:|---:|---:|---:|---:|
| ANTIQUE | 10 | 0.79 | 0.030 | 0.030 | 315 |
| ANTIQUE | 20 | 0.87 | 0.030 | 0.030 | 631 |
| ANTIQUE | 40 | 0.96 | 0.030 | 0.030 | 1,268 |
| ANTIQUE | 80 | 1.00 | 0.030 | 0.030 | 2,530 |
| CAsT | 10 | 0.91 | 0.029 | 0.022 | 491 |
| CAsT | 20 | 0.98 | 0.030 | 0.026 | 972 |
| CAsT | 40 | 1.00 | 0.030 | 0.029 | 1,955 |
| CAsT | 80 | 1.00 | 0.030 | 0.029 | 3,906 |
| DBpedia | 10 | 0.92 | 0.030 | 0.028 | 513 |
| DBpedia | 20 | 0.95 | 0.030 | 0.030 | 1,030 |
| DBpedia | 40 | 0.99 | 0.030 | 0.030 | 2,059 |
| DBpedia | 80 | 1.00 | 0.030 | 0.030 | 4,115 |
| DL 21-23 | 10 | 0.77 | 0.030 | 0.030 | 692 |
| DL 21-23 | 20 | 0.90 | 0.030 | 0.030 | 1,389 |
| DL 21-23 | 40 | 0.96 | 0.030 | 0.030 | 2,773 |
| DL 21-23 | 80 | 1.00 | 0.030 | 0.030 | 5,555 |

### 2. Certification given a feasible candidate (max over budgets of ACT | regret <= eps)

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 0.75 | 0.83 | 0.89 | 0.99 | - |
| ANTIQUE | static_sum | 0.97 | 0.98 | 0.99 | 1.00 | - |
| ANTIQUE | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | - |
| ANTIQUE | plugin_exact | 0.99 | 0.98 | 1.00 | 1.00 | - |
| CAsT | per_pair | 0.19 | 0.21 | 0.20 | 0.33 | - |
| CAsT | static_sum | 0.99 | 1.00 | 1.00 | 1.00 | - |
| CAsT | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | - |
| CAsT | plugin_exact | 1.00 | 1.00 | 1.00 | 1.00 | - |
| DBpedia | per_pair | 0.69 | 0.74 | 0.77 | 0.80 | - |
| DBpedia | static_sum | 0.88 | 0.93 | 0.97 | 0.99 | - |
| DBpedia | oracle_exact | 0.89 | 0.93 | 0.97 | 0.99 | - |
| DBpedia | plugin_exact | 0.90 | 0.93 | 0.97 | 1.00 | - |
| DL 21-23 | per_pair | 0.89 | 0.97 | 0.99 | 1.00 | - |
| DL 21-23 | static_sum | 0.97 | 0.99 | 1.00 | 1.00 | - |
| DL 21-23 | oracle_exact | 1.00 | 1.00 | 1.00 | 1.00 | - |
| DL 21-23 | plugin_exact | 0.97 | 0.99 | 1.00 | 1.00 | - |

### 3. Total unique documents (pilot included) to reach ACT 0.5

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 1,572 | 1,683 | 1,805 | 2,845 | - |
| ANTIQUE | static_sum | 755 | 991 | 1,588 | 2,769 | - |
| ANTIQUE | oracle_exact | 696 | 992 | 1,588 | 2,769 | - |
| ANTIQUE | plugin_exact | 753 | 990 | 1,589 | 2,770 | - |
| CAsT | per_pair | n.r. | n.r. | n.r. | n.r. | - |
| CAsT | static_sum | 2,117 | 2,217 | 2,991 | 4,436 | - |
| CAsT | oracle_exact | 1,881 | 2,164 | 2,848 | 4,382 | - |
| CAsT | plugin_exact | 2,066 | 2,137 | 2,890 | 4,385 | - |
| DBpedia | per_pair | 3,155 | 2,771 | 3,750 | 5,615 | - |
| DBpedia | static_sum | 1,376 | 1,789 | 2,778 | 4,754 | - |
| DBpedia | oracle_exact | 1,402 | 1,789 | 2,776 | 4,754 | - |
| DBpedia | plugin_exact | 1,455 | 1,786 | 2,778 | 4,752 | - |
| DL 21-23 | per_pair | 2,724 | 2,816 | 3,835 | 6,192 | - |
| DL 21-23 | static_sum | 1,428 | 1,920 | 3,212 | 5,860 | - |
| DL 21-23 | oracle_exact | 1,342 | 1,848 | 3,115 | 5,861 | - |
| DL 21-23 | plugin_exact | 1,404 | 1,875 | 3,145 | 5,861 | - |

### 3b. Sampling documents only (pilot excluded) to reach ACT 0.5

| Collection | arm | pilot 10 | pilot 20 | pilot 40 | pilot 80 | fixed best (pilot 20) |
|---|---|---:|---:|---:|---:|---:|
| ANTIQUE | per_pair | 1,257 | 1,052 | 536 | 315 | - |
| ANTIQUE | static_sum | 440 | 361 | 320 | 239 | - |
| ANTIQUE | oracle_exact | 381 | 361 | 320 | 239 | - |
| ANTIQUE | plugin_exact | 438 | 359 | 320 | 240 | - |
| CAsT | per_pair | n.r. | n.r. | n.r. | n.r. | - |
| CAsT | static_sum | 1,627 | 1,245 | 1,037 | 530 | - |
| CAsT | oracle_exact | 1,390 | 1,192 | 893 | 477 | - |
| CAsT | plugin_exact | 1,575 | 1,165 | 936 | 479 | - |
| DBpedia | per_pair | 2,642 | 1,741 | 1,691 | 1,500 | - |
| DBpedia | static_sum | 863 | 759 | 719 | 639 | - |
| DBpedia | oracle_exact | 889 | 759 | 717 | 639 | - |
| DBpedia | plugin_exact | 943 | 755 | 719 | 637 | - |
| DL 21-23 | per_pair | 2,031 | 1,427 | 1,062 | 637 | - |
| DL 21-23 | static_sum | 736 | 531 | 439 | 305 | - |
| DL 21-23 | oracle_exact | 650 | 459 | 342 | 306 | - |
| DL 21-23 | plugin_exact | 712 | 485 | 373 | 306 | - |

Max wrong-certificate rate over all cells at eps=0.03: 0.003

## Held-out check (eps = 0.03), predictions P1-P3 of REVIEW_RESPONSE §14

| Collection | arm | J50 p10/p20 | J50 p40/p20 | J50 p80/p40 | ACT|feasible min over pilots | oracle saving vs static_sum (pilot 20) | P1 | P2 | P3 |
|---|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|
| ANTIQUE | static_sum | 0.76 | 1.60 | 1.74 | 0.97 | -0.0% | NO | yes | yes |
| ANTIQUE | oracle_exact | 0.70 | 1.60 | 1.74 | 1.00 | -0.0% | NO | yes | yes |
| ANTIQUE | plugin_exact | 0.76 | 1.61 | 1.74 | 0.98 | -0.0% | NO | yes | yes |
| CAsT | static_sum | 0.95 | 1.35 | 1.48 | 0.99 | +2.4% | yes | yes | yes |
| CAsT | oracle_exact | 0.87 | 1.32 | 1.54 | 1.00 | +2.4% | yes | yes | yes |
| CAsT | plugin_exact | 0.97 | 1.35 | 1.52 | 1.00 | +2.4% | yes | yes | yes |
| DBpedia | static_sum | 0.77 | 1.55 | 1.71 | 0.88 | +0.0% | NO | - | yes |
| DBpedia | oracle_exact | 0.78 | 1.55 | 1.71 | 0.89 | +0.0% | NO | - | yes |
| DBpedia | plugin_exact | 0.82 | 1.56 | 1.71 | 0.90 | +0.0% | NO | - | yes |
| DL 21-23 | static_sum | 0.74 | 1.67 | 1.82 | 0.97 | +3.8% | NO | yes | yes |
| DL 21-23 | oracle_exact | 0.73 | 1.69 | 1.88 | 1.00 | +3.8% | NO | yes | yes |
| DL 21-23 | plugin_exact | 0.75 | 1.68 | 1.86 | 0.97 | +3.8% | NO | yes | yes |
