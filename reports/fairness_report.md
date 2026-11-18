# Fairness audit tables

Written by `scripts/audit_fairness.py`. Full precision in `fairness_groups.csv` and
`fairness_intersectional.csv`. Ranges are 95% bootstrap intervals (1000 resamples).
Recall and precision are for the top-20% list of each module presentation. Calibration
gap = mean predicted risk - observed at-risk rate. small = fewer than 100 students or 30 at-risk students.

## gbm minus lr_main, recall@top-20%, all students

| Week | gbm | lr_main | Difference [95% range] | Share of resamples gbm ahead |
|---|---|---|---|---|
| 2 | 0.3164 | 0.3092 | 0.0072 [-0.0003, 0.0154] | 0.972 |
| 4 | 0.3619 | 0.3550 | 0.0069 [0.0006, 0.0150] | 0.986 |
| 8 | 0.4155 | 0.4135 | 0.0020 [-0.0030, 0.0097] | 0.825 |
| 12 | 0.4454 | 0.4496 | -0.0042 [-0.0097, 0.0042] | 0.231 |

## Week 2

### all (reference: All students)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| All students | 9333 | 0.416 | 0.309 [0.299, 0.318] | 0.316 [0.306, 0.327] | 0.323 [0.314, 0.334] | 0.643 [0.619, 0.663] | 0.062 [0.052, 0.072] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |

### gender (reference: M)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| F | 3908 | 0.393 | 0.285 [0.263, 0.304] | 0.288 [0.269, 0.310] | 0.291 [0.273, 0.314] | 0.580 [0.541, 0.611] | 0.071 [0.057, 0.086] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| M | 5425 | 0.434 | 0.325 [0.311, 0.339] | 0.335 [0.320, 0.350] | 0.344 [0.329, 0.358] | 0.685 [0.658, 0.713] | 0.056 [0.043, 0.069] | lr_main: calibration; lr_with_demographics: calibration |

### age_band (reference: 0-35)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-35 | 6404 | 0.431 | 0.323 [0.309, 0.334] | 0.336 [0.323, 0.350] | 0.350 [0.337, 0.365] | 0.653 [0.625, 0.676] | 0.065 [0.053, 0.076] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| 35-55 | 2840 | 0.385 | 0.278 [0.254, 0.304] | 0.271 [0.246, 0.293] | 0.263 [0.237, 0.286] | 0.614 [0.570, 0.659] | 0.058 [0.042, 0.076] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| 55<= (small) | 89 | 0.371 | 0.212 [0.080, 0.382] | 0.152 [0.048, 0.310] | 0.091 [0.000, 0.189] | 0.636 [0.333, 0.933] | 0.016 [-0.078, 0.111] |  |

### region (reference: Scotland)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| East Anglian Region | 858 | 0.379 | 0.258 [0.215, 0.313] | 0.317 [0.266, 0.371] | 0.258 [0.209, 0.311] | 0.575 [0.496, 0.662] | 0.088 [0.057, 0.119] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| East Midlands Region | 646 | 0.399 | 0.264 [0.210, 0.316] | 0.271 [0.222, 0.328] | 0.322 [0.260, 0.370] | 0.607 [0.516, 0.691] | 0.068 [0.033, 0.106] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| Ireland | 416 | 0.399 | 0.343 [0.277, 0.432] | 0.386 [0.310, 0.462] | 0.319 [0.246, 0.394] | 0.770 [0.621, 0.857] | 0.093 [0.049, 0.137] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| London Region | 822 | 0.428 | 0.344 [0.298, 0.395] | 0.330 [0.290, 0.381] | 0.355 [0.316, 0.412] | 0.676 [0.611, 0.749] | 0.061 [0.029, 0.094] | lr_main: calibration; lr_with_demographics: calibration |
| North Region | 531 | 0.379 | 0.279 [0.213, 0.341] | 0.269 [0.204, 0.322] | 0.209 [0.156, 0.270] | 0.615 [0.500, 0.702] | 0.089 [0.049, 0.127] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| North Western Region | 765 | 0.465 | 0.329 [0.277, 0.378] | 0.331 [0.289, 0.386] | 0.497 [0.444, 0.547] | 0.696 [0.620, 0.765] | 0.019 [-0.015, 0.051] | lr_with_demographics: recall gap; calibration |
| Scotland | 1298 | 0.446 | 0.325 [0.287, 0.363] | 0.318 [0.282, 0.351] | 0.325 [0.290, 0.363] | 0.655 [0.594, 0.708] | 0.043 [0.017, 0.069] |  |
| South East Region | 607 | 0.390 | 0.304 [0.241, 0.361] | 0.291 [0.233, 0.347] | 0.203 [0.153, 0.259] | 0.600 [0.512, 0.686] | 0.083 [0.045, 0.120] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| South Region | 857 | 0.372 | 0.301 [0.254, 0.352] | 0.288 [0.238, 0.340] | 0.223 [0.179, 0.267] | 0.585 [0.517, 0.665] | 0.101 [0.066, 0.131] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| South West Region | 682 | 0.406 | 0.314 [0.258, 0.368] | 0.318 [0.265, 0.373] | 0.274 [0.221, 0.324] | 0.621 [0.534, 0.702] | 0.067 [0.031, 0.103] | lr_main: calibration; gbm: calibration |
| Wales | 679 | 0.476 | 0.325 [0.264, 0.375] | 0.334 [0.287, 0.389] | 0.424 [0.368, 0.476] | 0.682 [0.607, 0.762] | 0.011 [-0.024, 0.048] | lr_with_demographics: recall gap; calibration |
| West Midlands Region | 650 | 0.438 | 0.302 [0.249, 0.353] | 0.333 [0.278, 0.381] | 0.333 [0.279, 0.393] | 0.628 [0.544, 0.705] | 0.042 [0.005, 0.077] | lr_with_demographics: calibration |
| Yorkshire Region | 522 | 0.400 | 0.311 [0.249, 0.380] | 0.330 [0.263, 0.396] | 0.364 [0.298, 0.428] | 0.663 [0.568, 0.762] | 0.071 [0.032, 0.108] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |

### highest_education (reference: A Level or Equivalent)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| A Level or Equivalent | 4142 | 0.384 | 0.324 [0.305, 0.346] | 0.326 [0.308, 0.347] | 0.229 [0.211, 0.246] | 0.628 [0.596, 0.659] | 0.102 [0.088, 0.116] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| HE Qualification | 1575 | 0.345 | 0.278 [0.245, 0.313] | 0.271 [0.236, 0.308] | 0.160 [0.131, 0.193] | 0.514 [0.464, 0.575] | 0.129 [0.106, 0.151] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| Lower Than A Level | 3434 | 0.490 | 0.302 [0.281, 0.320] | 0.318 [0.298, 0.337] | 0.462 [0.443, 0.485] | 0.714 [0.675, 0.743] | -0.020 [-0.036, -0.003] | lr_with_demographics: recall gap; calibration |
| No Formal quals (small) | 69 | 0.551 | 0.421 [0.270, 0.571] | 0.447 [0.297, 0.615] | 0.632 [0.467, 0.781] | 0.842 [0.636, 1.000] | -0.047 [-0.154, 0.057] |  |
| Post Graduate Qualification | 113 | 0.292 | 0.364 [0.189, 0.539] | 0.364 [0.192, 0.531] | 0.061 [0.000, 0.184] | 0.462 [0.263, 0.667] | 0.228 [0.140, 0.311] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |

### imd_band (reference: 20-30%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-10% | 828 | 0.499 | 0.293 [0.248, 0.332] | 0.288 [0.249, 0.330] | 0.501 [0.455, 0.547] | 0.720 [0.645, 0.787] | -0.018 [-0.050, 0.015] | lr_with_demographics: calibration |
| 10-20 | 924 | 0.484 | 0.313 [0.276, 0.359] | 0.340 [0.299, 0.384] | 0.443 [0.403, 0.492] | 0.725 [0.667, 0.788] | 0.002 [-0.030, 0.032] | lr_with_demographics: calibration |
| 20-30% | 995 | 0.472 | 0.317 [0.277, 0.356] | 0.336 [0.298, 0.378] | 0.451 [0.409, 0.495] | 0.668 [0.598, 0.728] | 0.014 [-0.017, 0.044] | lr_with_demographics: calibration |
| 30-40% | 993 | 0.425 | 0.327 [0.283, 0.375] | 0.339 [0.293, 0.388] | 0.346 [0.297, 0.386] | 0.683 [0.616, 0.743] | 0.057 [0.029, 0.089] | lr_main: calibration; lr_with_demographics: recall gap; calibration |
| 40-50% | 935 | 0.417 | 0.315 [0.269, 0.359] | 0.326 [0.283, 0.378] | 0.292 [0.249, 0.341] | 0.612 [0.546, 0.681] | 0.064 [0.033, 0.097] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| 50-60% | 929 | 0.410 | 0.315 [0.270, 0.362] | 0.344 [0.296, 0.385] | 0.294 [0.249, 0.337] | 0.628 [0.555, 0.692] | 0.066 [0.035, 0.095] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| 60-70% | 906 | 0.391 | 0.280 [0.234, 0.327] | 0.274 [0.228, 0.321] | 0.215 [0.172, 0.258] | 0.593 [0.512, 0.659] | 0.082 [0.053, 0.113] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| 70-80% | 880 | 0.395 | 0.342 [0.289, 0.386] | 0.330 [0.281, 0.376] | 0.279 [0.234, 0.332] | 0.626 [0.558, 0.701] | 0.078 [0.050, 0.107] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| 80-90% | 816 | 0.360 | 0.272 [0.224, 0.326] | 0.276 [0.224, 0.324] | 0.170 [0.130, 0.215] | 0.516 [0.437, 0.594] | 0.123 [0.093, 0.156] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| 90-100% | 758 | 0.342 | 0.305 [0.245, 0.357] | 0.301 [0.243, 0.356] | 0.147 [0.102, 0.190] | 0.627 [0.534, 0.703] | 0.127 [0.095, 0.157] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| Unknown | 369 | 0.295 | 0.312 [0.220, 0.402] | 0.266 [0.181, 0.345] | 0.046 [0.010, 0.097] | 0.630 [0.500, 0.765] | 0.165 [0.122, 0.210] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |

### disability (reference: N)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| N | 8464 | 0.404 | 0.307 [0.296, 0.317] | 0.313 [0.301, 0.325] | 0.302 [0.291, 0.314] | 0.630 [0.602, 0.653] | 0.073 [0.063, 0.084] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| Y | 869 | 0.535 | 0.327 [0.285, 0.365] | 0.342 [0.298, 0.383] | 0.475 [0.436, 0.527] | 0.749 [0.686, 0.805] | -0.042 [-0.072, -0.011] | lr_with_demographics: recall gap |

### disability_x_imd (reference: disability N, IMD 30-70%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| disability N, IMD 0-30% | 2407 | 0.466 | 0.307 [0.282, 0.331] | 0.326 [0.304, 0.352] | 0.440 [0.414, 0.469] | 0.687 [0.646, 0.727] | 0.016 [-0.003, 0.036] | lr_with_demographics: recall gap; calibration |
| disability N, IMD 30-70% | 3419 | 0.402 | 0.304 [0.284, 0.328] | 0.313 [0.291, 0.336] | 0.273 [0.250, 0.294] | 0.620 [0.582, 0.658] | 0.075 [0.060, 0.092] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| disability N, IMD 70-100% | 2285 | 0.360 | 0.309 [0.278, 0.335] | 0.300 [0.270, 0.331] | 0.194 [0.171, 0.222] | 0.580 [0.530, 0.623] | 0.115 [0.097, 0.133] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| disability N, IMD Unknown | 353 | 0.292 | 0.330 [0.232, 0.423] | 0.272 [0.183, 0.352] | 0.049 [0.011, 0.103] | 0.630 [0.500, 0.765] | 0.171 [0.124, 0.215] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| disability Y, IMD 0-30% | 340 | 0.612 | 0.317 [0.254, 0.377] | 0.303 [0.242, 0.362] | 0.591 [0.539, 0.667] | 0.795 [0.693, 0.878] | -0.112 [-0.161, -0.060] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| disability Y, IMD 30-70% | 344 | 0.503 | 0.358 [0.281, 0.431] | 0.393 [0.326, 0.478] | 0.422 [0.352, 0.509] | 0.713 [0.613, 0.800] | -0.008 [-0.058, 0.042] | gbm: recall gap; lr_with_demographics: recall gap |
| disability Y, IMD 70-100% | 169 | 0.462 | 0.308 [0.208, 0.423] | 0.346 [0.222, 0.455] | 0.321 [0.222, 0.427] | 0.727 [0.581, 0.875] | 0.020 [-0.056, 0.087] |  |
| disability Y, IMD Unknown (small) | 16 | n/a | n/a | n/a | n/a | n/a | n/a |  |

## Week 4

### all (reference: All students)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| All students | 9214 | 0.408 | 0.355 [0.345, 0.364] | 0.362 [0.352, 0.372] | 0.359 [0.349, 0.369] | 0.723 [0.702, 0.740] | 0.034 [0.024, 0.043] |  |

### gender (reference: M)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| F | 3873 | 0.385 | 0.312 [0.289, 0.332] | 0.319 [0.301, 0.341] | 0.316 [0.295, 0.337] | 0.632 [0.593, 0.664] | 0.058 [0.040, 0.073] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| M | 5341 | 0.425 | 0.383 [0.369, 0.398] | 0.390 [0.374, 0.403] | 0.388 [0.373, 0.402] | 0.783 [0.757, 0.806] | 0.016 [0.003, 0.027] |  |

### age_band (reference: 0-35)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-35 | 6312 | 0.422 | 0.370 [0.356, 0.383] | 0.379 [0.365, 0.393] | 0.389 [0.375, 0.401] | 0.730 [0.706, 0.750] | 0.037 [0.025, 0.049] |  |
| 35-55 | 2815 | 0.377 | 0.318 [0.291, 0.343] | 0.320 [0.295, 0.345] | 0.291 [0.267, 0.316] | 0.701 [0.658, 0.744] | 0.028 [0.011, 0.044] | lr_with_demographics: recall gap |
| 55<= (small) | 87 | 0.356 | 0.290 [0.146, 0.467] | 0.355 [0.195, 0.543] | 0.161 [0.042, 0.321] | 0.750 [0.462, 1.000] | -0.050 [-0.141, 0.038] |  |

### region (reference: Scotland)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| East Anglian Region | 850 | 0.373 | 0.322 [0.268, 0.368] | 0.338 [0.285, 0.385] | 0.309 [0.265, 0.365] | 0.685 [0.607, 0.760] | 0.060 [0.027, 0.092] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| East Midlands Region | 630 | 0.384 | 0.322 [0.266, 0.391] | 0.343 [0.281, 0.402] | 0.351 [0.286, 0.406] | 0.743 [0.653, 0.833] | 0.040 [0.004, 0.074] | lr_with_demographics: calibration |
| Ireland | 421 | 0.387 | 0.301 [0.236, 0.377] | 0.344 [0.273, 0.418] | 0.276 [0.218, 0.351] | 0.778 [0.667, 0.868] | 0.001 [-0.042, 0.043] | lr_with_demographics: recall gap |
| London Region | 808 | 0.418 | 0.382 [0.329, 0.428] | 0.396 [0.349, 0.449] | 0.349 [0.297, 0.399] | 0.725 [0.659, 0.790] | 0.026 [-0.004, 0.056] |  |
| North Region | 524 | 0.368 | 0.347 [0.277, 0.417] | 0.332 [0.267, 0.401] | 0.321 [0.250, 0.378] | 0.698 [0.600, 0.791] | 0.061 [0.024, 0.104] | lr_main: calibration; gbm: calibration |
| North Western Region | 753 | 0.457 | 0.392 [0.341, 0.439] | 0.392 [0.346, 0.443] | 0.477 [0.428, 0.531] | 0.794 [0.719, 0.850] | 0.010 [-0.021, 0.041] | lr_with_demographics: recall gap; calibration |
| Scotland | 1292 | 0.443 | 0.349 [0.317, 0.390] | 0.370 [0.329, 0.403] | 0.370 [0.327, 0.405] | 0.763 [0.711, 0.817] | -0.003 [-0.027, 0.021] |  |
| South East Region | 600 | 0.382 | 0.293 [0.227, 0.347] | 0.323 [0.265, 0.389] | 0.258 [0.207, 0.321] | 0.644 [0.542, 0.728] | 0.049 [0.010, 0.089] | gbm: calibration; lr_with_demographics: recall gap |
| South Region | 841 | 0.360 | 0.396 [0.339, 0.450] | 0.356 [0.306, 0.414] | 0.310 [0.258, 0.359] | 0.642 [0.561, 0.708] | 0.099 [0.066, 0.129] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| South West Region | 672 | 0.397 | 0.371 [0.312, 0.428] | 0.363 [0.306, 0.419] | 0.333 [0.278, 0.389] | 0.723 [0.650, 0.798] | 0.042 [0.005, 0.074] |  |
| Wales | 665 | 0.465 | 0.330 [0.279, 0.380] | 0.343 [0.290, 0.397] | 0.430 [0.381, 0.484] | 0.703 [0.626, 0.769] | 0.000 [-0.039, 0.039] |  |
| West Midlands Region | 643 | 0.432 | 0.399 [0.343, 0.452] | 0.388 [0.333, 0.444] | 0.396 [0.338, 0.453] | 0.782 [0.715, 0.856] | 0.008 [-0.027, 0.044] |  |
| Yorkshire Region | 515 | 0.392 | 0.371 [0.303, 0.435] | 0.376 [0.307, 0.438] | 0.401 [0.335, 0.467] | 0.694 [0.598, 0.781] | 0.059 [0.021, 0.099] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |

### highest_education (reference: A Level or Equivalent)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| A Level or Equivalent | 4085 | 0.374 | 0.368 [0.345, 0.387] | 0.353 [0.333, 0.377] | 0.301 [0.281, 0.321] | 0.706 [0.672, 0.735] | 0.065 [0.049, 0.078] | lr_main: calibration; gbm: calibration |
| HE Qualification | 1559 | 0.337 | 0.316 [0.277, 0.355] | 0.335 [0.294, 0.371] | 0.230 [0.196, 0.266] | 0.634 [0.575, 0.692] | 0.066 [0.043, 0.087] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| Lower Than A Level | 3392 | 0.483 | 0.355 [0.334, 0.377] | 0.381 [0.358, 0.401] | 0.452 [0.430, 0.473] | 0.777 [0.747, 0.808] | -0.020 [-0.036, -0.004] | lr_with_demographics: recall gap |
| No Formal quals (small) | 67 | 0.537 | 0.500 [0.324, 0.655] | 0.389 [0.229, 0.559] | 0.694 [0.525, 0.848] | 0.750 [0.571, 0.929] | -0.005 [-0.114, 0.119] |  |
| Post Graduate Qualification | 111 | 0.279 | 0.194 [0.062, 0.346] | 0.226 [0.069, 0.385] | 0.129 [0.000, 0.250] | 0.429 [0.154, 0.688] | 0.093 [0.012, 0.178] | lr_main: recall gap; calibration; gbm: calibration; lr_with_demographics: recall gap |

### imd_band (reference: 30-40%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-10% | 820 | 0.493 | 0.324 [0.286, 0.371] | 0.356 [0.313, 0.403] | 0.438 [0.393, 0.487] | 0.789 [0.717, 0.852] | -0.035 [-0.070, -0.001] |  |
| 10-20 | 912 | 0.476 | 0.373 [0.332, 0.422] | 0.389 [0.347, 0.436] | 0.438 [0.393, 0.487] | 0.806 [0.756, 0.862] | -0.007 [-0.036, 0.020] |  |
| 20-30% | 973 | 0.459 | 0.378 [0.336, 0.419] | 0.380 [0.336, 0.425] | 0.445 [0.397, 0.491] | 0.772 [0.715, 0.825] | -0.012 [-0.041, 0.017] | lr_with_demographics: recall gap |
| 30-40% | 980 | 0.416 | 0.375 [0.329, 0.421] | 0.363 [0.323, 0.414] | 0.355 [0.313, 0.408] | 0.743 [0.672, 0.797] | 0.039 [0.007, 0.069] |  |
| 40-50% | 925 | 0.410 | 0.364 [0.312, 0.408] | 0.367 [0.319, 0.413] | 0.356 [0.313, 0.408] | 0.697 [0.631, 0.760] | 0.042 [0.010, 0.074] | gbm: calibration |
| 50-60% | 921 | 0.404 | 0.392 [0.337, 0.429] | 0.398 [0.346, 0.444] | 0.382 [0.324, 0.420] | 0.709 [0.652, 0.777] | 0.043 [0.008, 0.075] |  |
| 60-70% | 887 | 0.377 | 0.329 [0.280, 0.379] | 0.332 [0.281, 0.383] | 0.266 [0.221, 0.320] | 0.663 [0.583, 0.731] | 0.049 [0.020, 0.079] | gbm: calibration; lr_with_demographics: recall gap |
| 70-80% | 873 | 0.389 | 0.376 [0.326, 0.429] | 0.391 [0.333, 0.434] | 0.356 [0.304, 0.398] | 0.753 [0.680, 0.817] | 0.047 [0.016, 0.077] | gbm: calibration |
| 80-90% | 810 | 0.354 | 0.300 [0.244, 0.346] | 0.307 [0.254, 0.359] | 0.268 [0.214, 0.317] | 0.610 [0.520, 0.687] | 0.085 [0.054, 0.116] | lr_main: recall gap; calibration; gbm: calibration; lr_with_demographics: recall gap; calibration |
| 90-100% | 747 | 0.332 | 0.335 [0.268, 0.391] | 0.335 [0.273, 0.390] | 0.246 [0.195, 0.307] | 0.654 [0.569, 0.732] | 0.076 [0.045, 0.105] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| Unknown | 366 | 0.287 | 0.267 [0.186, 0.361] | 0.257 [0.175, 0.343] | 0.133 [0.069, 0.195] | 0.609 [0.460, 0.744] | 0.081 [0.039, 0.127] | lr_main: recall gap; calibration; gbm: recall gap; calibration; lr_with_demographics: recall gap |

### disability (reference: N)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| N | 8359 | 0.396 | 0.350 [0.338, 0.360] | 0.360 [0.348, 0.372] | 0.344 [0.332, 0.355] | 0.710 [0.687, 0.728] | 0.042 [0.032, 0.052] |  |
| Y | 855 | 0.527 | 0.390 [0.349, 0.438] | 0.373 [0.330, 0.417] | 0.472 [0.429, 0.522] | 0.819 [0.768, 0.867] | -0.048 [-0.082, -0.017] | lr_with_demographics: recall gap |

### disability_x_imd (reference: disability N, IMD 30-70%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| disability N, IMD 0-30% | 2375 | 0.458 | 0.356 [0.333, 0.383] | 0.376 [0.351, 0.403] | 0.425 [0.398, 0.452] | 0.776 [0.739, 0.810] | -0.003 [-0.021, 0.016] | lr_with_demographics: recall gap |
| disability N, IMD 30-70% | 3373 | 0.393 | 0.359 [0.332, 0.379] | 0.363 [0.342, 0.388] | 0.326 [0.303, 0.351] | 0.695 [0.660, 0.728] | 0.049 [0.032, 0.066] | gbm: calibration |
| disability N, IMD 70-100% | 2261 | 0.352 | 0.336 [0.305, 0.368] | 0.345 [0.309, 0.373] | 0.287 [0.258, 0.318] | 0.665 [0.614, 0.709] | 0.072 [0.053, 0.090] | lr_main: calibration; gbm: calibration |
| disability N, IMD Unknown | 350 | 0.283 | 0.283 [0.200, 0.379] | 0.273 [0.189, 0.360] | 0.141 [0.075, 0.209] | 0.609 [0.460, 0.744] | 0.088 [0.043, 0.136] | lr_main: calibration; gbm: recall gap; calibration; lr_with_demographics: recall gap |
| disability Y, IMD 0-30% | 330 | 0.600 | 0.379 [0.314, 0.451] | 0.374 [0.310, 0.438] | 0.525 [0.457, 0.597] | 0.862 [0.780, 0.936] | -0.119 [-0.172, -0.068] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| disability Y, IMD 30-70% | 340 | 0.497 | 0.426 [0.357, 0.503] | 0.385 [0.311, 0.457] | 0.467 [0.397, 0.543] | 0.774 [0.685, 0.857] | -0.009 [-0.059, 0.041] | lr_with_demographics: recall gap |
| disability Y, IMD 70-100% | 169 | 0.462 | 0.372 [0.262, 0.482] | 0.372 [0.267, 0.500] | 0.385 [0.268, 0.500] | 0.829 [0.683, 0.946] | 0.014 [-0.059, 0.084] |  |
| disability Y, IMD Unknown (small) | 16 | n/a | n/a | n/a | n/a | n/a | n/a |  |

## Week 8

### all (reference: All students)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| All students | 8879 | 0.385 | 0.414 [0.402, 0.423] | 0.416 [0.405, 0.427] | 0.417 [0.406, 0.428] | 0.796 [0.776, 0.813] | 0.011 [0.002, 0.019] |  |

### gender (reference: M)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| F | 3750 | 0.364 | 0.389 [0.366, 0.408] | 0.388 [0.366, 0.408] | 0.388 [0.366, 0.410] | 0.724 [0.688, 0.756] | 0.029 [0.015, 0.042] | gbm: calibration |
| M | 5129 | 0.401 | 0.429 [0.414, 0.446] | 0.434 [0.418, 0.451] | 0.436 [0.420, 0.452] | 0.847 [0.822, 0.868] | -0.003 [-0.014, 0.009] |  |

### age_band (reference: 0-35)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-35 | 6075 | 0.400 | 0.433 [0.418, 0.448] | 0.437 [0.423, 0.454] | 0.444 [0.429, 0.460] | 0.802 [0.781, 0.824] | 0.014 [0.004, 0.025] |  |
| 35-55 | 2722 | 0.356 | 0.370 [0.338, 0.397] | 0.368 [0.337, 0.395] | 0.356 [0.326, 0.382] | 0.782 [0.736, 0.815] | 0.004 [-0.012, 0.018] |  |
| 55<= (small) | 82 | 0.317 | 0.192 [0.048, 0.360] | 0.231 [0.087, 0.429] | 0.192 [0.045, 0.357] | 0.625 [0.250, 1.000] | -0.038 [-0.134, 0.052] |  |

### region (reference: Scotland)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| East Anglian Region | 822 | 0.350 | 0.358 [0.300, 0.412] | 0.340 [0.281, 0.394] | 0.347 [0.292, 0.401] | 0.746 [0.664, 0.819] | 0.026 [-0.007, 0.057] | gbm: recall gap; calibration; lr_with_demographics: recall gap |
| East Midlands Region | 609 | 0.363 | 0.394 [0.332, 0.460] | 0.398 [0.338, 0.467] | 0.434 [0.364, 0.500] | 0.791 [0.717, 0.879] | 0.021 [-0.012, 0.053] |  |
| Ireland | 408 | 0.368 | 0.367 [0.291, 0.450] | 0.327 [0.249, 0.407] | 0.340 [0.260, 0.417] | 0.821 [0.733, 0.918] | -0.016 [-0.058, 0.023] | gbm: recall gap; lr_with_demographics: recall gap |
| London Region | 787 | 0.403 | 0.438 [0.378, 0.491] | 0.429 [0.382, 0.492] | 0.416 [0.359, 0.475] | 0.808 [0.747, 0.870] | 0.001 [-0.027, 0.029] |  |
| North Region | 508 | 0.348 | 0.395 [0.320, 0.471] | 0.379 [0.312, 0.462] | 0.350 [0.286, 0.436] | 0.787 [0.699, 0.877] | 0.021 [-0.017, 0.058] | lr_with_demographics: recall gap |
| North Western Region | 710 | 0.424 | 0.468 [0.408, 0.518] | 0.475 [0.413, 0.523] | 0.522 [0.463, 0.576] | 0.829 [0.754, 0.880] | 0.003 [-0.028, 0.036] |  |
| Scotland | 1263 | 0.431 | 0.430 [0.393, 0.475] | 0.450 [0.415, 0.493] | 0.445 [0.407, 0.487] | 0.836 [0.795, 0.884] | -0.025 [-0.047, -0.002] |  |
| South East Region | 565 | 0.343 | 0.392 [0.321, 0.465] | 0.412 [0.337, 0.481] | 0.330 [0.266, 0.399] | 0.697 [0.605, 0.786] | 0.052 [0.018, 0.090] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| South Region | 809 | 0.335 | 0.384 [0.333, 0.448] | 0.369 [0.317, 0.435] | 0.362 [0.305, 0.417] | 0.748 [0.675, 0.818] | 0.060 [0.031, 0.088] | lr_main: calibration; gbm: calibration |
| South West Region | 654 | 0.381 | 0.454 [0.386, 0.509] | 0.478 [0.409, 0.529] | 0.418 [0.357, 0.476] | 0.819 [0.741, 0.880] | 0.009 [-0.021, 0.039] |  |
| Wales | 636 | 0.440 | 0.404 [0.343, 0.457] | 0.400 [0.337, 0.453] | 0.461 [0.404, 0.517] | 0.801 [0.727, 0.861] | -0.020 [-0.059, 0.017] |  |
| West Midlands Region | 612 | 0.404 | 0.413 [0.352, 0.467] | 0.417 [0.360, 0.481] | 0.437 [0.373, 0.492] | 0.810 [0.728, 0.867] | -0.006 [-0.040, 0.027] |  |
| Yorkshire Region | 496 | 0.369 | 0.426 [0.360, 0.500] | 0.448 [0.380, 0.529] | 0.459 [0.392, 0.535] | 0.788 [0.691, 0.860] | 0.032 [-0.004, 0.068] | gbm: calibration |

### highest_education (reference: A Level or Equivalent)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| A Level or Equivalent | 3950 | 0.353 | 0.423 [0.398, 0.445] | 0.423 [0.398, 0.446] | 0.381 [0.355, 0.403] | 0.774 [0.742, 0.801] | 0.036 [0.023, 0.049] | gbm: calibration |
| HE Qualification | 1516 | 0.318 | 0.338 [0.305, 0.379] | 0.349 [0.312, 0.392] | 0.295 [0.260, 0.335] | 0.738 [0.680, 0.799] | 0.028 [0.008, 0.047] | lr_main: recall gap; lr_with_demographics: recall gap |
| Lower Than A Level | 3239 | 0.458 | 0.430 [0.406, 0.451] | 0.430 [0.410, 0.455] | 0.488 [0.469, 0.513] | 0.839 [0.811, 0.865] | -0.030 [-0.045, -0.015] | lr_with_demographics: recall gap |
| No Formal quals (small) | 64 | 0.516 | 0.515 [0.367, 0.714] | 0.545 [0.355, 0.708] | 0.727 [0.562, 0.862] | 0.810 [0.632, 0.962] | -0.018 [-0.128, 0.093] |  |
| Post Graduate Qualification | 110 | 0.273 | 0.267 [0.121, 0.462] | 0.300 [0.130, 0.464] | 0.233 [0.071, 0.379] | 0.533 [0.278, 0.812] | 0.052 [-0.028, 0.132] |  |

### imd_band (reference: 30-40%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-10% | 789 | 0.473 | 0.421 [0.371, 0.467] | 0.418 [0.366, 0.467] | 0.480 [0.434, 0.534] | 0.831 [0.768, 0.875] | -0.047 [-0.076, -0.017] |  |
| 10-20 | 875 | 0.454 | 0.463 [0.418, 0.511] | 0.469 [0.418, 0.514] | 0.511 [0.465, 0.561] | 0.856 [0.807, 0.905] | -0.018 [-0.047, 0.010] |  |
| 20-30% | 941 | 0.441 | 0.455 [0.403, 0.499] | 0.460 [0.408, 0.502] | 0.487 [0.437, 0.530] | 0.848 [0.797, 0.896] | -0.028 [-0.055, -0.000] |  |
| 30-40% | 944 | 0.393 | 0.423 [0.370, 0.470] | 0.439 [0.393, 0.497] | 0.420 [0.366, 0.466] | 0.801 [0.737, 0.855] | 0.019 [-0.007, 0.046] |  |
| 40-50% | 884 | 0.382 | 0.423 [0.374, 0.481] | 0.438 [0.390, 0.492] | 0.420 [0.368, 0.476] | 0.753 [0.694, 0.816] | 0.029 [-0.000, 0.058] | gbm: calibration |
| 50-60% | 888 | 0.382 | 0.413 [0.358, 0.462] | 0.404 [0.352, 0.460] | 0.413 [0.360, 0.463] | 0.787 [0.719, 0.845] | 0.015 [-0.013, 0.044] |  |
| 60-70% | 858 | 0.355 | 0.380 [0.331, 0.438] | 0.377 [0.322, 0.432] | 0.334 [0.286, 0.397] | 0.758 [0.694, 0.824] | 0.018 [-0.008, 0.046] | lr_with_demographics: recall gap |
| 70-80% | 840 | 0.365 | 0.417 [0.366, 0.469] | 0.430 [0.377, 0.489] | 0.404 [0.348, 0.455] | 0.780 [0.712, 0.843] | 0.018 [-0.012, 0.043] |  |
| 80-90% | 783 | 0.332 | 0.388 [0.329, 0.445] | 0.346 [0.295, 0.416] | 0.358 [0.297, 0.416] | 0.748 [0.664, 0.816] | 0.048 [0.021, 0.075] | gbm: recall gap; calibration |
| 90-100% | 718 | 0.305 | 0.324 [0.266, 0.388] | 0.352 [0.287, 0.413] | 0.306 [0.241, 0.366] | 0.717 [0.613, 0.800] | 0.048 [0.018, 0.080] | lr_main: recall gap; gbm: calibration; lr_with_demographics: recall gap |
| Unknown | 359 | 0.273 | 0.296 [0.205, 0.394] | 0.276 [0.192, 0.375] | 0.194 [0.120, 0.281] | 0.806 [0.667, 0.939] | 0.035 [-0.003, 0.080] | lr_main: recall gap; gbm: recall gap; lr_with_demographics: recall gap |

### disability (reference: N)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| N | 8068 | 0.374 | 0.408 [0.395, 0.420] | 0.409 [0.398, 0.422] | 0.405 [0.393, 0.417] | 0.789 [0.767, 0.808] | 0.016 [0.007, 0.025] |  |
| Y | 811 | 0.502 | 0.455 [0.401, 0.500] | 0.467 [0.414, 0.508] | 0.509 [0.465, 0.557] | 0.845 [0.791, 0.891] | -0.044 [-0.076, -0.013] | lr_with_demographics: recall gap |

### disability_x_imd (reference: disability N, IMD 30-70%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| disability N, IMD 0-30% | 2290 | 0.438 | 0.441 [0.412, 0.470] | 0.441 [0.412, 0.469] | 0.477 [0.449, 0.507] | 0.837 [0.801, 0.866] | -0.022 [-0.039, -0.003] |  |
| disability N, IMD 30-70% | 3253 | 0.370 | 0.401 [0.376, 0.426] | 0.406 [0.386, 0.434] | 0.386 [0.361, 0.411] | 0.772 [0.738, 0.808] | 0.023 [0.009, 0.036] |  |
| disability N, IMD 70-100% | 2182 | 0.329 | 0.387 [0.354, 0.422] | 0.384 [0.353, 0.420] | 0.362 [0.326, 0.393] | 0.749 [0.698, 0.788] | 0.042 [0.025, 0.057] | gbm: calibration |
| disability N, IMD Unknown | 343 | 0.268 | 0.293 [0.205, 0.393] | 0.272 [0.187, 0.373] | 0.196 [0.118, 0.284] | 0.794 [0.649, 0.936] | 0.041 [0.001, 0.085] | lr_main: recall gap; gbm: recall gap; lr_with_demographics: recall gap |
| disability Y, IMD 0-30% | 315 | 0.581 | 0.481 [0.392, 0.546] | 0.497 [0.414, 0.560] | 0.579 [0.509, 0.650] | 0.889 [0.827, 0.949] | -0.094 [-0.143, -0.046] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| disability Y, IMD 30-70% | 321 | 0.467 | 0.487 [0.411, 0.566] | 0.493 [0.410, 0.566] | 0.507 [0.438, 0.597] | 0.802 [0.707, 0.876] | -0.004 [-0.052, 0.045] | lr_with_demographics: recall gap |
| disability Y, IMD 70-100% | 159 | 0.428 | 0.324 [0.212, 0.438] | 0.338 [0.224, 0.453] | 0.353 [0.233, 0.471] | 0.815 [0.649, 0.960] | -0.023 [-0.098, 0.043] |  |
| disability Y, IMD Unknown (small) | 16 | n/a | n/a | n/a | n/a | n/a | n/a |  |

## Week 12

### all (reference: All students)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| All students | 8571 | 0.363 | 0.450 [0.438, 0.460] | 0.445 [0.435, 0.458] | 0.452 [0.439, 0.463] | 0.815 [0.791, 0.832] | 0.039 [0.031, 0.048] |  |

### gender (reference: M)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| F | 3640 | 0.345 | 0.413 [0.392, 0.437] | 0.404 [0.380, 0.429] | 0.424 [0.400, 0.446] | 0.749 [0.712, 0.781] | 0.075 [0.060, 0.089] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |
| M | 4931 | 0.377 | 0.474 [0.456, 0.489] | 0.474 [0.457, 0.492] | 0.470 [0.453, 0.489] | 0.860 [0.831, 0.879] | 0.013 [0.002, 0.023] |  |

### age_band (reference: 0-35)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-35 | 5859 | 0.378 | 0.474 [0.456, 0.488] | 0.471 [0.454, 0.487] | 0.484 [0.465, 0.498] | 0.828 [0.800, 0.846] | 0.042 [0.031, 0.053] |  |
| 35-55 | 2632 | 0.334 | 0.391 [0.362, 0.421] | 0.384 [0.356, 0.416] | 0.376 [0.348, 0.406] | 0.780 [0.740, 0.821] | 0.034 [0.016, 0.049] | lr_with_demographics: recall gap |
| 55<= (small) | 80 | 0.300 | 0.375 [0.143, 0.562] | 0.375 [0.176, 0.600] | 0.250 [0.074, 0.455] | 0.818 [0.500, 1.000] | -0.017 [-0.105, 0.068] |  |

### region (reference: Scotland)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| East Anglian Region | 795 | 0.328 | 0.414 [0.351, 0.466] | 0.398 [0.338, 0.454] | 0.406 [0.349, 0.467] | 0.777 [0.695, 0.840] | 0.056 [0.025, 0.084] | lr_main: calibration |
| East Midlands Region | 582 | 0.333 | 0.428 [0.353, 0.491] | 0.402 [0.345, 0.487] | 0.443 [0.365, 0.508] | 0.847 [0.759, 0.907] | 0.046 [0.013, 0.078] | lr_with_demographics: calibration |
| Ireland | 398 | 0.352 | 0.379 [0.305, 0.468] | 0.386 [0.304, 0.467] | 0.379 [0.297, 0.455] | 0.855 [0.762, 0.935] | 0.026 [-0.012, 0.065] | lr_with_demographics: recall gap |
| London Region | 753 | 0.376 | 0.502 [0.439, 0.556] | 0.481 [0.420, 0.540] | 0.484 [0.419, 0.541] | 0.855 [0.798, 0.913] | 0.037 [0.009, 0.066] |  |
| North Region | 488 | 0.322 | 0.414 [0.339, 0.494] | 0.395 [0.322, 0.474] | 0.376 [0.304, 0.453] | 0.793 [0.700, 0.888] | 0.040 [0.004, 0.074] | lr_with_demographics: recall gap |
| North Western Region | 679 | 0.398 | 0.519 [0.458, 0.575] | 0.541 [0.486, 0.599] | 0.559 [0.498, 0.618] | 0.833 [0.774, 0.893] | 0.037 [0.007, 0.068] | lr_with_demographics: calibration |
| Scotland | 1222 | 0.412 | 0.471 [0.428, 0.513] | 0.471 [0.432, 0.519] | 0.475 [0.433, 0.517] | 0.878 [0.828, 0.912] | -0.003 [-0.025, 0.022] |  |
| South East Region | 550 | 0.325 | 0.419 [0.341, 0.488] | 0.408 [0.333, 0.473] | 0.369 [0.298, 0.436] | 0.728 [0.639, 0.816] | 0.070 [0.034, 0.103] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| South Region | 785 | 0.315 | 0.397 [0.333, 0.456] | 0.401 [0.339, 0.455] | 0.372 [0.310, 0.430] | 0.731 [0.654, 0.800] | 0.078 [0.050, 0.107] | lr_main: calibration; gbm: calibration; lr_with_demographics: recall gap |
| South West Region | 631 | 0.358 | 0.473 [0.409, 0.537] | 0.451 [0.386, 0.523] | 0.438 [0.378, 0.510] | 0.817 [0.748, 0.885] | 0.031 [0.002, 0.061] |  |
| Wales | 618 | 0.424 | 0.408 [0.350, 0.468] | 0.416 [0.357, 0.477] | 0.458 [0.403, 0.519] | 0.787 [0.703, 0.844] | 0.025 [-0.011, 0.062] | lr_with_demographics: calibration |
| West Midlands Region | 592 | 0.383 | 0.454 [0.392, 0.518] | 0.463 [0.392, 0.521] | 0.480 [0.413, 0.541] | 0.851 [0.777, 0.908] | 0.024 [-0.007, 0.058] |  |
| Yorkshire Region | 478 | 0.345 | 0.497 [0.436, 0.581] | 0.497 [0.421, 0.561] | 0.545 [0.463, 0.613] | 0.766 [0.694, 0.854] | 0.071 [0.037, 0.108] | lr_main: calibration; gbm: calibration; lr_with_demographics: calibration |

### highest_education (reference: A Level or Equivalent)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| A Level or Equivalent | 3829 | 0.332 | 0.458 [0.428, 0.476] | 0.449 [0.423, 0.474] | 0.419 [0.395, 0.441] | 0.794 [0.759, 0.821] | 0.061 [0.049, 0.074] | lr_main: calibration |
| HE Qualification | 1466 | 0.295 | 0.396 [0.352, 0.439] | 0.382 [0.341, 0.433] | 0.354 [0.306, 0.393] | 0.767 [0.708, 0.817] | 0.052 [0.032, 0.071] | lr_main: calibration |
| Lower Than A Level | 3108 | 0.435 | 0.459 [0.437, 0.485] | 0.464 [0.440, 0.487] | 0.513 [0.488, 0.536] | 0.853 [0.820, 0.877] | 0.006 [-0.009, 0.020] | lr_with_demographics: calibration |
| No Formal quals (small) | 61 | 0.492 | 0.567 [0.382, 0.750] | 0.567 [0.371, 0.750] | 0.667 [0.483, 0.821] | 0.895 [0.722, 1.000] | 0.006 [-0.101, 0.115] |  |
| Post Graduate Qualification (small) | 107 | 0.252 | 0.333 [0.148, 0.519] | 0.222 [0.077, 0.435] | 0.296 [0.130, 0.500] | 0.643 [0.364, 0.923] | 0.066 [-0.017, 0.147] |  |

### imd_band (reference: 30-40%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| 0-10% | 744 | 0.441 | 0.470 [0.418, 0.524] | 0.466 [0.414, 0.522] | 0.527 [0.469, 0.574] | 0.856 [0.799, 0.904] | 0.005 [-0.024, 0.035] | lr_with_demographics: calibration |
| 10-20 | 836 | 0.428 | 0.494 [0.443, 0.544] | 0.511 [0.458, 0.560] | 0.528 [0.479, 0.583] | 0.847 [0.790, 0.894] | 0.022 [-0.008, 0.050] | lr_with_demographics: calibration |
| 20-30% | 901 | 0.416 | 0.496 [0.446, 0.541] | 0.483 [0.437, 0.533] | 0.525 [0.473, 0.569] | 0.861 [0.808, 0.904] | 0.011 [-0.015, 0.038] |  |
| 30-40% | 907 | 0.368 | 0.485 [0.434, 0.538] | 0.485 [0.431, 0.537] | 0.467 [0.412, 0.518] | 0.827 [0.760, 0.876] | 0.052 [0.024, 0.077] | lr_main: calibration |
| 40-50% | 852 | 0.359 | 0.441 [0.386, 0.498] | 0.477 [0.414, 0.536] | 0.451 [0.399, 0.508] | 0.758 [0.702, 0.832] | 0.056 [0.029, 0.083] | lr_main: calibration; lr_with_demographics: calibration |
| 50-60% | 864 | 0.365 | 0.454 [0.401, 0.505] | 0.416 [0.367, 0.470] | 0.441 [0.391, 0.495] | 0.841 [0.782, 0.893] | 0.034 [0.008, 0.061] |  |
| 60-70% | 836 | 0.339 | 0.396 [0.337, 0.446] | 0.385 [0.331, 0.446] | 0.371 [0.323, 0.433] | 0.772 [0.691, 0.841] | 0.039 [0.009, 0.069] | gbm: recall gap; lr_with_demographics: recall gap |
| 70-80% | 821 | 0.351 | 0.472 [0.421, 0.531] | 0.458 [0.410, 0.522] | 0.476 [0.421, 0.534] | 0.840 [0.773, 0.892] | 0.041 [0.014, 0.067] |  |
| 80-90% | 760 | 0.312 | 0.397 [0.338, 0.462] | 0.392 [0.332, 0.454] | 0.388 [0.323, 0.451] | 0.752 [0.685, 0.838] | 0.068 [0.038, 0.095] | lr_main: calibration; lr_with_demographics: calibration |
| 90-100% | 698 | 0.285 | 0.362 [0.290, 0.426] | 0.357 [0.290, 0.422] | 0.302 [0.241, 0.370] | 0.720 [0.616, 0.791] | 0.066 [0.038, 0.094] | lr_main: recall gap; calibration; gbm: recall gap; lr_with_demographics: recall gap |
| Unknown | 352 | 0.259 | 0.319 [0.231, 0.431] | 0.286 [0.193, 0.378] | 0.231 [0.149, 0.329] | 0.806 [0.649, 0.929] | 0.043 [0.004, 0.082] | lr_main: recall gap; gbm: recall gap; lr_with_demographics: recall gap |

### disability (reference: N)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| N | 7802 | 0.352 | 0.440 [0.427, 0.453] | 0.436 [0.424, 0.450] | 0.436 [0.421, 0.449] | 0.808 [0.780, 0.826] | 0.043 [0.034, 0.052] |  |
| Y | 769 | 0.475 | 0.523 [0.466, 0.566] | 0.515 [0.465, 0.563] | 0.570 [0.521, 0.618] | 0.868 [0.820, 0.909] | 0.002 [-0.028, 0.032] | lr_with_demographics: recall gap |

### disability_x_imd (reference: disability N, IMD 30-70%)

| Group | n | At risk | lr_main recall | gbm recall | lr_with_demographics recall | lr_main precision | lr_main calibration gap | Flags |
|---|---|---|---|---|---|---|---|---|
| disability N, IMD 0-30% | 2190 | 0.412 | 0.467 [0.436, 0.498] | 0.469 [0.439, 0.499] | 0.499 [0.464, 0.525] | 0.844 [0.806, 0.875] | 0.017 [0.000, 0.036] |  |
| disability N, IMD 30-70% | 3153 | 0.350 | 0.435 [0.408, 0.461] | 0.438 [0.411, 0.466] | 0.421 [0.396, 0.449] | 0.797 [0.760, 0.830] | 0.047 [0.033, 0.062] |  |
| disability N, IMD 70-100% | 2122 | 0.310 | 0.426 [0.391, 0.459] | 0.407 [0.376, 0.446] | 0.401 [0.365, 0.438] | 0.776 [0.727, 0.818] | 0.061 [0.045, 0.077] | lr_main: calibration |
| disability N, IMD Unknown | 337 | 0.255 | 0.326 [0.240, 0.439] | 0.291 [0.197, 0.388] | 0.244 [0.156, 0.343] | 0.800 [0.636, 0.926] | 0.048 [0.009, 0.089] | gbm: recall gap; lr_with_demographics: recall gap |
| disability Y, IMD 0-30% | 291 | 0.546 | 0.604 [0.517, 0.669] | 0.591 [0.516, 0.671] | 0.686 [0.611, 0.755] | 0.906 [0.844, 0.956] | -0.020 [-0.066, 0.028] | lr_main: recall gap; gbm: recall gap; lr_with_demographics: recall gap |
| disability Y, IMD 30-70% | 306 | 0.441 | 0.533 [0.448, 0.611] | 0.481 [0.395, 0.557] | 0.548 [0.475, 0.636] | 0.828 [0.750, 0.910] | 0.024 [-0.021, 0.070] | lr_with_demographics: recall gap |
| disability Y, IMD 70-100% | 157 | 0.420 | 0.333 [0.220, 0.458] | 0.424 [0.292, 0.546] | 0.379 [0.259, 0.500] | 0.846 [0.690, 0.963] | 0.004 [-0.066, 0.068] |  |
| disability Y, IMD Unknown (small) | 15 | n/a | n/a | n/a | n/a | n/a | n/a |  |
