# GRU pulse write-mode comparison

Seed 42; final step 2012217344; mean absolute change in the pulsed coordinate.
First/top curves reuse exactly reproduced original results. All-layer targets use intact per-layer shadows, with sequential steered-input writes and carry replacement.

| GRU | Condition | Pulse | Write | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gru128 | Full-State Quotient | mirror | first | 0.075162 | 0.075706 | 0.072516 | 0.112556 | 0.152637 | 0.126915 | 0.067287 | 0.019848 |
| gru128 | Full-State Quotient | mirror | top | 0.553048 | 0.544448 | 0.528394 | 0.413227 | 0.231898 | 0.105384 | 0.031433 | 0.005176 |
| gru128 | Full-State Quotient | mirror | all | 0.553822 | 0.543549 | 0.532849 | 0.481896 | 0.369353 | 0.233330 | 0.094248 | 0.022503 |
| gru128 | Full-State Quotient | mirror | filter | 0.562869 | 0.541776 | 0.532318 | 0.488744 | 0.430179 | 0.340298 | 0.201632 | 0.069044 |
| gru128 | Full-State Quotient | rho_flip | first | 0.105681 | 0.146354 | 0.162148 | 0.157921 | 0.141794 | 0.114135 | 0.056415 | 0.013012 |
| gru128 | Full-State Quotient | rho_flip | top | 0.609643 | 0.498276 | 0.439917 | 0.376876 | 0.285742 | 0.139565 | 0.051298 | 0.012372 |
| gru128 | Full-State Quotient | rho_flip | all | 0.610290 | 0.535047 | 0.538685 | 0.498280 | 0.415986 | 0.258585 | 0.092234 | 0.023789 |
| gru128 | Full-State Quotient | rho_flip | filter | 0.628762 | 0.576303 | 0.531049 | 0.452289 | 0.350169 | 0.211599 | 0.077960 | 0.010940 |
| gru128 | Reward-State Quotient | mirror | first | 0.195107 | 0.238784 | 0.267730 | 0.281822 | 0.255484 | 0.167449 | 0.061422 | 0.018179 |
| gru128 | Reward-State Quotient | mirror | top | 0.442613 | 0.259966 | 0.169844 | 0.087411 | 0.060668 | 0.039077 | 0.011609 | 0.001642 |
| gru128 | Reward-State Quotient | mirror | all | 0.465074 | 0.386145 | 0.355215 | 0.324262 | 0.280307 | 0.181585 | 0.063695 | 0.017581 |
| gru128 | Reward-State Quotient | mirror | filter | 0.562556 | 0.537658 | 0.524433 | 0.474902 | 0.396013 | 0.291203 | 0.144051 | 0.037912 |
| gru128 | Reward-State Quotient | rho_flip | first | 0.524308 | 0.381879 | 0.285971 | 0.207429 | 0.164960 | 0.147780 | 0.052602 | 0.010926 |
| gru128 | Reward-State Quotient | rho_flip | top | 0.626506 | 0.173387 | 0.293035 | 0.499394 | 0.394957 | 0.140905 | 0.047155 | 0.012540 |
| gru128 | Reward-State Quotient | rho_flip | all | 0.627459 | 0.464916 | 0.361919 | 0.253816 | 0.193073 | 0.145934 | 0.056296 | 0.012007 |
| gru128 | Reward-State Quotient | rho_flip | filter | 0.635517 | 0.579721 | 0.531055 | 0.447842 | 0.348699 | 0.215734 | 0.080820 | 0.011728 |
| gru92 | Full-State Quotient | mirror | first | 0.077653 | 0.136609 | 0.183667 | 0.217870 | 0.206781 | 0.156665 | 0.076339 | 0.019259 |
| gru92 | Full-State Quotient | mirror | top | 0.587766 | 0.145565 | 0.092499 | 0.119728 | 0.125106 | 0.052806 | 0.013958 | 0.002135 |
| gru92 | Full-State Quotient | mirror | all | 0.599913 | 0.613852 | 0.570671 | 0.458344 | 0.349417 | 0.202720 | 0.085008 | 0.021061 |
| gru92 | Full-State Quotient | mirror | filter | 0.612752 | 0.594693 | 0.586227 | 0.534732 | 0.465780 | 0.362770 | 0.202244 | 0.058050 |
| gru92 | Full-State Quotient | rho_flip | first | 0.061798 | 0.074045 | 0.075921 | 0.076864 | 0.078289 | 0.059963 | 0.028670 | 0.008060 |
| gru92 | Full-State Quotient | rho_flip | top | 0.604575 | 0.533496 | 0.355440 | 0.140248 | 0.059721 | 0.018930 | 0.011675 | 0.002391 |
| gru92 | Full-State Quotient | rho_flip | all | 0.603363 | 0.457749 | 0.397574 | 0.268787 | 0.251693 | 0.167766 | 0.042274 | 0.013726 |
| gru92 | Full-State Quotient | rho_flip | filter | 0.636748 | 0.576298 | 0.519867 | 0.427596 | 0.320536 | 0.187914 | 0.059498 | 0.010895 |
| gru92 | Reward-State Quotient | mirror | first | 0.046480 | 0.069499 | 0.076191 | 0.069804 | 0.051572 | 0.023185 | 0.005798 | 0.000901 |
| gru92 | Reward-State Quotient | mirror | top | 0.356821 | 0.121595 | 0.068626 | 0.036511 | 0.017841 | 0.006875 | 0.003265 | 0.002435 |
| gru92 | Reward-State Quotient | mirror | all | 0.434039 | 0.305342 | 0.204621 | 0.153078 | 0.116876 | 0.057823 | 0.019975 | 0.006858 |
| gru92 | Reward-State Quotient | mirror | filter | 0.665878 | 0.637909 | 0.620815 | 0.560762 | 0.466123 | 0.330651 | 0.155486 | 0.039831 |
| gru92 | Reward-State Quotient | rho_flip | first | 0.186707 | 0.133334 | 0.079037 | 0.043105 | 0.039815 | 0.028449 | 0.014554 | 0.002553 |
| gru92 | Reward-State Quotient | rho_flip | top | 0.643188 | 0.582771 | 0.432077 | 0.240143 | 0.181563 | 0.100914 | 0.029792 | 0.008357 |
| gru92 | Reward-State Quotient | rho_flip | all | 0.636036 | 1.273855 | 0.858748 | 0.467055 | 0.599924 | 0.801938 | 0.741062 | 0.403695 |
| gru92 | Reward-State Quotient | rho_flip | filter | 0.654252 | 0.574639 | 0.519802 | 0.406580 | 0.290825 | 0.148492 | 0.032107 | 0.002475 |
