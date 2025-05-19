## MOSA

MOSA: MCSs Over South America [^mosa]

An MCS tracker intercomparison project of the NCAR SAAG Convective Working Group, focusing on the South American region.

Publication: [Prein et al. (2024)](https://doi.org/10.1029/2023JD040254).

TAMS is run on observational ([GPM_MERGIR](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary) brightness temperature and [IMERG](https://gpm.nasa.gov/data/imerg) precip) and model (WRF) datasets for three water years (2011, 2016, 2019).

> [!IMPORTANT]
> This used TAMS v0.1.x.
> The scripts may not work with future versions of TAMS.
> To install TAMS v0.1.x, use
>
> ```
> conda install -c conda-forge tams=0.1
> ```
>
> or clone the repo and check out the latest `v0.1.x` tag.

This directory includes Python and PBS job scripts (for NCAR Casper) for three steps in the run process:

1. Cloud element (CE) identification, creating Parquet files of CEs identified at each time, including precip stats within CEs

2. Track and classify, adding MCS IDs and assessing whether the MCSs meet the MOSA criteria, saving results as a single Parquet file (6 total)

3. Convert to mask representation (requires dissolving constituent CEs into single shape per time per track) and drop tracks that are not MCS under the MOSA criteria, saving to netCDF

[^mosa]: Original working title for the project, but not used in the journal article.
