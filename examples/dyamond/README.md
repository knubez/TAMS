## DYAMOND

DYAMOND: DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains [^dyamond]

Similar to [the MOSA runs](../mosa/README.md), the observations consist of [GPM_MERGIR](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary) brightness temperature and [IMERG](https://gpm.nasa.gov/data/imerg) precip (v07 instead of v06). In addition to the obs, we ran TAMS for multiple high-resolution global models, for one winter and one summer period, each a bit more than a month long.

Publication: [Feng et al., submitted](https://doi.org/10.22541/essoar.172405876.67413040/v1).

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

[^dyamond]: https://www.esiwace.eu/the-project/past-phases/dyamond-initiative
