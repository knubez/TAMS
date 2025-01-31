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

Compared to normal TAMS usage, there are a few differences:

- In the identification stage, we use `ctt_threshold=241, ctt_core_threshold=225`
  for consistency with the MOSA cold-core temperature threshold criterion.
  These are higher than the TAMS defaults (235 and 219 K, respectively) [^threshs].
  CE precip stats are also computed in this step, to later be used in classification
  (the standard TAMS classification routine does not need precip stats).

- After tracking, a custom classification routine computes whether a CE group is an MCS or not.
  In normal TAMS, `tams.classify()` assigns CE groups to one of four MCS classes.

- Conversion to a gridded mask representation is not needed in order to work with TAMS outputs.
  It was done in order to compare to other trackers.
  Some information is lost in this conversion.

[^dyamond]: https://www.esiwace.eu/the-project/past-phases/dyamond-initiative

[^threshs]:
    In general, a higher `ctt_core_threshold` leads to more CEs being identified,
    while a higher `ctt_threshold` leads to larger CE areas.
    However, since contouring is used (instead of watershedding or spread-and-grow or such),
    a higher `ctt_threshold` can also merge regions
    that might be considered separate CEs with a lower threshold.
