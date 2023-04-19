# Differences

## TAMS v1.0[^v1] and TAMS v2.0[^v2]

### Identification

1. Although both versions use masks of the data (and in that way, the identification does not need to depend on a specific grid), TAMS v1.0 uses the cloud-shaped masked areas of interest for identifications while TAMS v2.0 uses convex-hull-shaped masked areas.
2. In TAMS v2.0, 219 K areas that are very small ($\le$ 10 km$^2$) are eliminated as well as 235 areas K that do not meet the 235 K area of 4000 km$^2$.
3. The statistics calculated (e.g., 219 K and 235 K std and averages) are taken from the total of the corresponding areas rather than taking the average of the potential multiple 219 K area averages within a 235 K area as in TAMS v1.0

### Tracking

1. Unlike TAMS v1.0 which matches MCSs (and thus, matches "parent" and "kid" clouds) forward in time using a recursive function, TAMS v2.0 matches back in time such that for example: clouds at time $i$ are matched with clouds at time $i-1$.

### Classification

1. Unlike the definition in TAMS v1.0, {abbr}`DSL (disorganized short-lived)` in TAMS v2.0 are classified as anything with duration shorter than 6 hours.
2. Unlike in TAMS v1.0, the time criterion for MCC classification in TAMS v2.0 needs to hold for 6 consecutive hours.

[^v1]: "TAMS v1.0": TAMS as published in {cite:t}`TAMS1.0`, written in Matlab.

<!-- prettier-ignore-start -->
[^v2]: "TAMS v2.0": New version of TAMS, written in Python.
  Note that the major version of the Python package
  is not (currently) consistent with the "v2.0".
  Currently, the major version is 0, to indicate that the API is not stable yet.
<!-- prettier-ignore-end -->

## TAMS vs [tobac](https://tobac.readthedocs.io)

- tobac is a more general cloud tracking toolkit with more options, etc., while TAMS targets the MCS case (though much of the core API is purposely left more general)
- tobac[^tob] treats features as single points, while TAMS treats them as georeferenced polygonal areas (Shapely)
- tobac separates feature identification and segmentation, such that the geo areas of the original identified features based on thresholds can be different than the segmentation areas, which are calculated with a watershed method. In TAMS, the feautre geo area is treated as the segmentation area.
- To associate point data with the features (segmentation), tobac uses a feature ID mask array (currently iris, though xarray is planned). Since this uses a watershed methd, the input data must be on a structured grid(?). By default, TAMS associates data with features using GeoPandas spatial join with the feature polygons, which (in principle) can be used even with non-gridded input data.
- tobac uses [Trackpy](https://soft-matter.github.io/trackpy/) for tracking, while TAMS uses custom methods based on area overlap and (currently fixed) zonal projcetion velocities
  - Trackpy [includes](https://soft-matter.github.io/trackpy/v0.6.1/tutorial/prediction.html) options for predicting the next position of a point based on its trajectory history, which tobac uses(?). That is, tobac (with Trackpy) tracks points, while TAMS tracks geo areas.
- tobac v1.5 can identify and track 3-D features, while TAMS supports 2-D only
- tobac v1.4 added merge/split combination of cells (cell: "is a series of features linked together over multiple timesteps") as a post-processing step, while TAMS accounts for merge/splits in the tracking step based on overlaps

[^tob]: tobac refers to both the project and the Python package with lowercase.
