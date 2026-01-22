# Release notes

## v0.2.0 (unreleased)

### Changes

- `tams.data.download_examples()` has been removed ({pull}`69`).
  Example data files now download automatically
  when you use {func}`tams.data.open_example` or {func}`tams.data.load_example`,
  and to the user cache directory instead of into the package.
  These functions provide a unified interface to access the example datasets,
  as opposed to the separate `load_example_*` functions previously available.
  This functionality requires [pooch](https://www.fatiando.org/pooch/), in addition to [gdown](https://github.com/wkentaro/gdown).
- `tams.load_mpas_precip()` has been removed ({pull}`70`).
  This function name was confusing because "precip" here
  was a reference to the
  [PRECIP field campaign](http://precip.org/) (summer 2022).
  It was specifically designed to load postprocessed outputs
  from near-real-time runs associated with that campaign.
- `tams.contours()` has been renamed to {func}`tams.contour`
  and now returns a {class}`~geopandas.GeoDataFrame` of contour lines
  (instead of a list of arrays of line segment coordinates; {pull}`74`).
- {func}`tams.identify` now returns a single of list of {class}`~geopandas.GeoDataFrame`s
  instead of also returning the identified cores ({pull}`84`).
  In these frames, the `core` column contains the cold cores within each CE
  (renamed from `cs219`), with area `area_core_km2` (renamed from `area219_km2`).
  "235" and "219" language was generally removed
  in favor of the more general "CE" and "core" terminology,
  since different thresholds can be used.
  Scalar geometries of the `core` column are now either
  {class}`~shapely.MultiPolygon`, {class}`~shapely.Polygon`, or `None` (no cores)
  instead of always being {class}`~shapely.MultiPolygon`.
- In {func}`tams.identify`, the `size_filter` Boolean parameter is deprecated
  (use `size_threshold=0` instead to disable size filtering; {pull}`84`).

### New features

- The existing example datasets were updated to improve compression,
  and additional datasets have been added,
  including the MOSA test cases and a sample IMERG dataset ({pull}`70`).
  See {ref}`example_datasets` for details.
- Contouring ({func}`tams.contour`) now drops non-closed contours by default
  and computes whether the contour encloses higher or lower values
  (to account for holes in CE polygons; {pull}`74`).
- Convex hulling can be disabled in {func}`tams.identify` ({pull}`74`).
- Logger level and handler can be controlled using {func}`tams.set_options`
  and should still work even when using `parallel=True`.
  The logs now include more information about the contouring,
  including reasons for exclusion ({pull}`74`).
- Create idealized datasets for testing using {mod}`tams.idealized` ({pull}`80`).
