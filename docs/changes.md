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

### New features

- The existing example datasets were updated to improve compression,
  and additional datasets have been added,
  including the MOSA test cases and a sample IMERG dataset ({pull}`70`).
  See {ref}`example_datasets` for details.
