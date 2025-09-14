# Release notes

## v0.2.0 (unreleased)

### Changes

- `tams.data.download_examples()` has been removed ({pull}`69`).
  Example data files now download automatically,
  and to the user cache directory instead of into the package.
  This functionality requires [pooch](https://www.fatiando.org/pooch/), in addition to [gdown](https://github.com/wkentaro/gdown).
- `tams.load_mpas_precip()` has been removed ({pull}`70`).
  This function name was confusing because "precip" here
  was a reference to the
  [PRECIP field campaign](http://precip.org/) (summer 2022).
  It was specifically designed to load postprocessed outputs
  from near-real-time runs associated with that campaign.
