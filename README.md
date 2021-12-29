# Find Subimage

[<img alt="github" src="https://img.shields.io/badge/github-nico--abram/find--subimage-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/nico-abram/find-subimage)
[<img alt="crates.io" src="https://img.shields.io/crates/v/find-subimage.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/find-subimage)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-find--subimage-66c2a5?style=for-the-badge&labelColor=555555&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">](https://docs.rs/find-subimage)
[<img alt="build status" src="https://img.shields.io/github/workflow/status/nico-abram/find-subimage/find-subimage/main?style=for-the-badge" height="20">](https://github.com/nico-abram/find-subimage/actions?query=branch%3Amain)

<!--
[![Github Actions](https://github.com/nico-abram/find-subimage/workflows/find-subimage/badge.svg)](https://github.com/nico-abram/find-subimage/actions?query=workflow%3Afind-subimage)
[<img alt="Documentation" src="https://docs.rs/find-subimage/badge.svg" height="20">](https://docs.rs/find-subimage)
[<img alt="Build status" src="https://img.shields.io/crates/v/find-subimage.svg?style=for-the-badge" height="20">](https://crates.io/crates/find-subimage)
-->

This crate provides basic functionality to find likely positions of a subimage within a larger image by calculating an image distance. It has a naive scalar implementation in rust, and a simd implementation that selects the best implementation based on cpu features at runtime in rust using the [simdeez](https://github.com/jackmott/simdeez) crate. It also provides an implementation which uses [OpenCV](https://opencv.org/)'s (A C++ library) matchTemplate function using the [opencv-rust](https://github.com/twistedfall/opencv-rust) crate through an optional off-by-default feature. It can also optionally convert images to grayscale before applying the algorithms.

Here's a simple example showing how to use the API:

```rs
use find_subimage::{Image, SubImageFinderState};
// Make a dummy 128x128 black image with a red dot at (50, 0)
let (w, h) = (128, 128);
let mut rgb_image = vec![0u8; w * h * 3];
rgb_image[50 * 3] = 250;
// Make a dummy 32x32 black image
// with a white dot at (0, 0)
let (sub_w, sub_h) = (32, 32);
let mut rgb_subimage = vec![0u8; sub_w * sub_h * 3];
rgb_subimage[0] = 250;
//!
let mut finder = SubImageFinderState::new();
// These are (x, y, distance) where x and y are the position within the larger image
// and distance is the distance value, where a smaller distance means a mroe precise match
let positions: &[(usize, usize, f32)] =
  finder.find_subimage_positions((&rgb_image, w, h), (&rgb_subimage, sub_w, sub_h), 3);
let max: Option<&(usize, usize, f32)> = positions
  .iter()
  .min_by(|(_, _, dist), (_, _, dist2)| dist.partial_cmp(dist2).unwrap());
println!(
  "The subimage was found at position {:?}",
  positions
    .iter()
    .min_by(|(_, _, dist), (_, _, dist2)| dist.partial_cmp(dist2).unwrap())
);
assert_eq!(Some((50, 0)), max.map(|max| (max.0, max.1)));
// find_subimage_positions actually returns the results sorted by distance already,
// so we can skip finding the minimum
assert_eq!(Some((50, 0)), positions.get(0).map(|max| (max.0, max.1)));
```

Additional documentation can be found in the generated [rustdoc docs hosted on docs.rs](https://docs.rs/find-subimage/latest/find_subimage/).

## Licensing

The code in this repository is available under any of the following licenses, at your choice: MIT OR Apache-2.0 OR BSL-1.0 OR MPL-2.0 OR Zlib OR Unlicense

This crate optionally depends on [opencv](https://opencv.org/). You can find it's license [here](https://opencv.org/license/) (3-clause BSD or Apache 2 depending on the version). The rust bindings are [licensed as MIT](https://github.com/twistedfall/opencv-rust/blob/master/LICENSE).

It also optionally depends on [simdeez](https://github.com/jackmott/simdeez) which is licensed as [MIT](https://github.com/jackmott/simdeez/blob/master/LICENSE).
