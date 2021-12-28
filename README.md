# Find Subimage

[![Github Actions](https://github.com/nico-abram/find-subimage/workflows/find-subimage/badge.svg)](https://github.com/nico-abram/find-subimage/actions?query=workflow%3Afind-subimage)
[![Documentation](https://docs.rs/find-subimage/badge.svg)](https://docs.rs/find-subimage)
[![Package](https://img.shields.io/crates/v/find-subimage.svg)](https://crates.io/crates/find-subimage)

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
