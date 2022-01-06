//! This crate provides basic functionality to find likely positions of a subimage within a larger
//! image by calculating an image distance. It has a naive scalar implementation in rust, and a simd
//! implementation that selects the best implementation based on cpu features at runtime in rust
//! using the [simdeez](https://github.com/jackmott/simdeez) crate. It also provides an implementation which uses
//! [OpenCV](https://opencv.org/)'s (A C++ library) matchTemplate function using the
//! [opencv-rust](https://github.com/twistedfall/opencv-rust) crate through an optional off-by-default
//! feature. It can also optionally convert images to grayscale before applying the algorithms.
//!
//! Here's a simple example showing how to use the API:
//!
//! ```
//! # // If you modify this example, please also update it's copy in README.md
//! use find_subimage::{Image, SubImageFinderState};
//! // Make a dummy 128x128 black image with a red dot at (50, 0)
//! let (w, h) = (128, 128);
//! let mut rgb_image = vec![0u8; w * h * 3];
//! rgb_image[50 * 3] = 250;
//! // Make a dummy 32x32 black image
//! // with a red dot at (0, 0)
//! let (sub_w, sub_h) = (32, 32);
//! let mut rgb_subimage = vec![0u8; sub_w * sub_h * 3];
//! rgb_subimage[0] = 250;
//!
//! let mut finder = SubImageFinderState::new();
//! // These are (x, y, distance) where x and y are the position within the larger image
//! // and distance is the distance value, where a smaller distance means a more precise match
//! let positions: &[(usize, usize, f32)] =
//!     finder.find_subimage_positions((&rgb_image, w, h), (&rgb_subimage, sub_w, sub_h), 3);
//! let max: Option<&(usize, usize, f32)> = positions
//!     .iter()
//!     .min_by(|(_, _, dist), (_, _, dist2)| dist.partial_cmp(dist2).unwrap());
//! println!("The subimage was found at position {:?}", &max);
//! assert_eq!(Some((50, 0)), max.map(|max| (max.0, max.1)));
//! // find_subimage_positions actually returns the results sorted by distance already,
//! // so we can skip finding the minimum
//! assert_eq!(Some((50, 0)), positions.get(0).map(|max| (max.0, max.1)));
//! ```
//!
//! The most important functions provided are [find_subimage_positions] and
//! [find_subimage_positions_as_grayscale].
//!
//! You may find their "_with_backend" versions useful.
//!
//! By default, this library prunes results that are close together. You can disable (Set to 0) or
//! tweak this using [with_pruning].
//!
//! You can look at the page for the [Backend] enum to learn about the possible backends.
//!
//! There are some examples in the /examples folder in the repository.
//!
//! [with_pruning]: SubImageFinderState::with_pruning
//! [find_subimage_positions]: SubImageFinderState::find_subimage_positions
//! [find_subimage_positions_as_grayscale]: SubImageFinderState::find_subimage_positions_as_grayscale

/// A simple struct to group (bytes, width, height) arguments
pub struct Image<'a> {
    pub bytes: &'a [u8],
    pub width: usize,
    pub height: usize,
}
impl<'a> Image<'a> {
    fn new(bytes: &'a [u8], width: usize, height: usize) -> Self {
        Self {
            bytes,
            width,
            height,
        }
    }
}
impl<'a, T: AsRef<[u8]>, A: Into<usize>, B: Into<usize>> From<(&'a T, A, B)> for Image<'a> {
    fn from((bytes, width, height): (&'a T, A, B)) -> Self {
        Image::new(bytes.as_ref(), width.into(), height.into())
    }
}

/// The main context struct. This stores the necessary buffers for the search and grayscale
/// conversion.
///
/// u8 buffers are used if conversion to grayscale is necessary, and f32 buffers for the backends
/// that require them.
///
/// There is also a Vec<(usize, usize, f32)> used to store results.
pub struct SubImageFinderState {
    pub positions_buffer: Vec<(usize, usize, f32)>,
    pub backend: Backend,

    pub prune_width_scale: f32,
    pub prune_height_scale: f32,

    pub f32buf_search_image: Vec<f32>,
    pub f32buf_subimage: Vec<f32>,

    pub u8buf_search_image: Vec<u8>,
    pub u8buf_subimage: Vec<u8>,
}

/// The backend/algorithm to use.
///
/// There is an optional opencv backend, that uses the opencv-rust crate which depends on the OpenCV
/// C++ library. This requires enabling the opencv feature in find-subimage.
///
/// There is another simdeez optional dependency, which uses the simdeez crate for a rust SIMD
/// implementation. This is enabled by default.
///
/// The only implementation which cannot be disabled at present is the scalar one.
#[derive(Clone, Copy)]
pub enum Backend {
    /// OpenCV SQDIFF_NORMED MatchTemplate
    ///
    /// Note that the threshold values for this backend use a different scale than the others.
    #[cfg(feature = "opencv")]
    OpenCV { threshold: f32 },
    /// This should detect CPU features at runtime and use the best possible rust SIMD
    /// implementation of SQDIFF_NORMED (square difference).
    ///
    /// step_x and y let you customize it to skip every Nth x or y coordinate in case you need less
    /// accurate results, potentially giving large speedups.
    #[cfg(feature = "simdeez")]
    RuntimeDetectedSimd {
        threshold: f32,
        step_x: usize,
        step_y: usize,
    },
    /// Scalar SQDIFF_NORMED (square difference) implementation.
    ///
    /// Slowest, should work anywhere and be reliable.
    ///
    /// Smallest in terms of generated code size.
    ///
    /// step_x and y let you customize it to skip every Nth x or y coordinate in case you need less
    /// accurate results, potentially giving large speedups.
    Scalar {
        threshold: f32,
        step_x: usize,
        step_y: usize,
    },
}
/// The default value used in [fn@SubImageFinderState::new_opencv]
pub const OPENCV_DEFAULT_THRESHOLD: f32 = 0.05;
/// The default value used in [SubImageFinderState::new] and [SubImageFinderState::default]
pub const NONOPENCV_DEFAULT_THRESHOLD: f32 = 0.1;

impl SubImageFinderState {
    /// Create a SubImageFinderState
    ///
    /// This uses the [Backend::Scalar] backend by default, unless the "simdeez-default-new" e is
    /// enabled (It is currently enabled by default).
    ///
    /// See the backend and with_backend methods to change the backend.
    pub fn new() -> Self {
        #[cfg(feature = "simdeez-default-new")]
        let backend = Backend::RuntimeDetectedSimd {
            threshold: NONOPENCV_DEFAULT_THRESHOLD,
            step_x: 1,
            step_y: 1,
        };
        #[cfg(not(feature = "simdeez-default-new"))]
        let backend = Backend::Scalar {
            threshold: NONOPENCV_DEFAULT_THRESHOLD,
            step_x: 1,
            step_y: 1,
        };
        Self {
            positions_buffer: vec![],
            f32buf_search_image: vec![],
            f32buf_subimage: vec![],
            u8buf_search_image: vec![],
            u8buf_subimage: vec![],
            prune_width_scale: 0.5f32,
            prune_height_scale: 0.5f32,
            backend,
        }
    }

    /// Like [Self::new] but uses [Backend::OpenCV]
    #[cfg(feature = "opencv")]
    pub fn new_opencv(threshold: Option<f32>) -> Self {
        let mut ret = Self::new();
        ret.backend = Backend::OpenCV {
            threshold: threshold.unwrap_or(OPENCV_DEFAULT_THRESHOLD),
        };
        ret
    }

    pub fn backend(&mut self) -> &Backend {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut Backend {
        &mut self.backend
    }

    /// Set the currently configured backend.
    ///
    /// See also [Self::with_backend]
    pub fn set_backend(&mut self, new_backend: Backend) {
        self.backend = new_backend;
    }

    /// Set the currently configured prune width/height scaling.
    ///
    /// For more information see [Self::with_pruning]
    pub fn set_pruning(&mut self, prune_width_scale: f32, prune_height_scale: f32) {
        self.prune_height_scale = prune_height_scale;
        self.prune_width_scale = prune_width_scale;
    }

    /// Return a new state with the given backend
    /// ```
    /// use find_subimage::{Backend, SubImageFinderState};
    /// let state = SubImageFinderState::new().with_backend(Backend::Scalar {
    ///     threshold: 0.5,
    ///     step_x: 2,
    ///     step_y: 1,
    /// });
    /// ```
    #[must_use]
    pub fn with_backend(mut self, new_backend: Backend) -> Self {
        self.set_backend(new_backend);
        self
    }

    /// Return a new state with the given pruning width/height scaling parameters.
    ///
    /// These default to 0.5
    #[must_use]
    pub fn with_pruning(mut self, prune_width_scale: f32, prune_height_scale: f32) -> Self {
        self.set_pruning(prune_width_scale, prune_height_scale);
        self
    }

    /// Finds positions where the subimage is found within the search image. These positions
    /// represent the top-right corner of the subimage.
    ///
    /// You can tweak the likelyhood of positions found with the backend's threshold. Note that the
    /// threshold is backend-dependant.
    ///
    /// The `channel_count` argument should be the number of channels for both input images (For
    /// example, 3 for an RGB image or 1 for grayscale).
    ///
    /// The input image can optionally be converted to grayscale before applying the algorithm, see
    /// [Self::find_subimage_positions_as_grayscale].
    ///
    /// The third field of the tuples in the returned slice is the matching/distance value. Values
    /// closer to 1 mean a fuzzier match, and closer to 0 a more exact match. These values are
    /// returned sorted by distance, with the best matches first.
    pub fn find_subimage_positions<'a, 'b, T: Into<Image<'a>>, U: Into<Image<'b>>>(
        &mut self,
        search_image: T,
        subimage: U,
        channel_count: u8,
    ) -> &[(usize, usize, f32)] {
        let backend = self.backend;
        self.find_subimage_positions_with_backend(
            search_image.into(),
            subimage.into(),
            &backend,
            channel_count,
        )
    }

    /// Like [Self::find_subimage_positions_as_grayscale] but lets you use a different backend
    /// than the currently configured one.
    pub fn find_subimage_positions_with_backend<'a, 'b, T: Into<Image<'a>>, U: Into<Image<'b>>>(
        &mut self,
        search_image: T,
        subimage: U,
        backend: &Backend,
        channel_count: u8,
    ) -> &[(usize, usize, f32)] {
        self.find_subimage_positions_with_backend_impl(
            search_image.into(),
            subimage.into(),
            backend,
            false,
            channel_count,
            channel_count,
        )
    }

    /// Like [Self::find_subimage_positions], but before finding positions it converts the images to
    /// grayscale. This can speed up runtime, but depending on the images it can be harmful to
    /// results.
    ///
    /// This is done using internal buffers. If you reuse a [SubImageFinderState] for multiple
    /// images of the same size, it should only need to allocate once.
    ///
    /// If channel_count_subimage is None, channel_count_search is used in its place.
    pub fn find_subimage_positions_as_grayscale<'a, 'b, T: Into<Image<'a>>, U: Into<Image<'b>>>(
        &mut self,
        search_image: T,
        subimage: U,
        channel_count_search: u8,
        channel_count_subimage: Option<NonZeroU8>,
    ) -> &[(usize, usize, f32)] {
        let backend = self.backend;
        self.find_subimage_positions_as_grayscale_with_backend(
            search_image.into(),
            subimage.into(),
            &backend,
            channel_count_search,
            channel_count_subimage,
        )
    }

    /// Like [Self::find_subimage_positions_as_grayscale] but lets you use a different backend
    /// than the currently configured one.
    pub fn find_subimage_positions_as_grayscale_with_backend<
        'a,
        'b,
        T: Into<Image<'a>>,
        U: Into<Image<'b>>,
    >(
        &mut self,
        search_image: T,
        subimage: U,
        backend: &Backend,
        channel_count_search: u8,
        channel_count_subimage: Option<NonZeroU8>,
    ) -> &[(usize, usize, f32)] {
        self.find_subimage_positions_with_backend_impl(
            search_image.into(),
            subimage.into(),
            backend,
            true,
            channel_count_search,
            channel_count_subimage
                .map(|x| x.get())
                .unwrap_or(channel_count_search),
        )
    }

    /// The main implementation of the algorithm.
    ///
    /// This runs the hot loop, performs grayscale conversion, calls the appropiate backend, and
    /// prunes results at the end.
    ///
    /// All the public functions that find positions call into this.
    fn find_subimage_positions_with_backend_impl(
        &mut self,
        search_image: Image,
        subimage: Image,
        backend: &Backend,
        to_grayscale: bool,
        search_image_channel_count: u8,
        subimage_channel_count: u8,
    ) -> &[(usize, usize, f32)] {
        // If there is no grayscale conversion, channel counts should match
        if !to_grayscale && search_image_channel_count != subimage_channel_count {
            panic!(
              "Attempted to find_subimage_positions with different channel counts. search:{} subimage:{}",
              search_image_channel_count, subimage_channel_count
            );
        }

        self.positions_buffer.clear();

        let Image {
            bytes: search_image,
            width: search_width,
            height: search_height,
        } = search_image;
        let Image {
            bytes: subimage,
            width: subimage_width,
            height: subimage_height,
        } = subimage;

        let to_gray_sub = move |rgb: &[u8]| {
            rgb.iter()
                .map(|x| (*x as f32) / (subimage_channel_count as f32))
                .sum::<f32>() as u8
        };
        let to_gray_search = move |rgb: &[u8]| {
            rgb.iter()
                .map(|x| (*x as f32) / (search_image_channel_count as f32))
                .sum::<f32>() as u8
        };
        let to_f32 = |x: u8| x as f32;
        let ref_to_f32 = |&x: &u8| x as f32;

        match *backend {
            #[cfg(feature = "simdeez")]
            Backend::RuntimeDetectedSimd {
                threshold,
                step_x,
                step_y,
            } => {
                self.f32buf_subimage.clear();
                if to_grayscale && subimage_channel_count != 1 {
                    self.f32buf_subimage.extend(
                        subimage
                            .chunks_exact(subimage_channel_count as usize)
                            .map(to_gray_sub)
                            .map(to_f32),
                    );
                } else {
                    self.f32buf_subimage.extend(subimage.iter().map(ref_to_f32));
                }

                self.f32buf_search_image.clear();
                if to_grayscale && search_image_channel_count != 1 {
                    self.f32buf_search_image.extend(
                        search_image
                            .chunks_exact(search_image_channel_count as usize)
                            .map(to_gray_search)
                            .map(to_f32),
                    );
                } else {
                    self.f32buf_search_image
                        .extend(search_image.iter().map(ref_to_f32));
                }

                let simdeez_width = simdeez_width_runtime_select();
                let dist_function = if subimage_width % simdeez_width == 0 {
                    image_dist_simdeez_runtime_select
                } else {
                    image_dist_simdeez_with_remainder_runtime_select
                };

                let width_multiplier =
                    if to_grayscale { 1 } else { subimage_channel_count as usize };
                for y in (0..(search_height - subimage_height)).step_by(step_y) {
                    for x in (0..(search_width - subimage_width)).step_by(step_x) {
                        let dist = dist_function(
                            x * width_multiplier,
                            y,
                            &self.f32buf_search_image,
                            search_width * width_multiplier,
                            &self.f32buf_subimage,
                            subimage_width * width_multiplier,
                            subimage_height,
                        );
                        if dist < threshold {
                            self.positions_buffer.push((x, y, dist));
                        }
                    }
                }
            }
            Backend::Scalar {
                threshold,
                step_x,
                step_y,
            } => {
                let subimage_bytes: &[u8] = if to_grayscale && subimage_channel_count != 1 {
                    self.u8buf_subimage.clear();
                    self.u8buf_subimage.extend(
                        subimage
                            .chunks_exact(subimage_channel_count as usize)
                            .map(to_gray_sub),
                    );
                    &self.u8buf_subimage
                } else {
                    subimage
                };

                let search_bytes: &[u8] = if to_grayscale && search_image_channel_count != 1 {
                    self.u8buf_search_image.clear();
                    self.u8buf_search_image.extend(
                        search_image
                            .chunks_exact(search_image_channel_count as usize)
                            .map(to_gray_search),
                    );
                    &self.u8buf_search_image
                } else {
                    search_image
                };

                for y in (0..(search_height - subimage_height)).step_by(step_y) {
                    for x in (0..(search_width - subimage_width)).step_by(step_x) {
                        let dist = image_dist_naive(
                            (x, y),
                            (search_bytes, search_width),
                            (subimage_bytes, subimage_width, subimage_height),
                            if to_grayscale { 1 } else { subimage_channel_count as usize },
                        );
                        if dist < threshold {
                            self.positions_buffer.push((x, y, dist));
                        }
                    }
                }
            }
            #[cfg(feature = "opencv")]
            Backend::OpenCV { threshold } => {
                let subimage_ptr: *mut std::ffi::c_void =
                    if to_grayscale && subimage_channel_count != 1 {
                        self.u8buf_subimage.clear();
                        self.u8buf_subimage.extend(
                            subimage
                                .chunks_exact(subimage_channel_count as usize)
                                .map(to_gray_sub),
                        );
                        self.u8buf_subimage.as_mut_ptr() as *mut _
                    } else {
                        subimage.as_ptr() as *mut _
                    };

                let search_ptr: *mut std::ffi::c_void =
                    if to_grayscale && search_image_channel_count != 1 {
                        self.u8buf_search_image.clear();
                        self.u8buf_search_image.extend(
                            search_image
                                .chunks_exact(search_image_channel_count as usize)
                                .map(to_gray_search),
                        );
                        self.u8buf_search_image.as_mut_ptr() as *mut _
                    } else {
                        search_image.as_ptr() as *mut _
                    };

                let ch_count_to_mat_typ = |channels| match channels {
                    1 => opencv::core::CV_8UC1,
                    2 => opencv::core::CV_8UC2,
                    3 => opencv::core::CV_8UC3,
                    4 => opencv::core::CV_8UC4,
                    _ => panic!(
            "opencv matrices do not support more than 4 channels (Tried to use {} channels)",
            channels
          ),
                };
                let opencv_mat_typ_search: i32 = if to_grayscale {
                    opencv::core::CV_8UC1
                } else {
                    ch_count_to_mat_typ(search_image_channel_count)
                };
                let opencv_mat_typ_sub: i32 = if to_grayscale {
                    opencv::core::CV_8UC1
                } else {
                    ch_count_to_mat_typ(subimage_channel_count)
                };
                unsafe {
                    let mut out_mat = opencv::core::Mat::default();
                    opencv::imgproc::match_template(
                        &opencv::core::Mat::new_rows_cols_with_data(
                            search_height as i32,
                            search_width as i32,
                            opencv_mat_typ_search,
                            search_ptr,
                            0,
                        )
                        .unwrap(),
                        &opencv::core::Mat::new_rows_cols_with_data(
                            subimage_height as i32,
                            subimage_width as i32,
                            opencv_mat_typ_sub,
                            subimage_ptr,
                            0,
                        )
                        .unwrap(),
                        &mut out_mat,
                        opencv::imgproc::TM_SQDIFF_NORMED,
                        &opencv::core::no_array(),
                    )
                    .unwrap();

                    for (opencv::core::Point_ { x, y }, val) in out_mat.iter().unwrap() {
                        let val: f32 = val; // To help inference

                        if val < threshold {
                            self.positions_buffer.push((x as usize, y as usize, val));
                        }
                    }
                }
            }
        }

        self.prune_nearby_results(subimage_width, subimage_height);

        &self.positions_buffer
    }

    // TODO: Iterator API?
    // TODO: Allow custom strides w/ padding?

    /// Remove results that are too close together according to prune_[width|height]_scale
    /// prioritizing the ones with the lowest distance.
    fn prune_nearby_results(&mut self, subimage_width: usize, subimage_height: usize) {
        let width_threshold = (subimage_width as f32 * self.prune_width_scale) as isize;
        let height_threshold = (subimage_height as f32 * self.prune_height_scale) as isize;

        self.positions_buffer
            .sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let mut i = 0;
        while i < self.positions_buffer.len() {
            let a = self.positions_buffer[i];

            self.positions_buffer.retain(|b| {
                let dist = (
                    (b.0 as isize - a.0 as isize).abs(),
                    (b.1 as isize - a.1 as isize).abs(),
                );
                dist == (0, 0) || (dist.0 > width_threshold || dist.1 > height_threshold)
            });

            i += 1;
        }
    }

    /// This returns the same as the last value returned from [Self::find_subimage_positions],
    /// as long as you haven't modified them by calling [Self::most_recent_results_mut]
    pub fn most_recent_results(&self) -> &[(usize, usize, f32)] {
        &self.positions_buffer
    }

    /// Gives a mutable reference to the most recent results. Calling this after
    /// [Self::find_subimage_positions] gives you the same slice, but with mutable access. This can
    /// be useful if you want to sort the results without allocating a new Vec.
    ///
    /// For example, if you need to sort by y and then by x position:
    /// ```
    /// use find_subimage::{Image, SubImageFinderState};
    /// let (w, h) = (128, 128);
    /// let mut rgb_image = vec![0u8; w * h * 3];
    /// let (sub_w, sub_h) = (16, 16);
    /// let mut rgb_subimage = vec![0u8; sub_w * sub_h * 3];
    ///
    /// let mut finder = SubImageFinderState::new();
    /// finder.find_subimage_positions((&rgb_image, w, h), (&rgb_subimage, sub_w, sub_h), 3);
    ///
    /// let results = finder.most_recent_results_mut();
    /// results.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    /// ```
    pub fn most_recent_results_mut(&mut self) -> &mut [(usize, usize, f32)] {
        &mut self.positions_buffer
    }

    /// If for some reason you need an owned Vec of the results, you can
    /// use this function to avoid a copy and take ownership of the internal buffer.
    pub fn take_results_buffer(&mut self) -> Vec<(usize, usize, f32)> {
        std::mem::take(&mut self.positions_buffer)
    }
}

// I looked into std portable-simd but doing runtime detection with it seems way more complicated
// than the handy simdeez macro I'm pretty sure simdeez has UB in it though
// I may add a StaticTargetCpuSimd backend or something without runtime detection that expects users
// to compile with appropiate target cpu flags and uses portable-simd

use std::num::NonZeroU8;

#[cfg(feature = "simdeez")]
use simdeez::*;
#[cfg(feature = "simdeez")]
use simdeez::{avx2::*, scalar::*, sse2::*, sse41::*};
#[cfg(feature = "simdeez")]
simd_runtime_generate!(
    fn simdeez_width() -> usize {
        S::VF32_WIDTH
    }
);

macro_rules! make_simdeez_fn {
    ($with_remainder: expr, $fn_name: ident) => {
        #[cfg(feature = "simdeez")]
        simd_runtime_generate!(
            fn $fn_name(
                x_offset: usize,
                y_offset: usize,
                search_img: &[f32],
                search_w: usize,
                subimage: &[f32],
                w: usize,
                h: usize,
            ) -> f32 {
                #[cfg(not(feature = "checked-simdeez"))]
                let slice: fn(&[f32], _) -> &[f32] = |x, range| x.get_unchecked(range);
                #[cfg(feature = "checked-simdeez")]
                let slice: fn(&[f32], _) -> &[f32] = |x, range| &x[range];
                #[cfg(not(feature = "checked-simdeez"))]
                let slice_elem: fn(&[f32], _) -> &f32 = |x, idx| x.get_unchecked(idx);
                #[cfg(feature = "checked-simdeez")]
                let slice_elem: fn(&[f32], _) -> &f32 = |x, idx| &x[idx];

                // These 3 lines should do all the bounds checking we need
                // We use get_unchecked below
                let subimage = &subimage[..(w * h)];

                let search_img = &search_img[(x_offset + y_offset * search_w)..];
                let search_img = &search_img[..(h * search_w)];

                // [0.0; S::VF32_WIDTH] gave me a const generics error
                // In my case it's 8, 32 should be plenty conservative
                let zeroes = [0.0; 32];
                let mut res_simd = S::loadu_ps(&zeroes[0]);
                let mut res_scalar = 0.0f32;

                let simd_iters_per_row = w / S::VF32_WIDTH;
                let scalar_iters_per_row = w % S::VF32_WIDTH;

                for y in 0..h {
                    let row_sub = (y * w) as usize;
                    let row_search = (y * search_w) as usize;

                    let mut subimage = slice(subimage, row_sub..);
                    let mut search_img = slice(search_img, row_search..);

                    for _ in 0..simd_iters_per_row {
                        let search = S::loadu_ps(slice_elem(search_img, 0));
                        let sub = S::loadu_ps(slice_elem(subimage, 0));

                        let diff = S::sub_ps(sub, search);
                        let square = S::mul_ps(diff, diff);

                        res_simd = S::add_ps(res_simd, square);

                        subimage = slice(subimage, S::VF32_WIDTH..);
                        search_img = slice(search_img, S::VF32_WIDTH..);
                    }

                    if $with_remainder {
                        for i in 0..scalar_iters_per_row {
                            let search = slice_elem(search_img, i);
                            let sub = slice_elem(subimage, i);

                            let diff = sub - search;
                            let square = diff * diff;
                            res_scalar += square;
                        }
                    }
                }

                let res = S::horizontal_add_ps(res_simd) + res_scalar;

                //res.sqrt() / w as f32 / h as f32
                //res / (255.0 * 255.0) / w as f32 / h as f32
                (res / w as f32 / h as f32).sqrt() / 255.0
                //'res.sqrt() / ((w as f32 * h as f32).sqrt() * 255.0)
            }
        );
    };
}
make_simdeez_fn!(true, image_dist_simdeez_with_remainder);
make_simdeez_fn!(false, image_dist_simdeez);

fn image_dist_naive(
    (x_offset, y_offset): (usize, usize),
    (search_img, search_w): (&[u8], usize),
    (subimage, w, h): (&[u8], usize, usize),
    channel_count: usize,
) -> f32 {
    let subimage = &subimage[..w * h * channel_count];

    let search_stride = search_w * channel_count;
    let sub_stride = w * channel_count;

    let search_img = &search_img[x_offset * channel_count + y_offset * search_stride..];
    let search_img = &search_img[..h * search_stride];

    let calc_dist = |a, b| (a as f32 - b as f32).powi(2);
    let mut dist = 0.0f32;
    for y in 0..h {
        #[allow(clippy::identity_op)]
        for x in 0..sub_stride {
            let pos_sub = x + y * sub_stride;
            let pos_search = x + y * search_stride;

            dist += calc_dist(subimage[pos_sub], search_img[pos_search]);
        }
    }
    (dist / w as f32 / h as f32).sqrt() / 255.0
}

impl Default for SubImageFinderState {
    fn default() -> Self {
        Self::new()
    }
}
