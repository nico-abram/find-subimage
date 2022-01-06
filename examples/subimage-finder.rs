extern crate image;

use find_subimage::{Backend, SubImageFinderState, NONOPENCV_DEFAULT_THRESHOLD};

fn print_help() {
    const HELP: &str = r#"
Usage: subimage-finder [flags] search-image-file subimage-file
Flags:
-backend [runtime_simd | scalar]
-threshold {f32}
-pruning {prune_w f32} {prune_h f32}
-step_xy {step_x usize} {step_y usize}
-print {bool}
-out {filename string}
"#;
    println!("{}", HELP);
}

fn main() {
    let (
        backend,
        threshold,
        pruning,
        stepping,
        out_file,
        search_image_path_os_str,
        subimage_path_os_str,
    ): (
        Option<Backend>,
        Option<f32>,
        Option<(f32, f32)>,
        Option<(usize, usize)>,
        Option<std::ffi::OsString>,
        std::ffi::OsString,
        std::ffi::OsString,
    ) = parse_args();

    let mut state = SubImageFinderState::new();

    if let Some(backend) = backend {
        state.set_backend(backend);
    }
    if let Some((w, h)) = pruning {
        state.set_pruning(w, h);
    }
    if let Some((x, y)) = stepping {
        match state.backend_mut() {
            #[cfg(feature = "opencv")]
            Backend::OpenCV { .. } => {
                print_help();
                panic!("The opencv backend does not support stepping");
            }
            #[cfg(feature = "simdeez")]
            Backend::RuntimeDetectedSimd { step_x, step_y, .. } => {
                *step_x = x;
                *step_y = y;
            }
            Backend::Scalar { step_x, step_y, .. } => {
                *step_x = x;
                *step_y = y;
            }
        }
    }
    if let Some(new_threshold) = threshold {
        match state.backend_mut() {
            #[cfg(feature = "opencv")]
            Backend::OpenCV { threshold, .. } => {
                *threshold = new_threshold;
            }
            #[cfg(feature = "simdeez")]
            Backend::RuntimeDetectedSimd { threshold, .. } => {
                *threshold = new_threshold;
            }
            Backend::Scalar { threshold, .. } => {
                *threshold = new_threshold;
            }
        }
    }

    let subimage = image::open(subimage_path_os_str)
        .expect("Failed opening/parsing subimage file")
        .to_rgb8();
    let mut search_image = image::open(search_image_path_os_str)
        .expect("Failed opening/parsing search image file")
        .to_rgb8();

    let to_tuple: fn(&image::ImageBuffer<_, _>) -> (&Vec<u8>, usize, usize) =
        |img| (img.as_raw(), img.width() as usize, img.height() as usize);

    let results = state
        .find_subimage_positions(to_tuple(&search_image), to_tuple(&subimage), 3)
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    println!("results: {:?}", &results);

    let results_as_grayscale = state.find_subimage_positions_as_grayscale(
        to_tuple(&search_image),
        to_tuple(&subimage),
        3,
        None,
    );
    println!("results_as_grayscale: {:?}", results_as_grayscale);

    if let Some(out_file) = out_file {
        if let Some((x, y, _)) = results.get(0) {
            draw_rectangle_on(
                &mut search_image,
                (*x as u32, *y as u32),
                (subimage.width(), subimage.height()),
            );
            search_image.save(out_file).unwrap();
        }
    }
}

fn parse_args() -> (
    Option<Backend>,
    Option<f32>,
    Option<(f32, f32)>,
    Option<(usize, usize)>,
    Option<std::ffi::OsString>,
    std::ffi::OsString,
    std::ffi::OsString,
) {
    let mut args = std::env::args_os().into_iter().skip(1).collect::<Vec<_>>();

    let mut backend: Option<Backend> = None;
    let mut threshold: Option<f32> = None;
    let mut pruning: Option<(f32, f32)> = None;
    let mut stepping: Option<(usize, usize)> = None;
    let mut out_file: Option<std::ffi::OsString> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.to_str() {
            Some("-backend") => {
                let backend_str = args[i + 1].clone();
                args.remove(i);
                args.remove(i);
                backend = Some(match backend_str.to_str() {
                    #[cfg(feature = "simdeez")]
                    Some("runtime_simd") => Backend::RuntimeDetectedSimd {
                        step_x: 1,
                        step_y: 1,
                        threshold: NONOPENCV_DEFAULT_THRESHOLD,
                    },
                    Some("scalar") => Backend::Scalar {
                        step_x: 1,
                        step_y: 1,
                        threshold: NONOPENCV_DEFAULT_THRESHOLD,
                    },
                    #[cfg(feature = "opencv")]
                    Some("opencv") => Backend::OpenCV {
                        threshold: find_subimage::OPENCV_DEFAULT_THRESHOLD,
                    },
                    _ => {
                        print_help();
                        panic!(
                            "Invalid backend string {:?} (to_str: {:?})",
                            backend_str,
                            backend_str.to_str()
                        )
                    }
                });
            }
            Some("-pruning") => {
                let prune_w_str = args[i + 1].clone();
                let prune_h_str = args[i + 2].clone();
                args.remove(i);
                args.remove(i);
                args.remove(i);
                pruning = Some((
                    prune_w_str
                        .to_str()
                        .map(|prune_w| prune_w.parse::<f32>().unwrap())
                        .unwrap(),
                    prune_h_str
                        .to_str()
                        .map(|prune_h| prune_h.parse::<f32>().unwrap())
                        .unwrap(),
                ));
            }
            Some("-step_xy") => {
                let step_x_str = args[i + 1].clone();
                let step_y_str = args[i + 2].clone();
                args.remove(i);
                args.remove(i);
                args.remove(i);
                stepping = Some((
                    step_x_str
                        .to_str()
                        .map(|step_x| step_x.parse::<usize>().unwrap())
                        .unwrap(),
                    step_y_str
                        .to_str()
                        .map(|step_y| step_y.parse::<usize>().unwrap())
                        .unwrap(),
                ));
            }
            Some("-threshold") => {
                let threshold_str = args[i + 1].clone();
                args.remove(i);
                args.remove(i);
                threshold = threshold_str
                    .to_str()
                    .map(|threshold| threshold.parse::<f32>().unwrap());
            }
            Some("-out") => {
                out_file = Some(args[i + 1].clone());
                args.remove(i);
                args.remove(i);
            }
            _ => {
                i += 1;
            }
        }
    }

    if args.len() != 2 {
        print_help();
        panic!("More than 2 arguments remaining after flags: {:?}", args);
    }
    let search_image_path_os_str = args[0].clone();
    let subimage_path_os_str = args[1].clone();

    (
        backend,
        threshold,
        pruning,
        stepping,
        out_file,
        search_image_path_os_str,
        subimage_path_os_str,
    )
}

// Note: If you modify this, please also modify it's copy in ./tests/integration_tests.rs
fn draw_rectangle_on(
    img: &mut image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    (x, y): (u32, u32),
    (w, h): (u32, u32),
) {
    let border_col = image::Rgb([255u8, 0, 0]);

    const LINE_THICKNESS: u32 = 4;
    // Vertical line at (x,y)
    for off_x in 0..LINE_THICKNESS {
        for off_y in 0..h {
            *img.get_pixel_mut(x + off_x, y + off_y) = border_col;
        }
    }
    // Horizontal line at (x,y)
    for off_y in 0..LINE_THICKNESS {
        for off_x in 0..w {
            *img.get_pixel_mut(x + off_x, y + off_y) = border_col;
        }
    }
    // Vertical line at (x+w,y)
    for off_x in 0..LINE_THICKNESS {
        for off_y in 0..(h + 1) {
            *img.get_pixel_mut(x + off_x + w, y + off_y) = border_col;
        }
    }
    // Horizontal line at (x,y+h)
    for off_y in 0..LINE_THICKNESS {
        for off_x in 0..(w + 1) {
            *img.get_pixel_mut(x + off_x, y + off_y + h) = border_col;
        }
    }
}
