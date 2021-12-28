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
"#;
  println!("{}", HELP);
}

fn main() {
  let mut args = std::env::args_os().into_iter().skip(1).collect::<Vec<_>>();

  let mut backend: Option<Backend> = None;
  let mut threshold: Option<f32> = None;
  let mut pruning: Option<(f32, f32)> = None;
  let mut stepping: Option<(usize, usize)> = None;

  let mut i = 0;
  while i < args.len() {
    let arg = &args[i];
    match arg.to_str() {
      Some("-backend") => {
        let backend_str = args[i + 1].clone();
        args.remove(i);
        args.remove(i + 1);
        i += 1;
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
        args.remove(i + 1);
        args.remove(i + 2);
        i += 2;
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
        args.remove(i + 1);
        args.remove(i + 2);
        i += 1;
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
        args.remove(i + 1);
        i += 1;
        threshold = threshold_str
          .to_str()
          .map(|threshold| threshold.parse::<f32>().unwrap());
      }
      _ => {}
    }

    i += 1;
  }

  if args.len() != 2 {
    print_help();
    panic!("More than 2 arguments remaining after flags: {:?}", args);
  }
  let search_image_path_os_str = args[0].clone();
  let subimage_path_os_str = args[1].clone();

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

  let subimage = &image::open(subimage_path_os_str)
    .expect("Failed opening/parsing subimage file")
    .to_rgb8();
  let search_image = &image::open(search_image_path_os_str)
    .expect("Failed opening/parsing search image file")
    .to_rgb8();

  let to_tuple: fn(&image::ImageBuffer<_, _>) -> (&Vec<u8>, usize, usize) =
    |img| (img.as_raw(), img.width() as usize, img.height() as usize);

  let results = state.find_subimage_positions(to_tuple(subimage), to_tuple(search_image), 3);
  println!("results: {:?}", results);

  let results_as_grayscale =
    state.find_subimage_positions_as_grayscale(to_tuple(subimage), to_tuple(search_image), 3, None);
  println!("results: {:?}", results_as_grayscale);
}
