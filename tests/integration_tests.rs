extern crate image;

use find_subimage::SubImageFinderState;

// Note: If you modify this, please also modify it's copy in ./examples/subimager-finder.rs
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // TODO: Make this iterate and work on all images in /tests/img
        // with an established naming scheme for search/subimage/expected filenames like
        // ferris_search + ferris_subimage + ferris_expected
        // Will probably also want to add a way to automatically reconstruct expected images from
        // the other 2, and possibly have multiple expected images with some other naming scheme ?
        // (grayscale, settings, backend, etc)
        // Might also want a way to output a temporary comparison image when a test fails? (Which
        // would probably be in .gitignore)
        /*
        let paths = std::fs::read_dir("./tests/img").unwrap();

        let mut files = vec![];
        for path in paths {
            if let Ok(path) = path {
                let path = path.path();
                if let Some(file_name) = path.file_stem() {
                    files.push(path);
                }
            }
        }
        */

        let subimage = image::open("tests/img/ferris_eyes.png")
            .expect("Failed opening/parsing subimage file")
            .to_rgb8();
        let mut search_image = image::open("tests/img/ferris.png")
            .expect("Failed opening/parsing search image file")
            .to_rgb8();

        let to_tuple: fn(&image::ImageBuffer<_, _>) -> (&Vec<u8>, usize, usize) =
            |img| (img.as_raw(), img.width() as usize, img.height() as usize);

        let mut state = SubImageFinderState::new();
        let results_as_grayscale = state.find_subimage_positions_as_grayscale(
            to_tuple(&search_image),
            to_tuple(&subimage),
            3,
            None,
        );

        draw_rectangle_on(
            &mut search_image,
            results_as_grayscale
                .get(0)
                .map(|(x, y, _)| (*x as u32, *y as u32))
                .unwrap(),
            (subimage.width(), subimage.height()),
        );

        search_image
            .save("tests/img/tmp_test_output_ferris.png")
            .unwrap();

        assert_eq!(
            image::open("tests/img/ferris_eyes_matched.png")
                .unwrap()
                .to_rgb8()
                .as_raw()
                .as_slice(),
            search_image.as_raw().as_slice()
        );
    }
}
