import cv2
import numpy as np
import cv2 as cv
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import shift, rotate, zoom
import scipy
import os


def show_img(img_to_show, title):
    cv.imshow(f'{title}', img_to_show)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

    return


def save_img(img_to_save, title):
    cv.imwrite(f"{title}.jpg", np.clip(img_to_save, 0, 255).astype(np.uint8))

    return


def pad_image(image, kernel): # Chose to use 2 for loops for padding instead of np.pad, figured I would implement from
                              # scratch since this part of the assignment is about implementation
    HEIGHT, WIDTH = image.shape
    K_HEIGHT, K_WIDTH = kernel.shape

    pad_y = (K_HEIGHT - 1) // 2
    pad_x = (K_WIDTH - 1) // 2
    padded_img = np.zeros((HEIGHT + 2 * pad_y, WIDTH + 2 * pad_x))

    for y_px in range(HEIGHT):
        for x_px in range(WIDTH):
            padded_img[y_px + pad_y][x_px + pad_x] = image[y_px][x_px]

    return padded_img


def convolve_image(image, kernel):
    HEIGHT, WIDTH = image.shape
    K_HEIGHT, K_WIDTH = kernel.shape

    pad_y = K_HEIGHT // 2
    pad_x = K_WIDTH // 2

    kernel = np.flip(kernel)
    padded_img = pad_image(image, kernel)

    convolved_img = np.zeros((HEIGHT, WIDTH))
    for y_px in range(HEIGHT):
        for x_px in range(WIDTH):
            conv_val = 0
            for y_inc in range(-pad_y, pad_y + 1):
                for x_inc in range(-pad_x, pad_x + 1):
                    conv_val += padded_img[y_px + y_inc + pad_y][x_px + x_inc + pad_x] * kernel[y_inc + pad_y][x_inc + pad_x]

            convolved_img[y_px][x_px] = conv_val

    return convolved_img


def convolve_image_opt(image, kernel):
    HEIGHT, WIDTH = image.shape
    K_HEIGHT, K_WIDTH = kernel.shape

    pad_y = K_HEIGHT // 2
    pad_x = K_WIDTH // 2

    kernel = np.flip(kernel)
    padded_img = pad_image(image, kernel)

    convolved_img = np.zeros((HEIGHT, WIDTH))
    for y_px in range(HEIGHT):
        for x_px in range(WIDTH):
            img_patch = padded_img[y_px:y_px + 2 * pad_y + 1, x_px:x_px + 2 * pad_x + 1]
            convolved_img[y_px][x_px] = np.sum(img_patch * kernel)

    return convolved_img


def compute_grad_mag(dx_img, dy_img):
    return np.sqrt(dx_img ** 2 + dy_img ** 2)


def unsharp_mask_filter(image, gaussian=(cv.getGaussianKernel(ksize=9, sigma=0) @ cv.getGaussianKernel(ksize=9, sigma=0).T)):
    image = image.astype(np.float32)
    blur_image = image.copy()

    for channel in range(3):
        blur_image[:, :, channel] = scipy.signal.convolve2d(image[:, :, channel], gaussian, mode='same')
    high_freq_img = image - blur_image

    return image + high_freq_img


def save_image(image, filename, title, color=False, cmap="gray"):
    fig, ax = plt.subplots()

    if color and image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    ax.imshow(image, cmap=None if color else cmap)
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        os.path.join('2.2/results', filename),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


def _select_two_points(image, title):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("image")
    points = plt.ginput(2, timeout=0)
    plt.close(fig)

    return np.asarray(points, dtype=float)


def _translate_to_center(image, points):
    h, w = image.shape[:2]
    midpoint = points.mean(axis=0)  # (x, y)
    dx = w / 2 - midpoint[0]
    dy = h / 2 - midpoint[1]

    if image.ndim == 2:
        return shift(image, shift=(dy, dx), order=1, mode="constant", cval=0)
    return shift(image, shift=(dy, dx, 0), order=1, mode="constant", cval=0)


def _center_crop(image, target_h, target_w):
    h, w = image.shape[:2]
    y0 = max(0, (h - target_h) // 2)
    x0 = max(0, (w - target_w) // 2)

    return image[y0:y0 + target_h, x0:x0 + target_w, ...]


def align_images(im1, im2, save_process=True):
    im1 = np.asarray(im1, dtype=np.float32)
    im2 = np.asarray(im2, dtype=np.float32)

    pts1 = _select_two_points(im1, "Select two corresponding points in image 1")
    pts2 = _select_two_points(im2, "Select the same two points in image 2")

    len1 = np.linalg.norm(pts1[1] - pts1[0])
    len2 = np.linalg.norm(pts2[1] - pts2[0])

    im1 = _translate_to_center(im1, pts1)
    im2 = _translate_to_center(im2, pts2)

    scale = len2 / len1

    if scale < 1:
        zoom_factors = (scale, scale) if im1.ndim == 2 else (scale, scale, 1)
        im1 = zoom(im1, zoom_factors, order=1, mode="constant", cval=0)
    else:
        factor = 1 / scale
        zoom_factors = (factor, factor) if im2.ndim == 2 else (factor, factor, 1)
        im2 = zoom(im2, zoom_factors, order=1, mode="constant", cval=0)

    theta1 = np.arctan2(
        -(pts1[1, 1] - pts1[0, 1]),
        pts1[1, 0] - pts1[0, 0]
    )
    theta2 = np.arctan2(
        -(pts2[1, 1] - pts2[0, 1]),
        pts2[1, 0] - pts2[0, 0]
    )

    delta_degrees = np.degrees(theta2 - theta1)

    im1 = rotate(
        im1,
        delta_degrees,
        axes=(1, 0),
        reshape=True,
        order=1,
        mode="constant",
        cval=0
    )

    h = min(im1.shape[0], im2.shape[0])
    w = min(im1.shape[1], im2.shape[1])

    im1 = _center_crop(im1, h, w)
    im2 = _center_crop(im2, h, w)

    if save_process:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for ax, image, title in zip(
            axes,
            [im1, im2],
            ["Aligned image 1", "Aligned image 2"]
        ):
            display_img = (
                cv.cvtColor(image, cv.COLOR_BGR2RGB)
                if image.ndim == 3 else image
            )

            ax.imshow(
                display_img,
                cmap="gray" if image.ndim == 2 else None
            )
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(
            os.path.join('2.2/results', "aligned_images.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.show()
        plt.close(fig)

    return im1, im2


def make_hybrid_image(img_path_1, img_path_2, low_thresh, high_thresh, using_colors=False, name="hybrid", save_process=False):
    if using_colors:
        im1 = cv.imread(img_path_1, cv.IMREAD_COLOR)
        im2 = cv.imread(img_path_2, cv.IMREAD_COLOR)
    else:
        im1 = cv.imread(img_path_1, cv.IMREAD_GRAYSCALE)
        im2 = cv.imread(img_path_2, cv.IMREAD_GRAYSCALE)

    im1 = im1.astype(np.float32) / 255.0
    im2 = im2.astype(np.float32) / 255.0

    save_image(im1, f"{name}_original1.png",
               "Original image 1", using_colors)
    save_image(im2, f"{name}_original2.png",
               "Original image 2", using_colors)

    im2, im1 = align_images(im2, im1, save_process=save_process)

    if save_process:
        save_image(im1, f"{name}_aligned1.png",
                   "Aligned image 1", using_colors)
        save_image(im2, f"{name}_aligned2.png",
                   "Aligned image 2", using_colors)

    hybrid = blur_and_combine(
        im1, im2,
        low_thresh, high_thresh,
        color_imgs=using_colors,
        save_process=save_process,
        name=name
    )

    save_image(
        hybrid, f"{name}_final.png",
        "Final hybrid image", using_colors
    )

    return hybrid


def save_fourier(image, filename, title):
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    spectrum = np.log1p(np.abs(np.fft.fftshift(
        np.fft.fft2(image)
    )))

    save_image(spectrum, filename, title)


def blur_and_combine(img_1, img_2, low_sigma, high_sigma, color_imgs=False, save_process=False, name="hybrid"):
    gaussian_low = cv.getGaussianKernel(13, low_sigma)
    gaussian_low = gaussian_low @ gaussian_low.T

    gaussian_high = cv.getGaussianKernel(13, high_sigma)
    gaussian_high = gaussian_high @ gaussian_high.T

    img_1 = img_1.astype(np.float32)
    img_2 = img_2.astype(np.float32)

    blur_img_1 = img_1.copy()
    blur_img_2 = img_2.copy()

    if color_imgs:
        for channel in range(3):
            blur_img_1[:, :, channel] = scipy.signal.convolve2d(
                img_1[:, :, channel], gaussian_high, mode="same"
            )
            blur_img_2[:, :, channel] = scipy.signal.convolve2d(
                img_2[:, :, channel], gaussian_low, mode="same"
            )
    else:
        blur_img_1 = scipy.signal.convolve2d(
            img_1, gaussian_high, mode="same"
        )
        blur_img_2 = scipy.signal.convolve2d(
            img_2, gaussian_low, mode="same"
        )

    high_freq_img = img_1 - blur_img_1
    low_freq_img = blur_img_2
    hybrid_img = high_freq_img + low_freq_img

    if save_process:
        save_image(
            blur_img_1, f"{name}_high_blur.png",
            f"Image 1 blurred (sigma={high_sigma})", color_imgs
        )
        save_image(
            high_freq_img, f"{name}_high_frequencies.png",
            "High-frequency result", color_imgs
        )
        save_image(
            low_freq_img, f"{name}_low_frequencies.png",
            f"Low-frequency result (sigma={low_sigma})", color_imgs
        )

        save_fourier(
            img_1, f"{name}_fourier_original1.png",
            "Fourier transform: image 1"
        )
        save_fourier(
            img_2, f"{name}_fourier_original2.png",
            "Fourier transform: image 2"
        )
        save_fourier(
            high_freq_img, f"{name}_fourier_high.png",
            "Fourier transform: high-frequency result"
        )
        save_fourier(
            low_freq_img, f"{name}_fourier_low.png",
            "Fourier transform: low-frequency result"
        )

        save_image(
            hybrid_img, f"{name}_hybrid.png",
            "Final hybrid image", color_imgs
        )

        with open(os.path.join('2.2/results/', f"{name}_cutoffs.txt"), "w") as f:
            f.write(f"Low-frequency sigma: {low_sigma}\n")
            f.write(f"High-frequency sigma: {high_sigma}\n")
            f.write("Gaussian kernel size: 13 x 13\n")

    return hybrid_img


def gaussian_stack(image, num_levels, color_imgs=False):
    images_list = [image.copy()]

    if not color_imgs:
        for increment in range(num_levels-1):
            gaussian_1d = cv.getGaussianKernel(ksize=6*2**increment+1, sigma=2**increment)
            blurred = scipy.signal.convolve2d(images_list[-1], gaussian_1d, mode="same", boundary='symm')
            blurred = scipy.signal.convolve2d(blurred, gaussian_1d.T, mode="same", boundary='symm')
            images_list.append(blurred)

    else:
        for increment in range(num_levels-1):
            gaussian_1d = cv.getGaussianKernel(ksize=6*2**increment+1, sigma=2**increment)
            blur_img = images_list[-1].copy()
            for channel in range(3):
                blurred = scipy.signal.convolve2d(blur_img[:, :, channel], gaussian_1d, mode="same", boundary='symm')
                blur_img[:, :, channel] = scipy.signal.convolve2d(blurred, gaussian_1d.T, mode="same", boundary='symm')
            images_list.append(blur_img)

    return np.array(images_list)


def laplacian_stack(gauss_stack):
    laplace_stack = []

    for i in range(len(gauss_stack)-1):
        laplace_stack.append(gauss_stack[i] - gauss_stack[i+1])
    laplace_stack.append(gauss_stack[-1])

    return np.array(laplace_stack)


def show_return_stacks(img_path, num_levels, mask=False, color=False):
    if mask:
        img = img_path
    elif color:
        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    gauss_stack = gaussian_stack(img, num_levels, color_imgs=color)
    lapl_stack = laplacian_stack(gauss_stack)

    fig, axes = plt.subplots(2, num_levels, figsize=(15, 6))

    for i in range(num_levels):
        axes[0, i].imshow(gauss_stack[i], cmap="gray")
        axes[0, i].set_title(f"Gaussian - Level {i}")

        # Normalize only for visualization
        lap = lapl_stack[i]
        lap_display = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)

        axes[1, i].imshow(lap_display, cmap="gray")
        axes[1, i].set_title(f"Laplacian - Level {i}")

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

    return gauss_stack, lapl_stack


def build_mask(img_shape, orientation='horizontal'):
    mask = np.zeros(img_shape)
    if orientation == 'vertical':
        mask[:, :mask.shape[1] // 2] = 1

    elif orientation == 'horizontal':
        mask[:mask.shape[0] // 2, :] = 1

    else:
        raise ValueError('Unknown orientation')

    return mask


def blend_img(laplace_stack_1, laplace_stack_2, gauss_mask):
    blended_stack_vals = []
    for i in range(gauss_mask.shape[0]):
        m = gauss_mask[i][..., None] if laplace_stack_1.ndim == 4 else gauss_mask[i]
        blended_stack_vals.append(m * laplace_stack_1[i] + (1 - m) * laplace_stack_2[i])

    return np.array(blended_stack_vals)
