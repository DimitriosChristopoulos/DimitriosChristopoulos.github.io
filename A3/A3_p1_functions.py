import cv2
import json
from harris import *

def label_image(img, unique_name, save_dir='part_1_results/'):
    image_to_label = img.copy()

    cv2.namedWindow("Image to Label")
    cv2.imshow("Image to Label", image_to_label)

    chosen_points = []
    done_labeling = False

    def record_points(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if len(chosen_points) < 8:
                cv2.circle(image_to_label, (x, y), 2, (0, 0, 255), -1)
                cv2.imshow('Image to Label', image_to_label)

                chosen_points.append((x, y))

    cv2.setMouseCallback("Image to Label", record_points)
    while (len(chosen_points) < 8 or not done_labeling) and cv2.getWindowProperty("Image to Label", cv2.WND_PROP_VISIBLE) >= 1:
        key_pressed = cv2.waitKey(10)
        if key_pressed == 13: # 13 is for enter key
            done_labeling = True

    cv2.destroyAllWindows()
    cv2.waitKey(10)

    cv2.imwrite(f'{save_dir}labeled_{unique_name}.jpg', image_to_label)

    with open(f'{save_dir}img_{unique_name}_points.json', 'w') as write_file:
        json.dump(chosen_points, write_file)
    write_file.close()

    return chosen_points


def compute_homography(img_1_pts, img_2_pts):
    A_matrix = []
    B_matrix = []
    for img_1_pt, img_2_pt in zip(img_1_pts, img_2_pts):
        x_prime = [img_1_pt[0], img_1_pt[1], 1, 0, 0, 0, -img_1_pt[0]*img_2_pt[0], -img_1_pt[1]*img_2_pt[0]]
        y_prime = [0, 0, 0, img_1_pt[0], img_1_pt[1], 1, -img_1_pt[0]*img_2_pt[1], -img_1_pt[1]*img_2_pt[1]]

        A_matrix.append(x_prime)
        A_matrix.append(y_prime)
        B_matrix.append(img_2_pt[0])
        B_matrix.append(img_2_pt[1])

    A_matrix = np.array(A_matrix, dtype=float)
    B_matrix = np.array(B_matrix, dtype=float)

    homography, residuals, rank, s = np.linalg.lstsq(A_matrix, B_matrix, rcond=None)

    homography = np.append(homography, 1).reshape(3, 3)

    return homography


def show_points(img_to_map_path, mapped, save_dir='part_1_results/'):
    background_img = cv2.imread(img_to_map_path)

    for point in mapped:
        cv2.circle(background_img, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

    cv2.imwrite(f'{save_dir}img_mapped.jpg', background_img)

    return


def get_new_img_size(img_1, img_2, homography):
    IMG_HEIGHT, IMG_WIDTH, _ = img_1.shape
    IMG_HEIGHT_2, IMG_WIDTH_2, _ = img_2.shape

    img_1_pts = np.array([[0, 0], [0, IMG_HEIGHT - 1], [IMG_WIDTH - 1, 0], [IMG_WIDTH - 1, IMG_HEIGHT - 1]])
    img_1_pts = np.column_stack([img_1_pts, np.ones(4)])

    img_1_pts_mapped = (homography @ img_1_pts.T).T
    img_1_pts_mapped = img_1_pts_mapped[:, :2] / img_1_pts_mapped[:, 2:3]

    img_1_pts_mapped = np.round(img_1_pts_mapped).astype(int)

    ADD_WIDTH_LEFT = abs(min(np.min(img_1_pts_mapped[:, 0]), 0))
    ADD_WIDTH_RIGHT = max(np.max(img_1_pts_mapped[:, 0]) - (IMG_WIDTH_2 - 1), 0)

    ADD_HEIGHT_UP = abs(min(np.min(img_1_pts_mapped[:, 1]), 0))
    ADD_HEIGHT_DOWN = max(np.max(img_1_pts_mapped[:, 1]) - (IMG_HEIGHT_2 - 1), 0)

    NEW_WIDTH = IMG_WIDTH_2 + ADD_WIDTH_LEFT + ADD_WIDTH_RIGHT
    NEW_HEIGHT = IMG_HEIGHT_2 + ADD_HEIGHT_UP + ADD_HEIGHT_DOWN

    return (NEW_HEIGHT, NEW_WIDTH), (ADD_HEIGHT_UP, ADD_WIDTH_LEFT)


def bilinear_interpolation(px, img):
    px_floor = np.floor(px).astype(int)
    px_ceil = px_floor + 1

    intensities = []
    mat_1 = np.array([px_ceil[0] - px[0], px[0] - px_floor[0]])
    mat_3 = np.array([px_ceil[1] - px[1], px[1] - px_floor[1]])
    for c in range(img.shape[2]):
        mat_2 = np.array([[img[px_floor[1]][px_floor[0]][c], img[px_ceil[1]][px_floor[0]][c]],
                         [img[px_floor[1]][px_ceil[0]][c], img[px_ceil[1]][px_ceil[0]][c]]])

        intensities.append(mat_1 @ mat_2 @ mat_3)

    return intensities


def create_alpha_mask(img, mask_name='', save_dir='part_1_results/'):
    img_mask = (np.sum(img, axis=2) > 0).astype(np.uint8)
    mask_dist = cv2.distanceTransform(img_mask, cv2.DIST_L2, 5)
    mask_norm = mask_dist / (mask_dist.max() + 1e-8)

    cv2.imwrite(f'{save_dir}mask_{mask_name}_dist.png', mask_dist)
    cv2.imwrite(f'{save_dir}mask_{mask_name}_norm.png', mask_norm)

    return mask_norm


def gain_correct_img(img_1, img_2, IMG_HEIGHT, IMG_WIDTH):
    overlap_mask = (np.sum(img_1, axis=2) > 0) & (np.sum(img_2, axis=2) > 0)

    corrected_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    for c in range(3):
        gain_val = np.mean(img_1[overlap_mask][:, c]) / (np.mean(img_2[overlap_mask][:, c]) + 1e-8)
        corrected_img[:, :, c] = img_2[:, :, c] * gain_val

    return np.clip(corrected_img, 0, 255).astype(np.uint8)


def warp_2_imgs(img_1, img_2, homography, method='nearest_neighbor', img_name='warped_img', create_mosaic=False, gain_correct=False, save_dir='part_1_results/'):
    assert method=='nearest_neighbor' or method=='bilinear', 'Method must be nearest_neighbor or bilinear'

    homography_inv = np.linalg.inv(homography)

    (NEW_HEIGHT, NEW_WIDTH), (ADD_HEIGHT_UP, ADD_WIDTH_LEFT) = get_new_img_size(img_1, img_2, homography)

    new_img = np.zeros((NEW_HEIGHT, NEW_WIDTH, 3), dtype=np.uint8)
    new_img_2 = np.zeros((NEW_HEIGHT, NEW_WIDTH, 3), dtype=np.uint8)

    y_px, x_px = np.mgrid[0:new_img.shape[0], 0:new_img.shape[1]]
    y_px_2, x_px_2 = np.mgrid[0:img_2.shape[0], 0:img_2.shape[1]]

    all_points = np.stack([x_px.ravel() - ADD_WIDTH_LEFT, y_px.ravel() - ADD_HEIGHT_UP, np.ones(x_px.size)])
    original_points = np.stack([x_px.ravel(), y_px.ravel()]).T

    orig_2 = np.stack([x_px_2.ravel(), y_px_2.ravel()]).T

    all_points_mapped = (homography_inv @ all_points).T
    all_points_mapped = all_points_mapped[:, :2] / all_points_mapped[:, 2:3]

    within_image_mask_1 = (0 <= all_points_mapped[:, 0]) & (all_points_mapped[:, 0] < (img_1.shape[1] - 1))
    within_image_mask_2 = (0 <= all_points_mapped[:, 1]) & (all_points_mapped[:, 1] < (img_1.shape[0] - 1))

    final_mask = within_image_mask_1 & within_image_mask_2

    mapped_points_used = all_points_mapped[final_mask]
    original_points_used = original_points[final_mask]

    if method == 'nearest_neighbor':
        mapped_points_used = np.round(mapped_points_used).astype(int)

    for orig_px, mapped_px in zip(original_points_used, mapped_points_used):
        if method == 'nearest_neighbor':
            mapped_img_val = img_1[mapped_px[1], mapped_px[0]]
        elif method == 'bilinear':
            mapped_img_val = bilinear_interpolation(mapped_px, img_1)

        new_img[orig_px[1], orig_px[0]] = mapped_img_val

    for orig_px_2 in orig_2:
        new_img_2[orig_px_2[1] + ADD_HEIGHT_UP, orig_px_2[0] + ADD_WIDTH_LEFT] = img_2[orig_px_2[1], orig_px_2[0]]

    cv2.imwrite(f'{save_dir}' + img_name + '_1.jpg', new_img)
    cv2.imwrite(f'{save_dir}' + img_name + '_2.jpg', new_img_2)

    if create_mosaic:
        if gain_correct:
            new_img = gain_correct_img(new_img_2, new_img, NEW_HEIGHT, NEW_WIDTH)

        img_1_alpha = create_alpha_mask(new_img, mask_name=f'{img_name}_1', save_dir=save_dir)
        img_2_alpha = create_alpha_mask(new_img_2, mask_name=f'{img_name}_2', save_dir=save_dir)

        final_mosaic = np.zeros((NEW_HEIGHT, NEW_WIDTH, 3), dtype=np.uint8)

        for x_px in range(NEW_WIDTH):
            for y_px in range(NEW_HEIGHT):
                num = new_img[y_px,x_px] * img_1_alpha[y_px,x_px] + new_img_2[y_px,x_px] * img_2_alpha[y_px,x_px]
                den = img_1_alpha[y_px, x_px] + img_2_alpha[y_px, x_px]

                final_mosaic[y_px, x_px] = num / (den + 1e-8)

        cv2.imwrite(f'{save_dir}' + img_name + '_mosaic.jpg', final_mosaic)

        return new_img, new_img_2, final_mosaic

    return new_img, new_img_2
