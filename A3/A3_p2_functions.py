from A3_p1_functions import *

def plot_harris_corners(img_path, h, coords, corners_thresh=0, img_name='corners'):
    img = cv2.imread(img_path)

    strength_thresh = corners_thresh * h.max()
    for corner in coords.T:
        if h[corner[0]][corner[1]] >= strength_thresh:
            cv2.circle(img, (corner[1], corner[0]), 2, (0, 0, 255), -1)

    cv2.imwrite(f'part_2_results/{img_name}_{corners_thresh}.jpg', img)


def filter_corners(h, coords, img, thresh_rel=0.01, window=40):
    valid = (img.sum(axis=2) > 0).astype(np.uint8)
    kernel = np.ones((window + 1, window + 1), np.uint8)

    valid = cv2.erode(valid, kernel)
    keep = valid[coords[0], coords[1]].astype(bool)

    coords = coords[:, keep]

    strengths = h[coords[0], coords[1]]
    coords = coords[:, strengths > thresh_rel * strengths.max()]

    return coords


def ANMS(h, coords, num_points=500, c_robust=0.9):
    y_vals, x_vals = coords
    h_vals = h[y_vals, x_vals]

    all_points = coords.T.astype(np.float32)
    N = h_vals.shape[0]

    diff_matrix = all_points.reshape(N, 1, 2) - all_points.reshape(1, N, 2)
    diff_matrix = (diff_matrix ** 2).sum(axis=2)

    suppress_mask = h_vals.reshape(N, 1) < c_robust * h_vals.reshape(1, N)
    suppress_mask = ~suppress_mask

    diff_matrix[suppress_mask] = np.inf
    min_dists = diff_matrix.min(axis=1)

    indexes_to_keep = np.argsort(-min_dists)[:num_points]

    return coords[:, indexes_to_keep]


def feature_descript_extract(img_bw, ANMS_coords):
    blurred_img = cv2.GaussianBlur(img_bw.astype(np.float32), (11, 11), 0)

    patches = []
    for coord in ANMS_coords.T:
        patches.append(blurred_img[coord[0] - 20:coord[0] + 20, coord[1] - 20:coord[1] + 20][::5, ::5])

    patches = np.array(patches)

    patch_means = np.mean(patches, axis=(1, 2), keepdims=True)
    patch_stds = np.std(patches, axis=(1, 2), keepdims=True)

    patches_norm = (patches - patch_means) / (patch_stds + 1e-8)
    patches_vecs = patches_norm.reshape(patches_norm.shape[0], -1)

    return patches_vecs, patches


def match_features(vecs_left, vecs_right, eps_thresh=0.675):
    N = vecs_left.shape[0]
    M = vecs_right.shape[0]

    dist_matrix = np.sum((vecs_left.reshape(N, 1, -1) - vecs_right.reshape(1, M, -1)) ** 2, axis=2)

    arg_sorted_row = np.argsort(dist_matrix, axis=1)

    matches = np.ones((N, 2), dtype=int)

    matches[:, 0] = np.arange(N)
    matches[:, 1] = arg_sorted_row[:, 0]

    rows = np.arange(N)
    e_1 = dist_matrix[rows, arg_sorted_row[:, 0]]
    e_2 = dist_matrix[rows, arg_sorted_row[:, 1]]

    matches_mask = e_1 / (e_2 + 1e-8) < eps_thresh

    return matches[matches_mask]


def RANSAC4(matched_pairs, coords_1_ordered, coords_2_ordered, NUM_ITERATIONS=5000, ERR_THRESH=4):
    rng = np.random.default_rng()
    N = matched_pairs.shape[0]

    max_inliers = -1
    max_inliers_mask = np.nan
    for _ in range(NUM_ITERATIONS):
        left_coords = []
        right_coords = []
        rand_nums = rng.choice(N, size=4, replace=False)
        for rand_num in rand_nums:
            left_coords.append(coords_1_ordered[matched_pairs[rand_num, 0]])
            right_coords.append(coords_2_ordered[matched_pairs[rand_num, 1]])

        try:
            cur_homography = compute_homography(np.array(left_coords), np.array(right_coords))
        except np.linalg.LinAlgError:
            continue

        row_1 = coords_1_ordered[matched_pairs[:, 0]][:, 0]
        row_2 = coords_1_ordered[matched_pairs[:, 0]][:, 1]

        all_points_to_map = np.stack([row_1, row_2, np.ones(N)])

        mapped = (cur_homography @ all_points_to_map).T
        mapped = mapped[:, :2] / mapped[:, 2:3]

        labels = coords_2_ordered[matched_pairs[:, 1]]

        dist_vals = np.sqrt(np.sum((mapped - labels)**2, axis=1))

        cur_inliers = dist_vals[dist_vals < ERR_THRESH]
        if len(cur_inliers) > max_inliers:
            max_inliers_mask = dist_vals < ERR_THRESH
            max_inliers = len(cur_inliers)

    print('RANSAC done, max_inliers found = ', max_inliers)

    final_left_pts = coords_1_ordered[matched_pairs[max_inliers_mask, 0]]
    final_right_pts = coords_2_ordered[matched_pairs[max_inliers_mask, 1]]

    final_homography = compute_homography(final_left_pts, final_right_pts)

    return final_homography, max_inliers_mask


def warp_n_imgs(img_paths_list, stitching='automatic', save_dir='part_1_results/', method='nearest_neighbor',
                img_name='warped_img', gain_correct=False, NUM_RANSAC_ITERS=5000):
    assert stitching == 'automatic' or stitching == 'manual', "Stitching must be either automatic or manual"

    cur_mosaic = cv2.imread(img_paths_list[0])
    for idx in range(len(img_paths_list) - 1):
        next_img = cv2.imread(img_paths_list[idx + 1])
        if stitching == 'manual':
            img_idx_pts = label_image(cur_mosaic, f'{img_name}_{idx}_mosaic', save_dir=save_dir)
            img_idx_plus_1_pts = label_image(next_img, f'{img_name}_{idx}_next', save_dir=save_dir)

            homography = compute_homography(img_idx_pts, img_idx_plus_1_pts)

        elif stitching == 'automatic':
            cur_mosaic_bw = cv2.cvtColor(cur_mosaic, cv2.COLOR_BGR2GRAY)
            next_img_bw = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

            h_1, coords_1 = get_harris_corners(cur_mosaic_bw)
            h_2, coords_2 = get_harris_corners(next_img_bw)

            coords_1 = filter_corners(h_1, coords_1, cur_mosaic)
            coords_2 = filter_corners(h_2, coords_2, next_img)

            ANMS_coords_1 = ANMS(h_1, coords_1)
            ANMS_coords_2 = ANMS(h_2, coords_2)

            vecs_left, patches_left = feature_descript_extract(cur_mosaic_bw, ANMS_coords_1)
            vecs_right, patches_right = feature_descript_extract(next_img_bw, ANMS_coords_2)

            matched_pairs = match_features(vecs_left, vecs_right)

            coords_1_ordered = ANMS_coords_1[::-1].T
            coords_2_ordered = ANMS_coords_2[::-1].T

            homography, max_inliers_mask = RANSAC4(matched_pairs, coords_1_ordered, coords_2_ordered,
                                                   NUM_ITERATIONS=NUM_RANSAC_ITERS)

        homography = np.linalg.inv(homography)

        next_img_mapped, cur_mosaic_mapped, cur_mosaic = warp_2_imgs(next_img, cur_mosaic, homography,
                                                                     method=method, img_name=f'{img_name}_{idx}',
                                                                     create_mosaic=True, gain_correct=gain_correct, save_dir=save_dir)

    cv2.imwrite(f'{save_dir}{img_name}_{method}.jpg', cur_mosaic)

    return cur_mosaic