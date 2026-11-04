def warp_image_forward(img_1, img_2, img_1_points_path, img_2_points_path, method='nearest_neighbor'):
    y_px_1, x_px_1 = np.mgrid[0:img_1.shape[0], 0:img_1.shape[1]]
    orig_1 = np.stack([x_px_1.ravel(), y_px_1.ravel()]).T
    all_points_1 = np.stack([x_px_1.ravel(), y_px_1.ravel(), np.ones(x_px_1.size)])

    y_px_2, x_px_2 = np.mgrid[0:img_2.shape[0], 0:img_2.shape[1]]
    orig_2 = np.stack([x_px_2.ravel(), y_px_2.ravel()]).T

    homography = compute_homography(img_1_points_path, img_2_points_path)
    mapped = (homography @ all_points_1).T
    mapped = mapped[:, :2] / mapped[:, 2:3]

    if method == 'nearest_neighbor':
        mapped = np.round(mapped).astype(int)
    elif method == 'bilinear':
        return -1
    else:
        raise ValueError('method must be nearest_neighbor or bilinear')

    (NEW_HEIGHT, NEW_WIDTH), (ADD_HEIGHT_UP, ADD_WIDTH_LEFT) = get_new_img_size(img_1, img_2, homography)

    new_img = np.zeros((NEW_HEIGHT, NEW_WIDTH, 3), dtype=np.uint8)
    for mapped_px, orig_px_1, orig_px_2 in zip(mapped, orig_1, orig_2):
        new_img[mapped_px[1] + ADD_HEIGHT_UP][mapped_px[0] + ADD_WIDTH_LEFT] = img_1[orig_px_1[1]][orig_px_1[0]]

    cv2.imwrite('part_1_results/warped_image.jpg', new_img)

    return new_img