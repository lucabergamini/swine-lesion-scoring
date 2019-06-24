import numpy as np
import cv2


def get_lesion_score_from_segmentation(seg: np.ndarray, th_lesion: float = 0.05, th_chest: float = 0.025,
                                       th_small: float = 50):
    """
    :param seg: binary seg BxNCxHxW
    :param th_lesion: threshold between intersection and lesion area [0,1]
    :param th_chest: threshold between intersection and chestwall area [0,1]
    :param th_small: threshold over lesion area in pixels
    :return:
    """
    scores = []
    for seg_el in seg:
        on_chestwall_1 = 0
        on_chestwall_3 = 0
        # get individual planes
        lesions = seg_el[-1, ...]
        chestwall_1 = seg_el[4, ...]
        chestwall_3 = seg_el[5, ...]
        # -------------------- cv2 Connected Component ------------------------------------------
        ret, labels, stats, _ = cv2.connectedComponentsWithStats(lesions.astype(np.uint8) * 255)
        if ret < 2:  # no lesions
            scores.append(0)
            continue
        # ---------------------- Lesion analysis using IoA-------------------------------------------------
        for i in range(1, ret):
            if stats[i][-1] < th_small:
                continue
            chestwall_1_area = np.sum(chestwall_1)
            chestwall_3_area = np.sum(chestwall_3)
            lesion_area = stats[i, -1]
            intersection_1 = np.sum(((labels == i) * chestwall_1))
            intersection_2 = np.sum(((labels == i) * chestwall_3))
            ioa_1_l = intersection_1 / (lesion_area + 1e-8)  # ratio of lesion covered
            ioa_1_c = intersection_1 / (chestwall_1_area + 1e-8)  # ratio of chestwall_1 covered
            ioa_2_l = intersection_2 / (lesion_area + 1e-8)
            ioa_2_c = intersection_2 / (chestwall_3_area + 1e-8)
            if ioa_1_l > th_lesion or ioa_1_c > th_chest:
                on_chestwall_1 = 1
            if ioa_2_l > th_lesion or ioa_2_c > th_chest:
                on_chestwall_3 = 2  # this way we get the score just by summing up
        scores.append(on_chestwall_1 + on_chestwall_3)
    return np.asarray(scores)
