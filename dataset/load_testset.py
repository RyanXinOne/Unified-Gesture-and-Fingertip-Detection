import cv2
import numpy as np


def visualize(image, prob, key, label):
    index = 0

    color = [(120, 20, 240), (240, 55, 210), (240, 55, 140), (240, 75, 55), (170, 240, 55)]
    for c, p in enumerate(prob):
        if p > 0.5:
            image = cv2.circle(image, (int(key[index]), int(key[index + 1])),
                               radius=10, color=color[c], thickness=-2)
        index = index + 2

    cv2.imshow(label, image)
    return


""" load full images """
x = np.load('test/test_x.npy')

""" load probabilistic labels """
y_prob = np.load('test/test_y_prob.npy')

""" load keypoints coordinates """
y_keys = np.load('test/test_y_keys.npy')

""" load cropped images """
x_cropped = np.load('test/cropped_image.npy')

""" load crop info """
crop_info = np.load('test/crop_info.npy')

print('images:', x.shape)
print('cropped images:', x_cropped.shape)
print('probabilistic labels:', y_prob.shape)
print('keypoints coordinates:', y_keys.shape)

""" 
total testing samples: 2930
"""
sample_number = 2000

x_sample = x[sample_number]
y_prob_sample = y_prob[sample_number]
y_keys_sample = y_keys[sample_number]

x_crop_sample = x_cropped[sample_number]
crop_info_sample = crop_info[sample_number]

""" visualize original image """
visualize(image=x_sample,
          prob=y_prob_sample,
          key=y_keys_sample,
          label='Original Image')

# tl: top-left
tl_x, tl_y, height, width = crop_info_sample
pos = y_keys_sample

for i in range(0, len(pos), 2):
    pos[i] = (pos[i] - tl_x) / width * 128
    pos[i + 1] = (pos[i + 1] - tl_y) / height * 128

""" visualize cropped image """
visualize(image=x_crop_sample,
          prob=y_prob_sample,
          key=pos,
          label='Cropped Image')

print("Press 'Esc' to CLOSE")
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
