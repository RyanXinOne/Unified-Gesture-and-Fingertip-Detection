import cv2
import numpy as np


def visualize(image, prob, key):
    index = 0

    color = [(120, 20, 240), (240, 55, 210), (240, 55, 140), (240, 75, 55), (170, 240, 55)]
    for c, p in enumerate(prob):
        if p > 0.5:
            image = cv2.circle(image, (int(key[index]), int(key[index + 1])),
                               radius=5, color=color[c], thickness=-2)
        index = index + 2

    cv2.imshow("Press 'Esc' to CLOSE", image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return


# train or valid
dataset = 'train'

""" load images """
x = np.load(dataset + '/' + dataset + '_x.npy')

""" load probabilistic labels """
y_prob = np.load(dataset + '/' + dataset + '_y_prob.npy')

""" load keypoints coordinates """
y_keys = np.load(dataset + '/' + dataset + '_y_keys.npy')

print('images:', x.shape)
print('probabilistic labels:', y_prob.shape)
print('keypoints coordinates:', y_keys.shape)

""" 
total training samples: 25090
total validation samples: 1317
"""
sample_number = 100

x_sample = x[sample_number]
y_prob_sample = y_prob[sample_number]
y_keys_sample = y_keys[sample_number]

visualize(image=x_sample,
          prob=y_prob_sample,
          key=y_keys_sample)
