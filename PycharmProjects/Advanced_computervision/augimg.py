import imgaug as ia
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
% matplotlib inline

ia.seed(1)

image = imageio.imread("/content/20220504_123457.jpg")
h = float(k[4]) * 3096
w = float(k[3]) * 4128
a = float(k[1]) * 2 * 4128
b = float(k[2]) * 2 * 3096
X1 = (a - w) / 2
Y1 = (b - h) / 2
X2 = (a + w) / 2
Y2 = (b + h) / 2
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=X1, y1=Y1, x2=X2, y2=Y2),

], shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
    iaa.Affine(rotate=45,
               translate_px={"x": 40, "y": 60},
               scale=(0.5, 0.7)
               )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

# Augment BBs and images.
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
          )

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

% matplotlib
inline
# The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
from matplotlib import pyplot as plt

image_after = cv2.rectangle(image_after, (1438, 807), (3073, 2443), (255, 0, 0), 10)

# Import image
# image = cv2.imread("/content/20220504_123457.jpg")

# Show the image with matplotlib
plt.imshow(image_after)
plt.show()
cv2.imwrite("/content/a.jpg", image_after)