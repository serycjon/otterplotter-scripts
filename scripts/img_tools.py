import numpy as np
import cv2
from PIL import Image, ImageCms
from collections import deque
from various_utils import with_debugger
import tqdm
import logging
logger = logging.getLogger(__name__)
format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"


def in_bounds(pos_xy, canvas):
    return pos_xy[0] >= 0 and pos_xy[1] >= 0 and pos_xy[0] < canvas.shape[1] and pos_xy[1] < canvas.shape[0]


def clip_to_img(pos_xy, canvas):
    return np.clip(pos_xy,
                   (0, 0),
                   (canvas.shape[1] - 1, canvas.shape[0] - 1))


def resize_to_px_count(img, count):
    H, W = img.shape[:2]
    orig_count = H * W

    scale = np.sqrt(count / orig_count)
    return cv2.resize(img, None, fx=scale, fy=scale)


def resize_to_max(img, max_sz):
    H_scale = max_sz / img.shape[0]
    W_scale = max_sz / img.shape[1]

    scale = min(H_scale, W_scale)
    return cv2.resize(img, None, fx=scale, fy=scale)


def bgr2cmyk(img):
    rgb = img[:, :, ::-1]
    p_rgb = Image.fromarray(rgb)

    srgb_path = '/usr/share/color/icc/colord/sRGB.icc'
    cmyk_path = '/usr/share/color/icc/krita/cmyk.icm'
    ImageCms.getOpenProfile(srgb_path)
    ImageCms.getOpenProfile(cmyk_path)
    p_cmyk = ImageCms.profileToProfile(p_rgb,
                                       srgb_path,
                                       cmyk_path,
                                       # 'USWebCoatedSWOP.icc',
                                       renderingIntent=0, outputMode='CMYK')
    cmyk = np.array(p_cmyk)
    return cmyk


def sample_points_naive(img, N=100):
    H, W = img.shape[:2]
    sz = H * W
    probs = img.flatten().astype(np.float64) / 255
    probs /= np.sum(probs)
    idx = np.random.choice(sz, N, replace=False, p=probs)
    pts = np.stack(np.unravel_index(idx, (H, W))[::-1]).T
    return pts


def sample_points_sequential(img):
    """ Sample points in image

    args:
        img: float 2-dimensional image (0-1)

    returns:
        points: (N, 2) np.array of x,y coordinates """
    assert len(img.shape) == 2, "Expecting grayscale (H, W) image"
    H, W = img.shape
    points = []
    for r in range(H):
        for c in range(W):
            intensity = img[r, c]
            if np.random.rand() <= intensity:
                points.append((c, r))

    points = np.array(points)
    assert points.shape[1] == 2
    return points


def floyd_steinberg(image):
    # image: np.array of shape (height, width), dtype=float, 0.0-1.0
    # works in-place!
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            # precomputing the constants helps
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375  # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625  # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125  # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h):
                image[y + 1, x - 1] += error * 0.1875  # left, down, 3 / 16
    return image


@with_debugger
def ascencio2010stippling(img, K, vis_every=None, min_r=None, max_r=None):
    """ Stipple image

    args:
        img: (H, W) float density image (range 0-1)
        K: density per stipple
    """
    class Range:
        def __init__(self, lb, ub):
            self.intervals = [(lb, ub)]

        def length(self):
            res = 0
            for lb, ub in self.intervals:
                res += ub - lb

            return res

        def sample(self):
            subint_probs = np.array([ub - lb for lb, ub in self.intervals])
            subint_probs /= np.sum(subint_probs)
            subint = np.random.choice(len(self.intervals), p=subint_probs)
            subint = self.intervals[subint]

            value = subint[0] + np.random.rand() * (subint[1] - subint[0])
            return value

        def subtract(self, lb, ub):
            new_intervals = []
            for subint_lb, subint_ub in self.intervals:
                left = subint_lb, lb
                right = ub, subint_ub

                if left[1] > left[0]:
                    new_intervals.append(left)

                if right[1] > right[0]:
                    new_intervals.append(right)
            self.intervals = new_intervals

    class Disk:
        def __init__(self, x, y, r):
            self.x = x
            self.y = y
            self.r = r
            self.boundary = Range(0, 2 * np.pi)

        def tune_radius(self, parent, angle, density_img, K):
            self.density = local_density(density_img, self.x, self.y, self.r)
            ratio_R = np.sqrt(K / (self.density + 1e-10))
            epsilon = 0.001
            i = 0
            while i < 6 and np.abs(ratio_R - 1) > epsilon:
                self.r *= ratio_R
                self.r = np.clip(self.r, min_r, max_r)  # not in the paper
                self.x = parent.x + (parent.r + self.r) * np.cos(angle)
                self.y = parent.y + (parent.r + self.r) * np.sin(angle)
                self.density = local_density(density_img, self.x, self.y, self.r)
                ratio_R = np.sqrt(K / (self.density + 1e-10))
                i += 1

            if min_r is not None and max_r is not None:
                if self.r <= min_r and in_bounds((self.x, self.y), density_img):
                    logger.warn('min_clip')
                if self.r >= max_r and in_bounds((self.x, self.y), density_img):
                    logger.warn('max_clip')
                self.r = np.clip(self.r, min_r, max_r)  # not in the paper
                self.x = parent.x + (parent.r + self.r) * np.cos(angle)
                self.y = parent.y + (parent.r + self.r) * np.sin(angle)

        def can_place(self, buffer):
            if not in_bounds((self.x, self.y), buffer):
                return False
            overlap = local_density(buffer, self.x, self.y, self.r)
            return overlap < (np.pi * self.r**2) * 0.05

        def draw(self, buffer):
            cv2.circle(buffer,
                       (int(np.round(self.x)), int(np.round(self.y))),
                       int(np.round(self.r)),
                       color=1,
                       thickness=-1)

        def available_range(self):
            return self.boundary.length()

        def sample_available_angle(self):
            return self.boundary.sample()

        def subtract_range(self, other, alpha):
            c = (self.r + other.r) / 2
            cos_beta = c / (2 * max(self.r, other.r))
            beta = np.arccos(cos_beta)
            lb = np.mod(alpha - beta, 2 * np.pi)
            ub = np.mod(alpha + beta, 2 * np.pi)

            if ub < lb:  # interval includes 0
                self.boundary.subtract(lb, 2 * np.pi)
                self.boundary.subtract(0, ub)
            else:
                self.boundary.subtract(lb, ub)

    def local_density(density_img, x, y, r):
        # extract bounding box ROI
        c_x, c_y = int(np.round(x)), int(np.round(y))
        R = int(np.round(r))
        tl_x, tl_y = c_x - R, c_y - R
        # br_x, br_y = int(c_x + R), int(c_y + R)

        A = np.array([[1.0, 0, -tl_x],
                      [0, 1.0, -tl_y]])

        roi = cv2.warpAffine(density_img, A, (2 * R + 1,
                                              2 * R + 1),
                             flags=cv2.INTER_NEAREST,
                             borderValue=0)
        mask = circle_mask(roi.shape)
        density = np.sum(roi[mask])
        return density

    circle_masks = {}  # cache

    def circle_mask(shape):
        if shape in circle_masks:
            return circle_masks[shape]
        assert shape[0] % 2 == 1
        assert shape[1] % 2 == 1
        c_x, c_y = (shape[1] - 1) // 2, (shape[0] - 1) // 2
        r = min(c_x, c_y)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (c_x, c_y), r, 1, thickness=-1)
        mask = mask > 0
        circle_masks[shape] = mask
        return mask

    disk_buffer = np.zeros_like(img)
    pbar_every = 100
    pbar = tqdm.tqdm(desc="stippling", total=np.prod(disk_buffer.shape))
    last_covered = 0

    H, W = img.shape
    total_density = np.sum(img)
    r1 = np.sqrt(H * W / (total_density / K))

    output_disk_list = []
    disk_queue = deque()

    # init with a random disk
    init_x, init_y = np.random.randint(0, W), np.random.randint(0, H)
    D = Disk(init_x, init_y, r1)
    null_disk = Disk(init_x, init_y, 0)
    D.tune_radius(null_disk, 0, img, K)

    disk_queue.append(D)

    if vis_every is not None:
        cv2.namedWindow('cv: disk_buffer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cv: disk_buffer', 800, 600)

    while len(disk_queue) > 0:
        Q = disk_queue.popleft()

        while Q.available_range() > 0:
            alpha = Q.sample_available_angle()
            new_x = Q.x + (Q.r + r1) * np.cos(alpha)
            new_y = Q.y + (Q.r + r1) * np.sin(alpha)
            D = Disk(new_x, new_y, r1)
            D.tune_radius(Q, alpha, img, K)

            if D.can_place(disk_buffer):
                disk_queue.append(D)
                output_disk_list.append(D)
                D.draw(disk_buffer)

            Q.subtract_range(D, alpha)

            if vis_every is not None and len(output_disk_list) % vis_every == 0:
                cv2.imshow("cv: disk_buffer", 1 - disk_buffer)
                c = cv2.waitKey(1)
                if c == ord('q'):
                    break

            if len(output_disk_list) % pbar_every == 0:
                current_covered = np.sum(disk_buffer)
                if current_covered > last_covered:
                    pbar.update(current_covered - last_covered)
                    last_covered = current_covered

    if vis_every is not None:
        cv2.destroyAllWindows()

    pbar.close()

    stipples = [(D.x, D.y) for D in output_disk_list]
    stipples = np.array(stipples)
    return stipples


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    img = cv2.imread("/home/jonas/Pictures/profile_big.jpg", 0)

    density = 1 - (img.astype(np.float32) / 255)
    density = resize_to_max(density, 4000)
    stippling = ascencio2010stippling(density, K=400, min_r=10, max_r=45, vis_every=100)
    from opengl_utils import Drawer
    gui = Drawer(point_r=5)
    # gui.add_lines([points.copy()], 'g')
    gui.add_points([stippling], 'k')
    gui.draw()
