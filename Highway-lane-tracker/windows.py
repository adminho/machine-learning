from typing import List, Tuple

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf
from scipy.ndimage.filters import gaussian_filter

# Fixbug
def dot3(a, b, c):
    return np.dot(a, b).dot(c)

class Window:
    def __init__(self, level, window_shape, img_shape, x_init, max_frozen_dur):
        """
        Tracks a window as used for selecting lane lines in an image.

        :param level: Level of the window, as counted from the bottom of the image up.
        :param window_shape: (height, width) of the window in pixels.
        :param img_shape: (height, width) of the image the window resides in.
        :param x_init: Initial x position of the window.
        :param max_frozen_dur: The maximum amount of frames a window can continue to be used when frozen (eg when not
        found or when measurements are uncertain).
        """
        if window_shape[1] % 2 == 0:
            raise Exception("width must be odd")
        # Image info
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]

        # Window shape
        self.height = window_shape[0]
        self.width = window_shape[1]
        self.y_begin = self.img_h - (level + 1) * self.height  # top row of pixels for window
        self.y_end = self.y_begin + self.height  # one past the bottom row of pixels for window

        # Window position
        self.x_filtered = x_init
        self.y = self.y_begin + self.height / 2.0
        self.level = level

        # Detection info
        self.filter = WindowFilter(pos_init=x_init)
        self.x_measured = None
        self.frozen = False
        self.detected = False
        self.max_frozen_dur = max_frozen_dur
        self.frozen_dur = max_frozen_dur + 1
        self.undrop_buffer = 1  # Number of calls to unfreeze() needed to go from dropped back to normal.

    def x_begin(self, param='x_filtered'):
        """
        The leftmost position of the window, relative to the last filtered position or measurement.

        :param param: Whether to use the 'x_filtered' or 'x_measured' position.
        """
        self.check_x_param(param)
        x = getattr(self, param)
        return int(max(0, x - self.width // 2))

    def x_end(self, param='x_filtered'):
        """
        One past the rightmost position of the window, relative to the last filtered position or measurement.

        :param param: Whether to use the 'x_filtered' or 'x_measured' position.
        """
        self.check_x_param(param)
        x = getattr(self, param)
        return int(min(x + self.width // 2, self.img_w))

    def area(self):
        """Area of the window."""
        return self.height * self.width

    def freeze(self):
        """Marks the window as frozen, drops it if it's been frozen for too long, and increases filter uncertainty."""
        self.frozen = True
        self.frozen_dur += 1
        self.filter.grow_uncertainty(1)

    def unfreeze(self):
        """Marks the window as not frozen and not dropped, reduces frozen counter by 1."""
        # Reduce frozen duration to max (plus some buffer)
        self.frozen_dur = min(self.frozen_dur, self.max_frozen_dur + 1 + self.undrop_buffer)
        self.frozen_dur -= 1
        self.frozen_dur = max(0, self.frozen_dur)

        # Change states
        self.frozen = False

    @property
    def dropped(self):
        return self.frozen_dur > self.max_frozen_dur

    def update(self, score_img, x_search_range, min_log_likelihood=-40):
        """
        Given a score image and the x search bounds, updates the window position to the likely position of the lane.

        If the measurement is deemed suspect for some reason, the update will be rejected and the window will be
        'frozen', causing it to stay in place. If the window is frozen beyond its  `max_frozen_dur` then it will be
        dropped entirely until a non-suspect measurement is made.

        The window only searches within its y range defined at initialization.

        :param score_img: A score image, where pixel intensity represents where the lane most likely is.
        :param x_search_range: The (x_begin, x_end) range the window should search between in the score image.
        :param min_log_likelihood: The minimum log likelihood allowed for a measurement before it is rejected.
        """
        assert score_img.shape[0] == self.img_h and \
               score_img.shape[1] == self.img_w, 'Window not parametrized for this score_img size'

        # Apply a column-wise gaussian filter to score the x-positions in this window's search region
        x_search_range = (max(0, int(x_search_range[0])), min(int(x_search_range[1]), self.img_w))
        x_offset = x_search_range[0]
        search_region = score_img[self.y_begin: self.y_end, x_offset: x_search_range[1]]
        column_scores = gaussian_filter(np.sum(search_region, axis=0), sigma=self.width / 3, truncate=3.0)

        if max(column_scores) != 0:
            self.detected = True
            # Update measurement
            self.x_measured = np.argmax(column_scores) + x_offset
            window_magnitude = \
                np.sum(column_scores[self.x_begin('x_measured') - x_offset: self.x_end('x_measured') - x_offset])
            noise_magnitude = np.sum(column_scores) - window_magnitude
            signal_noise_ratio = \
                window_magnitude / (window_magnitude + noise_magnitude) if window_magnitude is not 0 else 0

            # Filter measurement and set position
            if signal_noise_ratio < 0.6 or self.filter.loglikelihood(self.x_measured) < min_log_likelihood:
                # Suspect / bad measurement, don't update filter/position
                self.freeze()
                return
            self.unfreeze()
            self.filter.update(self.x_measured)
            self.x_filtered = self.filter.get_position()

        else:
            # No signal in search region
            self.detected = False
            self.freeze()

    def get_mask(self, param='x_filtered'):
        """
        Returns a masking image of shape (self.img_h, self.img_w) with the pixels occupied by this window set to 1.

        :param param: Whether to use the 'x_filtered' or 'x_measured' position of the window.
        :return: An image with the pixels occupied by the window set to 1 and all other pixels set to 0.
        """
        self.check_x_param(param)
        mask = np.zeros((self.img_h, self.img_w))
        mask[self.y_begin: self.y_end, self.x_begin(param): self.x_end(param)] = 1
        return mask

    def pos_xy(self, param: str = 'x_filtered') -> Tuple[float, float]:
        """Returns the (x, y) position of this window."""
        self.check_x_param(param)
        return getattr(self, param), self.y

    def check_x_param(self, param):
        assert param == 'x_filtered' or param == 'x_measured', "Invalid position parameter. `param` must be " \
                                                               "'x_filtered' or 'x_measured' "


def sliding_window_update(windows: List[Window], score_img, margin, mode):
    """
    Updates each window in a list, constraining their search regions to a marginal distance of the last valid window.
    Generally improved upon in `joint_sliding_window_update()`, which is typically recommended instead.

    Each window's search region will be centered on the last undropped window position and extend a margin to the
    left and right.

    :param windows: A list of Window objects.
    :param score_img: A score image, where pixel intensity represents where the lane most likely is.
    :param margin: The maximum x distance the next window can be placed from the last undropped window.
    :param mode: 'left' or 'right' for the lane the Windows are tracking.
    """
    search_center = start_sliding_search(windows, score_img, mode)

    # Update each window, searching nearby the last undropped window.
    for window in windows[1:]:
        x_search_range = (search_center - margin, search_center + margin)
        window.update(score_img, x_search_range)

        if not window.dropped:
            search_center = window.x_filtered


def joint_sliding_window_update(windows_left: List[Window], windows_right: List[Window], score_img, margin):
    """
    Updates Windows from both lists, preventing window crossover and constraining their search regions to a margin.

    This improves on `sliding_window_update()` by preventing windows from different lanes from crossing over each other
    or detecting the same part of the image.

    Each window's search region will be centered on the last undropped window position and extend a margin to the
    left and right. In cases where the margins of the left and right lane may overlap, they are truncated to the
    halfway point between.

    :param windows_left: A list of Window objects for the left lane.
    :param windows_right: A list of Window objects for the right lane.
    :param score_img: A score image, where pixel intensity represents where the lane most likely is.
    :param margin: The maximum x distance the next window can be placed from the last undropped window.
    """
    assert len(windows_left) == len(windows_right), "Window lists should be same length. Did you filter already?"

    search_centers = [start_sliding_search(windows_left, score_img, 'left'),
                      start_sliding_search(windows_right, score_img, 'right')]

    # Update each window, searching nearby the last undropped window.
    for i in range(len(windows_left)):
        # Find search range for the left and right
        x_search_ranges = [None, None]
        for j in [0, 1]:
            x_search_ranges[j] = [search_centers[j] - margin, search_centers[j] + margin]

        # Fix any crossover
        if x_search_ranges[0][1] > x_search_ranges[1][0]:
            average = (x_search_ranges[0][1] + x_search_ranges[1][0]) // 2
            x_search_ranges[0][1] = average
            x_search_ranges[1][0] = average

        # Perform update
        for j, window in enumerate([windows_left[i], windows_right[i]]):
            window.update(score_img, x_search_ranges[j])
            if not window.dropped:
                search_centers[j] = window.x_filtered


def start_sliding_search(windows, score_img, mode):
    assert mode == 'left' or mode == 'right', "Mode not valid."
    assert strictly_decreasing([w.y for w in windows]), "Windows not ordered properly. Should start at image bottom"
    img_h, img_w = score_img.shape[0:2]
    # Update the bottom window
    if mode == 'left':
        windows[0].update(score_img, (0, img_w // 2))
    elif mode == 'right':
        windows[0].update(score_img, (img_w // 2, img_w))

    # Find the starting point for our search
    if windows[0].dropped:
        # Starting window does not exist, find an approximation.
        search_region = score_img[2 * img_h // 3:, :]  # search bottom 1/3rd of score_img
        column_scores = gaussian_filter(np.sum(search_region, axis=0), sigma=windows[0].width / 3, truncate=3.0)
        if mode == 'left':
            search_center = argmax_between(column_scores, 0, img_w // 2)
        elif mode == 'right':
            search_center = argmax_between(column_scores, img_w // 2, img_w)
        assert 'search_center' in locals(), 'No lane was found to start with.'
        # TODO: Do something if still no lane is found
    else:
        # Reuse the position of the bottom window
        search_center = windows[0].x_filtered

    return search_center


def strictly_decreasing(L):
    """Returns True if elements of L are strictly decreasing."""
    return all(x > y for x, y in zip(L, L[1:]))


def argmax_between(arr: np.ndarray, begin: int, end: int) -> int:
    """
    Returns the position of the maximal value between indexes `begin` and `end`.

    In case of multiple occurrences of the maximum value, the index of the first occurrence is returned.
    """
    max_ndx = np.argmax(arr[begin:end]) + begin
    return max_ndx


def filter_window_list(windows: List[Window], remove_frozen=False, remove_dropped=True, remove_undetected=False):
    """
    Given a list of Windows, returns a new list with frozen and dropped windows optionally removed.

    :param windows: A list of Window objects.
    :param remove_frozen: Set True to prevent returning all frozen windows.
    :param remove_dropped: Set True to prevent returning all dropped windows.
    :param remove_undetected: Set True to prevent returning all undetected windows.
    :return: (windows_filtered, args)
             windows_filtered: The new list of Windows after filters are applied.
             args: The index in `windows` that each window in `windows_filtered` originated from.
    """
    windows_filtered = []
    args = []
    for i, window in enumerate(windows):
        if window.dropped and remove_dropped:
            continue
        if window.frozen and remove_frozen:
            continue
        if not window.detected and remove_undetected:
            continue
        windows_filtered.append(window)
        args.append(i)
    return windows_filtered, args


def window_image(windows: List[Window], param='x_filtered', color=(0, 255, 0), color_frozen=None, color_dropped=None):
    """
    Creates an image with the given `windows` colored in. By default dims frozen windows and hides dropped windows.

    :param windows: A List of Windows to image.
    :param param: Whether to use the 'x_filtered' or 'x_measured' position of the window.
    :param color: Color for each window that is not frozen or dropped.
    :param color_frozen: Color for each frozen window.
    :param color_dropped: Color for each dropped window.
    :return: An image with the windows colored in and all black elsewhere.
    """
    if color_frozen is None:
        color_frozen = [ch * 0.6 for ch in color]
    if color_dropped is None:
        color_dropped = [0, 0, 0]
    mask = np.zeros((windows[0].img_h, windows[0].img_w, 3))
    for window in windows:
        if getattr(window, param) is None:
            continue
        if window.dropped:
            color_curr = color_dropped
        elif window.frozen:
            color_curr = color_frozen
        else:
            color_curr = color
        mask[window.get_mask(param) > 0] = color_curr
    return mask.astype('uint8')


class WindowFilter:
    def __init__(self, pos_init=0.0, meas_variance=50, process_variance=1, uncertainty_init=2 ** 30):
        """
        A one dimensional Kalman filter tuned to track the position of a window.

        State variable:   = [position,
                             velocity]

        :param pos_init: Initial position.
        :param meas_variance: Variance of each measurement. Decrease to have the filter chase each measurement faster.
        :param process_variance: Variance of each prediction. Decrease to follow predictions more.
        :param uncertainty_init: Uncertainty of initial position.
        """
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition function
        self.kf.F = np.array([[1., 1],
                              [0., 0.5]])

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

        # Measurement noise
        self.kf.R = np.array([[meas_variance]])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_variance)

    def update(self, pos):
        """
        Given an estimate x position, uses the kalman filter to estimate the most likely true position of the
        lane pixel.

        :param pos: measured x position of the pixel
        """
        self.kf.predict()
        self.kf.update(pos)

    def grow_uncertainty(self, mag):
        """Grows state uncertainty."""
        for i in range(mag):
            # P = FPF' + Q
            self.kf.P = self.kf._alpha_sq * dot3(self.kf.F, self.kf.P, self.kf.F.T) + self.kf.Q

    def loglikelihood(self, pos):
        """Calculates the likelihood of a measurement given the filter parameters and gaussian assumption."""
        self.kf.S = dot3(self.kf.H, self.kf.P, self.kf.H.T) + self.kf.R
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)

    def get_position(self):
        return self.kf.x[0]
