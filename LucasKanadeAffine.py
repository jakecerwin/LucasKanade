import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift, affine_transform
import cv2

def LeastSquares(A, f):
    u = np.linalg.inv(A.T @ A) @ A.T @ f
    return u


def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 50
    # p1 = np.zeros((6,1))
    x1, y1, x2, y2 = rect

    # put your implementation here
    p = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    y, x = np.arange(It.shape[0]), np.arange(It.shape[1])
    interp_spline1 = RectBivariateSpline(y, x, It1)

    iterNum = 0
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    X, Y = np.meshgrid(x, y)

    while True:

        # calculate affine projected rect
        # transform points
        Xt = p[0] * X + p[1] * Y + p[2]
        Yt = p[3] * X + p[4] * Y + p[5]
        hbounds = (x2 > Xt) & (Xt >= x1)
        vbounds = (Yt >= y1) & (Yt < y2)
        projected_rect = hbounds & vbounds
        Xt = Xt[projected_rect]
        Yt = Yt[projected_rect]

        # gradient for jacobian Yt replace Y in original lk
        interp_It1 = interp_spline1.ev(Yt, Xt)
        interped_gx = interp_spline1.ev(Yt, Xt, dx=0, dy=1).flatten()
        interped_gy = interp_spline1.ev(Yt, Xt, dx=1, dy=0).flatten()

        # get jacobian [dI(x)/du , dI(x)/dv],  di(x) = [gx * x, gx * y, gx]
        jacobian = np.zeros((len(interped_gx), 6))              # eq 8
        jacobian[:, 0] = interped_gx * X[projected_rect]        # eq 8
        jacobian[:, 1] = interped_gx * Y[projected_rect]        # eq 8
        jacobian[:, 2] = interped_gx                            # eq 8
        jacobian[:, 3] = interped_gy * X[projected_rect]        # eq 8
        jacobian[:, 4] = interped_gy * Y[projected_rect]        # eq 8
        jacobian[:, 5] = interped_gy                            # eq 8

        # T(x) - I(W(x;p))
        b = It[projected_rect].flatten() - interp_It1.flatten()     # eq 9

        # Heterogenous solution dp = H^{-1} @ J^T @ b
        dp = LeastSquares(jacobian, b)

        # p <- p + dp
        p += dp

        p += dp.flatten()

        iterNum += 1
        if iterNum >= maxIters or np.sum(np.square(dp)) < threshold:  # eq 4
            break

    # reshape the output affine matrix
    M = np.array([[p[0], p[1], p[2]],
                  [p[3], p[4], p[5]]]).reshape(2, 3)

    return M
