import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    #p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    interp_spline_It1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)
    interp_spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    x = np.arange(0, It.shape[1] - 0.5)
    y = np.arange(0, It.shape[0] - 0.5)
    X, Y = np.meshgrid(x, y)

    interp_gradientX = interp_spline_It.ev(Y, X, dx=0, dy=1).flatten()
    interp_gradientY = interp_spline_It.ev(Y, X, dx=1, dy=0).flatten()

    jacobian = np.zeros((len(interp_gradientX), 6))
    jacobian[:, 0] = interp_gradientX * X.flatten()
    jacobian[:, 1] = interp_gradientX * Y.flatten()
    jacobian[:, 2] = interp_gradientX
    jacobian[:, 3] = interp_gradientY * X.flatten()
    jacobian[:, 4] = interp_gradientY * Y.flatten()
    jacobian[:, 5] = interp_gradientY
    iterCount = 0
    while True:
        iterCount += 1

        Xt = p[0] * X + p[1] * Y + p[2]
        Yt = p[3] * X + p[4] * Y + p[5]
        hbounds = (x2 > Xt) & (Xt >= x1)
        vbounds = (Yt >= y1) & (Yt < y2)
        pr = hbounds & vbounds
        Xt = Xt[pr]
        Yt = Yt[pr]

        interped_I = interp_spline_It1.ev(Yt, Xt)

        jacobian_pr = jacobian[pr.flatten()]
        b = interped_I.flatten() - It[pr].flatten()
        bt = np.dot(jacobian_pr.T, b)
        try:
            dp = np.dot(np.linalg.inv(jacobian_pr.T @ jacobian_pr), bt)
        except:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        M = np.vstack((np.reshape(p, (2, 3)), np.array([[0, 0, 1]])))
        dM = np.vstack((np.reshape(dp, (2, 3)), np.array([[0, 0, 1]])))
        dM[0, 0] = dM[0, 0] + 1
        dM[1, 1] = dM[1, 1] + 1
        M = np.dot(M, np.linalg.inv(dM))

        p = M[:2, :].flatten()
        if np.sum(dp ** 2) < threshold or iterCount > maxIters:
            break

    # deals with brightness change
    if p[2] > 1000: return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # reshape the output affine matrix
    M = np.array([[p[0], p[1],   p[2]],
                 [p[3],    p[4], p[5]]]).reshape(2, 3)

    return M
