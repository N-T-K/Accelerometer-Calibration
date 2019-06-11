# description: computes the scale factor matrix and the bias vector of a MEMS accelerometer

import numpy as np

class CalibAccel:

    sensitivity = 16384
    zeroX = zeroY = zeroZ = sensitivity/2
    expected_shape = (9, 3)

    def __init__(self, data_in=None):
        # Changed the kl parameter to 0.05(originally 0.01) - Connor
        # Also changed the cross axis contaminant scale factor intial guesses from .5to .25 - Connor
        # Configurable variables
        self.lambd = 1  # Damping Gain - Start with 1 TODO lambda is inline function call, need to rename
        # Damping parameter - must be < 1. Affects rate of convergence  Recommend to use k1 between 0.01 - 0.05
        self.k1 = 0.01
        self.tol = 1e-9  #300 vergence criterion threshold
        self.Rold = 100000  # Better to leave this No.big.
        self.Rnew, self.vold = None, None
        # No.Of iterations.If solutions don't converge then try increasing this. Usually converges within 20 iterations
        self.itr = 1000

        self.data = data_in     # assigned at instantiation or later on but can't calibrate without data
        self.done = False   # used with check_convergence
        self.R, self.J, self.H, self.D, self.Vx, self.Vy, self.Vz = None, None, None, None, None, None, None
        self.M, self.B = None, None  # main arrays to return
        self.input_len = None
        self.tmp = None     # for debugging TODO: remove when submitting to final

        # Initial Guess values of M and B. Change this only if you need to
        self.Mxx0 = 5.0
        self.Mxy0 = 0.25
        self.Mxz0 = 0.25
        self.Myy0 = 5.0
        self.Myz0 = 0.25
        self.Mzz0 = 5.0
        self.Bx0 = 0.5
        self.By0 = 0.5
        self.Bz0 = 0.5

        # set v to initial value guess
        self.v = np.array([[self.Mxx0], [self.Mxy0], [self.Mxz0], [self.Myy0],
                           [self.Myz0], [self.Mzz0], [self.Bx0], [self.By0], [self.Bz0]])

        self.error = None   # stores the error string

    def set_data(self, data_in):
        self.data = data_in

    # returns two arrays (scale vector matrix M and bias vector B)
    def calibrate(self):
        if not self.input_check():
            # more details stored in self.error
            return

        self.get_columns()  # gets the x,y,z values from data input

        for n in range(self.itr):
            self.calc_jacob()
            self.Rnew = np.linalg.norm(self.R)   # get magnitude of R (error function)

            self.H = np.linalg.inv(self.J.T @ self.J)  # Hessian matrix inv(J'*J)
            self.D = (self.J.T @ self.R).T  # (J'*R)'

            self.v = self.v - np.multiply(self.lambd, (self.D @ self.H).T)   # v - lambd*(D*H)'

            if self.Rnew <= self.Rold:
                self.lambd = self.lambd - self.k1*self.lambd
            else:
                self.lambd = self.k1*self.lambd

            if self.check_convergence(n):
                print('Calibration convergence\n converged at n: ', n)
                break
            else:
                self.update_vals()

        # Final results are stored in the object field
        self.M = np.array([[self.v[0, 0], self.v[1, 0], self.v[2, 0]],
                           [self.v[1, 0], self.v[3, 0], self.v[4, 0]],
                           [self.v[2, 0], self.v[4, 0], self.v[5, 0]]])
        self.B = np.array([[self.v[6, 0], self.v[7, 0], self.v[8, 0]]])
        return True

    # verifies that data is a 9x3 np.array (numpy suggests using arrays; matrix will be deprecated)
    def input_check(self):
        if self.data is None:  # null check
            self.error = 'No data'
            return False
        if not isinstance(self.data, np.ndarray):  # type check
            self.error = 'Input data is not of type '+str(type(np.ndarray))
            return False
        if np.shape(self.data) != CalibAccel.expected_shape:  # need 9 samples with the values (accel x,y,z)--> 9x3
            self.error = 'Input data does not match expected shape. Expected: ' + str(CalibAccel.expected_shape) \
                         + ' Received: ' + str(self.data.shape)
            # rows are samples and columns are the x,y,z values respectively
            # TODO: need at least 9, maybe accept larger array? rows > 9
            return False
        return True  # data is correct format

    # get's the x,y,z values form each sample in the data input array
    def get_columns(self):
        self.Vx = self.data[:, 0]    # get x values
        self.Vy = self.data[:, 1]    # get y values
        self.Vz = self.data[:, 2]    # get z values

    def calc_jacob(self):
        if self.R is None and self.J is None:
            self.input_len = len(self.data)
            self.R = np.zeros(shape=(self.input_len, 1))
            self.J = np.zeros(shape=(self.input_len, 9))
        for i in range(0, self.input_len):
            try:
                self.R[i, 0] = self.get_jacob(0, i)
            except ValueError:  # todo need to figure out why loops trigger ValueError Exception only on second call
                pass
            for j in range(1, 10):  # start is inclusive stop is exclusive --> 0 to 8
                try:
                    self.J[i, j-1] = self.get_jacob(j, i)
                except ValueError:  # exception thrown when trying to write a matrix via loop second time around
                    pass

    # partial derivatives of the error function with respect to gain and bias (elements of Jacobian vector)
    def get_jacob(self, f, i):
        """get the an element of the Jacobian Vector."""
        Mxx = self.Mxx0
        Mxy = self.Mxy0
        Mxz = self.Mxz0
        Myy = self.Myy0
        Myz = self.Myz0
        Mzz = self.Mzz0
        Bx = self.Bx0
        By = self.By0
        Bz = self.Bz0
        Vx = self.Vx[i]
        Vy = self.Vy[i]
        Vz = self.Vz[i]

        switcher = {
            0: (Mxx*(Bx - Vx) + Mxy*(By - Vy) + Mxz*(Bz - Vz))**2 + (Mxy*(Bx - Vx) + Myy*(By - Vy) + Myz*(Bz - Vz))**2 +
               (Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz))**2 - 1,
            1: 2*(Bx - Vx)*(Mxx*(Bx - Vx) + Mxy*(By - Vy) + Mxz*(Bz - Vz)),
            2: 2*(By - Vy)*(Mxx*(Bx - Vx) + Mxy*(By - Vy) + Mxz*(Bz - Vz)) + 2*(Bx - Vx)*(Mxy*(Bx - Vx) + Myy*(By - Vy)
                                                                                          + Myz*(Bz - Vz)),
            3: 2*(Bx - Vx)*(Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz)) + 2*(Bz - Vz)*(Mxx*(Bx - Vx) + Mxy*(By - Vy)
                                                                                          + Mxz*(Bz - Vz)),
            4: 2*(By - Vy)*(Mxy*(Bx - Vx) + Myy*(By - Vy) + Myz*(Bz - Vz)),
            5: 2*(By - Vy)*(Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz)) + 2*(Bz - Vz)*(Mxy*(Bx - Vx) + Myy*(By - Vy)
                                                                                          + Myz*(Bz - Vz)),
            6: 2*(Bz - Vz)*(Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz)),
            7: 2*Mxx*(Mxx*(Bx - Vx) + Mxy*(By - Vy) + Mxz*(Bz - Vz)) + 2*Mxy*(Mxy*(Bx - Vx) + Myy*(By - Vy)
                                            + Myz*(Bz - Vz)) + 2*Mxz*(Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz)),
            8: 2*Mxy*(Mxx*(Bx - Vx) + Mxy*(By - Vy) + Mxz*(Bz - Vz)) + 2*Myy*(Mxy*(Bx - Vx) + Myy*(By - Vy)
                                            + Myz*(Bz - Vz)) + 2*Myz*(Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz)),
            9: 2*Mxz*(Mxx*(Bx - Vx) + Mxy*(By - Vy) + Mxz*(Bz - Vz)) + 2*Myz*(Mxy*(Bx - Vx) + Myy*(By - Vy)
                                            + Myz*(Bz - Vz)) + 2*Mzz*(Mxz*(Bx - Vx) + Myz*(By - Vy) + Mzz*(Bz - Vz)),
        }
        return switcher.get(f)

    def check_convergence(self, n):
        if n > 1:
            # abs(max(2*(v-vold)/(v+vold))
            # print(abs(max(2 * np.linalg.lstsq(self.v + self.vold, self.v - self.vold, rcond=0)[0].T)))
            # print(np.linalg.lstsq(self.v+self.vold, self.v-self.vold, rcond=0)[0].T)
            # print(self.v+self.vold)
            # print(self.v-self.vold)
            if abs(max(2 * np.linalg.lstsq(self.v + self.vold, self.v - self.vold, rcond=0)[0].T)) <= self.tol:
                return True
            else:
                return False
            # if abs(max(2*((self.v-self.vold) / (self.v+self.vold)))) <= self.tol:
            #     return True
        return False

    # TODO: issue when copying from ndarray to float/double etc. Copies as ndarray type instead.
    def update_vals(self):
        self.Mxx0 = self.v[0, 0]
        self.Mxy0 = self.v[1, 0]
        self.Mxz0 = self.v[2, 0]
        self.Myy0 = self.v[3, 0]
        self.Myz0 = self.v[4, 0]
        self.Mzz0 = self.v[5, 0]
        self.Bx0 = self.v[6, 0]
        self.By0 = self.v[7, 0]
        self.Bz0 = self.v[8, 0]
        self.vold = self.v
        self.Rold = self.Rnew

    def test_calibration(self, raw_data):
        """expects a sample list of arrays as input with any row count for single sample"""
        corrected = list()
        for sample in raw_data:
            mag_list = [sample.shape[0]]
            for i in range(sample.shape[0]):
                curr = np.zeros(shape=(3, 1))
                curr[0, 0] = (sample[i, 0]-CalibAccel.zeroX)/CalibAccel.sensitivity
                curr[1, 0] = (sample[i, 1]-CalibAccel.zeroY)/CalibAccel.sensitivity
                curr[2, 0] = (sample[i, 1]-CalibAccel.zeroZ)/CalibAccel.sensitivity
                print('before: ', np.linalg.norm(curr))
                after = np.linalg.norm(self.M*(curr-self.B))
                print('after: ', after)
            corrected.append(mag_list)
            print('median mag for sample:', np.median(mag_list))


# TODO: test code goes here
# taken from a sample calibration using the matlab cal_accel.m
# data = np.array([[0.0157470703125000, 0.0239257812500000, 0.980834960937500],
#                  [1.00854492187500, 0.00927734375000000, 0.0256347656250000],
#                  [-0.985229492187500, 0.0391845703125000, 0.0512695312500000],
#                  [-0.0208740234375000, -0.988891601562500, 0.0385742187500000],
#                  [0.0406494140625000, 1.00793457031250, 0.0296630859375000],
#                  [-0.0137939453125000, -0.989379882812500, 0.0144042968750000],
#                  [0.00158691406250000, -0.000732421875000000, -1.03320312500000],
#                  [-0.0161132812500000, -0.191040039062500, -1.01464843750000],
#                  [0.981689453125000, -0.228759765625000, 0.00830078125000000]])
# print(np.shape(data))   # should be a (9,3) calibrate() will catch it
# calib = CalibAccel(data_in=data)
# calib.calibrate()
# print('calib.M: \n', calib.M, '\ncalib.B: \n', calib.B)
# expected = np.array([[0.598157662814795, 0.285498967615745, -1.90123191724753],
#                      [0.285498967615745, 1.78053602754867, 1.47516492408496],
#                      [-1.90123191724753, 1.47516492408496, -1.32553758219363]])
# print('expected', expected)
# print("equality check: ", np.array_equal(calib.M, expected))


# print('this is test code')
# data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                 np.int32)  # int32 is largest expected value 4 bytes --> probably could use shorts?
# print(type(np.shape(data)))
# print(np.shape(data))
#
#
# v = np.array([[random.randint(1,10)], [random.randint(1,10)], [random.randint(1,10)],
#               [random.randint(1,10)], [random.randint(1,10)]])
# v2 = np.array([[random.randint(1,10)], [random.randint(1,10)], [random.randint(1,10)],
#               [random.randint(1,10)], [random.randint(1,10)]])
# print(np.shape(v))
# print(v)
#
# # does transpose return a new array? answer is yes
# print('v: ', hex(id(v)))
# vnew = np.transpose(v)
# print('vnew: ', hex(id(vnew)))
# print(vnew)
#
#
# ans = np.multiply(v, v2)
# print(ans)
# M = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
# print('M:\n ', M)
# print(np.shape(M))
# print(type(M))
# M = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
# print('original: ', id(M))
# new = np.transpose(M)
# print('tranposed: ', id(new))
# print( id(M) == id(new))
# print(M)
# print(new)

test = np.array([[ 1.0400e+03, -6.0180e+03, 1.5038e+04],
 [ 1.6294e+04,  1.8000e+01,  3.3600e+02],
 [ 1.6288e+04,  6.0000e+00,  3.1400e+02],
 [-1.9200e+02, -4.5740e+03, -1.6224e+04],
 [ 4.0000e+00, -5.3440e+03,  1.5332e+04],
 [-1.7200e+02, -1.6290e+04,  2.8400e+02],
 [ 5.0000e+01,  1.6496e+04,  3.8000e+02],
 [-3.9760e+03, -3.6920e+03,  1.5314e+04],
 [-9.4180e+03, -6.8000e+01,  1.3300e+04]])
cal = CalibAccel(data_in=test)
cal.calibrate()
print(cal.M)