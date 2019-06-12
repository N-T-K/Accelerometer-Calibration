# Accelerometer-Calibration
Calibration program for an IMU-MEMS accelerometer. 

##Input: a 9x3 matrix where each row is a sample vector recorded in a unique static position. The number of unique samples can be inreased, but for now this is a fixed value. 
##Output: Scale matrix (M) and a bias vector (B). 

To apply the calibration solve A = M*(V-B) where V is a 3x1 vector of the x,y,z measurements taken from the IMU. 

NOTE: Input should be converted to g's before running the solver. Refer to your IMU-MEMS spec sheet for this info. 

Error factor should be calcualted using chi^2. 
