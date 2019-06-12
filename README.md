# Accelerometer-Calibration
Calibration program for an IMU-MEMS accelerometer. 

## Input:
a 9x3 matrix where each row is a sample vector recorded in a unique static position. The number of unique samples can be inreased, but for now this is a fixed value. 

## Output: 
Scale matrix (M) and a bias vector (B). 

To apply the calibration solve A = M*(V-B) where V is a 3x1 vector of the x,y,z measurements taken from the IMU. 

NOTE: Input should be converted to g's before running the solver. Refer to your IMU-MEMS spec sheet for this info. 

Example plot of before/after calibration of data:

Error factor calcualted using chi^2:
* raw mag = 0.1088
* corrected mag = 0.0583
![alt text](https://trello-attachments.s3.amazonaws.com/5bbe2c9c29d3bd6dff10d5f0/5c5c8c02ea0e0701efc883e2/a67318ae7c306e97c9f9da26dd5837d9/Screen_Shot_2019-06-01_at_1.34.43_PM.png)
