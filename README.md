# Toy Car Vision – Global Vision System

Car Tracking Server for the Robotics Course
Tracks 2 Cars and publishes the Real World Coordinates using UPD
to port 5000

```sh
# Starts the detection using inside a video
python3 src/toy_car_vision/server.py --source data/videos/long_car2.mp4
# Using a live Video Feed
python3 src/toy_car_vision/server.py --source 0
```

## Calibrate

The System designed to be used on a clear tile grid.
In order to calculate the real world coordinates based on a
inside a local coordinate System a calibration grid is required.

A Helper System is created to calibrate a positioning system

```sh
    python3 src/toy_car_vision/calibrate_ground_plane.py \
    --source data/videos/long_car2.mp4 --frame 120 \
    --world-points "-1200,0;0,0;1200,0;-1200,600;0,600;1200,600;-1200,1200;0,1200;1200,1200" \
    --ransac \
    --output calibration/ground_plane_manual.json
    # world-points: Marker Points. units in mm
    # ransac: safeguard that ignores grossly wrong points (> 20 mm off, e.g. a wrongly clicked crossing)
```

## How the Conversion works?
A homography matrix is constructed from the marker points within the image and the assumed/measured distance.
This matrix makes it possible to project points from one plane onto another.
In this application, points are projected from the “image plane” onto the “ground plane.”
