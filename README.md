# multilayer-stixel-world
A real-time implementation of multi-layered stixel computation

![Real-Time-Stixel-Computation](https://github.com/dimovk/Real-Time-Stixel-Computation/tree/master/Example%20Output/stixels2.png)

## Description
- An implementation of the Multi-Layered Stixel computation based on [1].
- Extracts the Stixels from the input disparity map.
- Allows for multiple Stixels along every column.

## References
- [1] [The Stixel World - A Compact Medium-level Representation for Efficiently Modeling Three-dimensional Environments](https://www.mydlt.de/david/page/publications.html)
- gishi523

## Requirement
- OpenCV 3.4.9
- CUDA 9.0
- ZED SDK 3.1.0

## How to build
```
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to run
```
./stixels-live
```
  
## Author
dimovk# Real-Time-Stixel-Computation
