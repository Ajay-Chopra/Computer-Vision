# My Personal Projects in Computer Vision

### Inspiration
Computer vision has long been an interest of mine ever since I began studying computer science. The use cases for robust intelligent vision software are growing more numerous every year, including everything from detecting early stage lung cancer to autonomous vehicles. This repository contains some small files that utilize common computer vision techniques to perform interactive functions such as live face swapping, yawn detection and object tracking. By no means are these groundbreaking projects, but they do show how developers can leverage the power of cv algorithms to create some fun applications simply and quickly. 




### Projects

#### Live Face Swapper
This code lets the user implement live face swapping using a webcam. Using an image from the user's local directory, the program maps that
image onto the user's face in real time. The project uses several common CV techniques, such as the DLIB implementation of the Viola and Jones facial landmark detection algorithm. After the facial landmarks of the directory image are identified, the face is translated, scaled and rotated to fit over the webcam image. Lastly, I modify the color balance of the directory image to match that of the first, along with blending some of the features around the edges to make the transition more seamless. One result can be seen below.


![Office Face Swap](https://github.com/Ajay-Chopra/Computer-Vision/blob/master/Images/officeFaceSwap1.jpg)



#### Yawn Detector
The inspiration for this project came when one of my coworkers, during a particularly boring afternoon, asked me 'How many times do you think
you yawn at your desk per day?' I decided it would be fun to build a live yawn detector
that could detect when a subject is yawning based on the position of their top and bottom lip. It uses the same algorithms to determine 
facial landmarks as the face swapping app, but instead only focuses on those mapping to the top and bottom lip. Though my program is short,
simple and quite inaccurate by industry standards, similar programs are being used in DMS (Driver Monitoring Systems) to detect if a driver
is becoming drowsy. 


#### Object Tracker
This file allows a user to use their webcam in order to track an object as they move it across the screen. The program expects the user to input three integers which represent the RGB values of the object they wish to track. This is because I use color filtering to help identify the contours of the object. Once the contours are identified, I locate the centroid positon and use the line function of OpenCV to track the movement of the object. Note that I threshold the size of tracked images at 15px. So, if the bouding circle around the object is less that 15px in raidus, the object stops being tracked. There is also some strange initialization behavior that I have not been able to determine the cause of, but the program is fairly robust otherwise. Below is the output of the program being used to track a tennis ball in a video.


