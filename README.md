# SY32-homography-project

The zip contains :

* the 5 video sequences to treat.
* "fennec.jpg" contains the image to insert into the video.

Command to extract the images from the sequences :  
```
ffmpeg -i input.mp4 %03d.png
```

Command to create a new video from the modified images ( low compression to preserve the quality) :  
```
ffmpeg -framerate 10 -i seq_%03d.png -c:v libx264rgb -crf 18 output.mp4
```

**Approach**:
* retrieve image by image
* detect white paper, to reduce noise (avoid mis-detecting things outside the paper, aka false positives )
  * apply closing to make sure the entire paper is detected
* detect colored circles 
  * detect a certain shade of color, within a specific error accepted ( variability) 
  * apply closing the make sure the entire colored circle is detected

* calculate the coordinates of the center of gravity of every colored circles
* 2 cases:
  * the 4 circles are detected, in this case, use the coordinates of the center of gravity to do the homography

  * one of the circles in not detected, in this case, predict/determin the coordinates of the 4th circle using the other 3 and apply the homography
    * to do so, detect the edges of the paper and use it to calculate a vector
    * the intersection of the vectors of 2 edges, applied from the other 2 circles closest, is the coord of the 4th circle


**Next steps**:
* consider if we'll apply color detection to every picture of the video
  or use some sort of affectation?

* consider how to calculate the position of the 4th circle, if we fail to detect one
  * either predict the coordinations of the 4th circle using the paper edges (the approach used for now)
  * * detect the edge (OK)
  * * calculate a vector of the edge (OK-ish)
  * * apply the edges to the circles and calculate intersection to find last coordinate (TO DO)

  * or, use the coordinates of the circles in the preceding photo, approximate the tranformation necessry to go from the last picture coordinates to the new one(for the 3 visible circles), and apply the same transformation to the last circle
k


