# Panoramic-Image-Stitching
The objective of the project is to create a panoramic image given a sequence of unstitched images.<br>
For this purpose, the main steps are:
1. Project the images on a cylinder surface (already provided function).
2. Extract the features from the projected images.
3. Compute the matching features between adjacent images.
4. Compute the translation between matching features.
5. Stitch images together.
<a/>
We added edge blurring affect between adjacent images during the stitching passage to smooth the image transition area and merged the last image with the first to close the loop.<br>
In the end we implemented a simple panoramic image viewer that allow us to see the merged image and rotate the view to left or right without restrictions.<br>
The implemented functions work also with colour images.
