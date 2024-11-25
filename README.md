<img src="https://cdn.sanity.io/images/kuana2sp/production-main/af463efe521a434f882e84ab2e28b855c8fe884e-1988x876.png?w=1920&fit=max&auto=format" style="width: 1000px; height:400px;">

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
 
  
</head>
<body>

<header>
  <h1>Image Segmentation Using K-Means Clustering with OpenCV</h1>
  <p>An intuitive approach to simplify images with clustering techniques</p>
</header>

<section>
  <h2>About the Project</h2>
  <p>This project demonstrates how to perform image segmentation using the <strong>K-Means Clustering</strong> algorithm with Python, OpenCV, and Scikit-learn. The goal is to reduce the image complexity by grouping similar pixel values into <em>clusters</em>, effectively segmenting the image into dominant colors or regions.</p>
</section>

<section>
  <h2>Features</h2>
  <ul>
    <li>Uses the <code>K-Means Clustering</code> algorithm for efficient segmentation.</li>
    <li>Integrates the <strong>Elbow Method</strong> to determine the optimal number of clusters (<code>K</code>).</li>
    <li>Visualizes results with <strong>Matplotlib</strong> for comparison between the original and segmented images.</li>
  </ul>
</section>

<section>
  <h2>Technologies Used</h2>
  <ul>
    <li><strong>Python</strong>: Programming language</li>
    <li><strong>OpenCV</strong>: Image processing library</li>
    <li><strong>Scikit-learn</strong>: Machine learning library</li>
    <li><strong>Matplotlib</strong>: Visualization library</li>
  </ul>
</section>

<section>
  <h2>How It Works</h2>
  <ol>
    <li>Load and preprocess an image using OpenCV.</li>
    <li>Reshape the image data into a 2D array of pixels with RGB values.</li>
    <li>Apply the <strong>Elbow Method</strong> to determine the optimal value of <code>K</code>.</li>
    <li>Run the <strong>K-Means Clustering</strong> algorithm to group pixels into clusters.</li>
    <li>Reconstruct the segmented image using the cluster centers.</li>
  </ol>
</section>

<section>
  <h2>Code Example</h2>
  <pre>

  ```
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```
```
# Load and preprocess image
image = cv2.imread("path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
```
# Reshape and cluster
pixels = image.reshape((-1, 3)).astype(np.float32)
kmeans = KMeans(n_clusters=3)
kmeans.fit(pixels)
```
```
# Map clusters back to image
segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
```
```
# Plot results
plt.imshow(segmented_image / 255)
plt.show()
```
  </pre>
</section>

<section>
  <h2>How to Run</h2>
  <ol>
    <li>Install the required libraries: <code>pip install opencv-python scikit-learn matplotlib</code></li>
    <li>Clone or download the repository.</li>
    <li>Run the Python script and provide an input image.</li>
    <li>Visualize the results in the output window.</li>
  </ol>
</section>


<section>
  <h2>Contributions</h2>
  
  <p>Feel free to contribute to this project! Fork the repository, make changes, and submit a pull request.</p>
</section>
<section>
  <h3>Acknowledgement</h3>
  <p>I would like to extend my deepest gratitude to Dr. Victor A.I, professor at Maharaja Institute of Technology Mysore,for his invaluable guidance throughout this project</p>
  
</section>

<footer>
  <p>Created by [MOHAMMED YOUSUF]</p>
  
</footer>

</body>
</html>
