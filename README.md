# MMTS-PCA

Fault monitoring and visualization (FMV) as well as fault detection and classification (FDC) 
are significant for the purpose of predictive maintenance in manufacturing systems. FMV 
helps to identify abnormal conditions or unknown failures in advance differently from FDC. 
To reduce the distortion of the system state shift on a plot, in this study, a linearly transformed 
visualization tool for FMV, called MMTS-PCA, is developed by integrating the multiclass 
Mahalanobis-Taguchi system (MMTS), which is a distance-based multiclass classification
method, and the principal component analysis (PCA), which provides a 2D plot of visualizing 
the system state. We first propose a method, MMTS-w, for selecting common features to detect 
known failures using asymmetric weight. Next, MMTS-PCA is developed to clearly visualize 
the risk of failure on the PCA plot using the common features selected for failure monitoring. 
In particular, the Mahalanobis distance from the normal state is annotated as a contour on the 
plot. In the experiment, the proposed method is illustrated using the system failure data 
collected by vibration sensors in a real-world manufacturing system. It was proved that not 
only the failure detection of the MMTS-w was successful, but also the visualization 
performance of the MMTS-PCA for normal and failure modes was improved in terms of two 
clustering measures, the Davies-Bouldin index and the Silhouette score.
