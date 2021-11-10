import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('TkAgg')


class Clustering:
    def KmeansCluster(self,cloud,k,centers=None):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np = cloud
        if centers!=None:
            cc= np.array(list(centers.values()))
            kmeans = KMeans(n_clusters=k,init=cc,max_iter=10)
        else:
            kmeans = KMeans(n_clusters=k)

        clusters=kmeans.fit_predict(cloud_np[:,0:3])
        # colors=["red","green","blue","cyan","magenta","yellow","black","red","green","blue","cyan","magenta","yellow","black"]

        clouds={}
        for i in range(k):
            clouds[i]=cloud_np[clusters==i,:]
        return clouds
        # for i in range(k):
        #     plt.scatter(cloud_np[clusters==i,0],cloud_np[clusters==i,1],color=colors[i])

        # plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="red")
        # plt.show()

    def HierarchialCluster(self,cloud,k,centers=None):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np = cloud

        agglomerative=AgglomerativeClustering(n_clusters=k,linkage="average" ) #linkage{“ward”, “complete”, “average”, “single”}, default=”ward”  connectivity="callable"
        clusters=agglomerative.fit_predict(cloud_np[:,0:3])
        clouds={}
        for i in range(k):
            clouds[i]=cloud_np[clusters==i,:]
        return clouds

    def GaussianCluster(self,cloud,k,centers=None):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np = cloud

        gmm = GaussianMixture(n_components=k)
        clusters = gmm.fit_predict(cloud_np[:,0:3])

        clouds={}
        for i in range(k):
            clouds[i]=cloud_np[clusters==i,:]
        return clouds

    def OpticsCluster(self,cloud,k,centers=None):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np = cloud
        # db = DBSCAN(eps=0.03, min_samples=10)
        agglomerative=OPTICS(min_samples = 10, xi = 0.45, min_cluster_size = 0.05) #,max_eps=0.1linkage{“ward”, “complete”, “average”, “single”}, default=”ward”
        clusters=agglomerative.fit_predict(cloud_np[:,0:3])
        clouds={}
        for i in range(k):
            clouds[i]=cloud_np[clusters==i,:]
        return clouds

    def SpectralCluster(self,cloud,k,centers=None):
        if type(cloud) != np.ndarray:
            cloud_np = cloud.to_array()
        else:
            cloud_np = cloud

        spectral=SpectralClustering(n_clusters=k,eigen_solver='arpack',  assign_labels='discretize')

        cc=cloud_np[:5000,0:2]
        cc4=cloud_np[:5000,0:4]
        clusters=spectral.fit_predict(cc)

        clouds={}
        for i in range(k):
            clouds[i]=cc4[clusters==i,:]
        return clouds