import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

object_folder = r'C:\Users\jojo otstoi\OneDrive\Desktop\ObjectsNscenes\objects'
scene_folder = r'C:\Users\jojo otstoi\OneDrive\Desktop\ObjectsNscenes\scenes'
output_edges_object = r'C:\Users\jojo otstoi\OneDrive\Desktop\ObjectsNscenes\edges\objects'
output_edges_scene = r'C:\Users\jojo otstoi\OneDrive\Desktop\ObjectsNscenes\edges\scenes'
output_clustered_object = r'C:\Users\jojo otstoi\OneDrive\Desktop\ObjectsNscenes\clustered_edges\objects'
output_clustered_scene = r'C:\Users\jojo otstoi\OneDrive\Desktop\ObjectsNscenes\clustered_edges\scenes'

os.makedirs(output_edges_object, exist_ok=True)
os.makedirs(output_edges_scene, exist_ok=True)
os.makedirs(output_clustered_object, exist_ok=True)
os.makedirs(output_clustered_scene, exist_ok=True)

def process_images(input_folder, output_edges_folder, output_clustered_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            edge_output_path = os.path.join(output_edges_folder, filename)
            cv2.imwrite(edge_output_path, edges)
            pixels = edges.reshape((-1, 1))
            pixels = np.float32(pixels)
            kmeans = KMeans(n_clusters=2, random_state=0)
            labels = kmeans.fit_predict(pixels)
            clustered = labels.reshape(edges.shape)
            clustered_img = (clustered * 255).astype(np.uint8)
            cluster_output_path = os.path.join(output_clustered_folder, filename)
            cv2.imwrite(cluster_output_path, clustered_img)

process_images(object_folder, output_edges_object, output_clustered_object)
process_images(scene_folder, output_edges_scene, output_clustered_scene)
