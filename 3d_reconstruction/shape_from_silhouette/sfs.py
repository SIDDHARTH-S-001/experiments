import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

def load_projection_matrix(pa_file):
    """Load projection matrix from .pa file with corrected parsing"""
    with open(pa_file, 'r') as f:
        lines = f.readlines()
    
    # Extract all numerical values (should be 12 for 3x4 matrix)
    values = []
    for line in lines:
        if line.strip():  # Skip empty lines
            values.extend(map(float, line.split()))
    
    if len(values) == 8:
        # Handle the 2x4 case by adding a third row [0,0,0,1]
        print(f"Warning: Only 8 values found in {pa_file}, adding default third row")
        values.extend([0, 0, 0, 1])
    elif len(values) != 12:
        raise ValueError(f"Expected 12 values for 3x4 matrix, got {len(values)} in {pa_file}")
    
    # Reshape into 3x4 matrix
    return np.array(values).reshape(3, 4)

def binarize_image(image_path, threshold=128):
    """Convert image to binary silhouette"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def create_voxel_grid(bounds, resolution=50):
    """Create a voxel grid within given bounds"""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    z = np.linspace(bounds[4], bounds[5], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.ones_like(xx, dtype=bool)
    return xx, yy, zz, voxels

def project_voxels(voxels, P, image_shape):
    """Project voxels onto image plane using projection matrix P"""
    # Convert voxels to homogeneous coordinates
    voxel_coords = np.column_stack([voxels[0].flatten(), 
                                   voxels[1].flatten(), 
                                   voxels[2].flatten(), 
                                   np.ones(len(voxels[0].flatten()))])
    
    # Project to 2D (P is 3x4)
    projected = P @ voxel_coords.T
    projected = projected / projected[2, :]  # normalize by z
    
    # Reshape and scale to image coordinates
    u = np.round(projected[0, :]).astype(int)
    v = np.round(projected[1, :]).astype(int)
    
    # Create mask of valid projections (within image bounds)
    valid = (u >= 0) & (u < image_shape[1]) & (v >= 0) & (v < image_shape[0])
    
    return u, v, valid

def carve_voxels(voxels, silhouette, P, image_shape):
    """Carve voxels that don't project into the silhouette"""
    u, v, valid = project_voxels(voxels, P, image_shape)
    
    # Initialize mask (True means keep the voxel)
    mask = np.ones_like(voxels[3], dtype=bool)
    
    # For each voxel, check if it projects inside the silhouette
    for idx in np.ndindex(voxels[3].shape):
        flat_idx = np.ravel_multi_index(idx, voxels[3].shape)
        if valid[flat_idx]:
            if silhouette[v[flat_idx], u[flat_idx]] == 0:
                mask[idx] = False
    
    # Apply the mask
    voxels[3][~mask] = False
    
    return voxels

def visualize_visual_hull(voxels, step=None):
    """Visualize the current state of the visual hull"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract surface voxels using marching cubes
    if np.any(voxels[3]):
        try:
            # Adjust level for marching cubes
            verts, faces, _, _ = measure.marching_cubes(voxels[3].astype(np.float32), level=0.5)
            
            # Scale vertices back to original coordinates
            verts[:, 0] = voxels[0].min() + verts[:, 0] * (voxels[0].max() - voxels[0].min()) / (voxels[0].shape[0] - 1)
            verts[:, 1] = voxels[1].min() + verts[:, 1] * (voxels[1].max() - voxels[1].min()) / (voxels[1].shape[0] - 1)
            verts[:, 2] = voxels[2].min() + verts[:, 2] * (voxels[2].max() - voxels[2].min()) / (voxels[2].shape[0] - 1)
            
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                           cmap='Spectral', lw=1, alpha=0.8)
        except RuntimeError as e:
            print(f"Marching cubes error at step {step}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error visualizing at step {step}: {str(e)}")
    else:
        print(f"No voxels remaining at step {step}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Visual Hull {f"after step {step}" if step is not None else ""}')
    plt.tight_layout()
    plt.show()

def visualize_silhouette_and_contour(silhouette, step):
    """Visualize the silhouette and its contour"""
    plt.figure(figsize=(8, 6))
    plt.imshow(silhouette, cmap='gray')
    
    # Find contours
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
    
    plt.title(f'Silhouette and Contour (Step {step})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    num_images = 18  # david_00 to david_17
    bounds = [-2, 2, -2, 2, -2, 2]  # Expanded bounds
    resolution = 50  # Reduced for testing
    image_folder = "D:/Onedrive/experiments/experiments/3d_reconstruction/shape_from_silhouette/data/KKeishiro Shape-from-Silhouettes master data"
    
    # Debug: Verify folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    print(f"Processing images from: {image_folder}")
    
    # Create voxel grid
    xx, yy, zz, voxels = create_voxel_grid(bounds, resolution)
    voxel_grid = (xx, yy, zz, voxels)
    
    # Process each image
    for i in range(num_images):
        print(f"\nProcessing image {i}...")
        
        # Load image and projection matrix
        img_path = os.path.join(image_folder, f'david_{i:02d}.jpg')
        pa_path = os.path.join(image_folder, f'david_{i:02d}.pa')
        
        if not os.path.exists(img_path):
            print(f"Warning: Missing image file {img_path}")
            continue
        if not os.path.exists(pa_path):
            print(f"Warning: Missing projection file {pa_path}")
            continue
        
        # Binarize image
        try:
            silhouette = binarize_image(img_path)
            print(f"Silhouette shape: {silhouette.shape}")
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue
        
        # Load projection matrix
        try:
            P = load_projection_matrix(pa_path)
            print(f"Projection matrix:\n{P}")
        except Exception as e:
            print(f"Error loading projection matrix {pa_path}: {str(e)}")
            continue
        
        # Visualize silhouette and contour
        visualize_silhouette_and_contour(silhouette, i)
        
        # Carve voxels
        voxel_grid = carve_voxels(voxel_grid, silhouette, P, silhouette.shape)
        
        # Visualize intermediate result
        if i % 3 == 0 or i == num_images - 1:
            visualize_visual_hull(voxel_grid, i)
    
    # Final visualization
    visualize_visual_hull(voxel_grid, "final")

if __name__ == "__main__":
    main()