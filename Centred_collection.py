import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import os
import pickle
import time
from collections import Counter

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Color codes
YELLOW = '\033[93m'
PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'


def recenter_digit(image):
    """Recenters a digit within its 20x20 bounding box"""
    # Find the bounding box of the digit
    non_zero_indices = torch.nonzero(image > 0)
    if len(non_zero_indices) == 0:
        return image  # Blank image

    min_y = torch.min(non_zero_indices[:, 0])
    max_y = torch.max(non_zero_indices[:, 0])
    min_x = torch.min(non_zero_indices[:, 1])
    max_x = torch.max(non_zero_indices[:, 1])

    # Calculate center of mass
    y_coords, x_coords = torch.nonzero(image > 0, as_tuple=True)
    center_y = torch.mean(y_coords.float())
    center_x = torch.mean(x_coords.float())

    # Calculate offset to center the digit
    height, width = image.shape
    target_center_y = height / 2
    target_center_x = width / 2

    offset_y = int(target_center_y - center_y)
    offset_x = int(target_center_x - center_x)

    # Apply translation
    translated = torch.zeros_like(image)
    for y in range(height):
        for x in range(width):
            new_y = y + offset_y
            new_x = x + offset_x
            if 0 <= new_y < height and 0 <= new_x < width:
                translated[new_y, new_x] = image[y, x]

    return translated


def load_mnist_class_only(class_digit, batch_size=1024, max_samples=50000):
    """Load MNIST with only specified digit class"""
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)

    # Combine all data
    all_images = torch.cat([mnist_train.data, mnist_test.data])
    all_labels = torch.cat([mnist_train.targets, mnist_test.targets])

    # Filter only specified class
    class_mask = (all_labels == class_digit)
    class_images = all_images[class_mask].float() / 255.0
    class_labels = all_labels[class_mask]

    # Limit to max_samples
    if len(class_images) > max_samples:
        indices = torch.randperm(len(class_images))[:max_samples]
        class_images = class_images[indices]
        class_labels = class_labels[indices]

    # Recenter all digits
    print(f"{GREEN}Recentering {len(class_images)} {class_digit} digits...{ENDC}")
    recentered_images = torch.stack([recenter_digit(img) for img in tqdm(class_images, desc="Recentering")])

    print(f"{GREEN}Found {len(recentered_images)} {class_digit} digits in MNIST{ENDC}")

    # Create batches
    class_batches = [(recentered_images[i:i + batch_size], class_labels[i:i + batch_size])
                     for i in range(0, len(recentered_images), batch_size)]

    return class_batches


def fast_edge_detection(batch, device='cuda'):
    """Detect edges in the images"""
    batch = batch.to(device)
    padded = F.pad(batch.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0)
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device=device)
    neighbor_counts = F.conv2d(padded, kernel)
    edges = ((batch > 0) & (neighbor_counts.squeeze(1) < 8) & (neighbor_counts.squeeze(1) > 0))
    return edges


def extract_pattern_features(batch, edges, batch_offset, device='cuda'):
    """Extract pattern features from edge pixels (without label)"""
    batch = batch.to(device)
    edges = edges.to(device)
    edge_coords = torch.nonzero(edges)

    if len(edge_coords) == 0:
        return torch.empty((0, 11), device='cpu')

    # Get 8 neighbors for each edge pixel
    padded = F.pad(batch.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0).squeeze(1)
    shifts = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
                          dtype=torch.long, device=device)

    # Extract all neighbor values
    neighbor_values = torch.stack([
        padded[edge_coords[:, 0], edge_coords[:, 1] + dx + 1, edge_coords[:, 2] + dy + 1]
        for dx, dy in shifts
    ], dim=1)

    # Create rotary pattern features (sum of adjacent neighbors)
    rotary_patterns = neighbor_values + torch.roll(neighbor_values, shifts=1, dims=1)

    # Add metadata (without label)
    coordinates = edge_coords[:, 1:].float()  # (y, x) coordinates
    batch_indices = (edge_coords[:, 0] + batch_offset).float().unsqueeze(1)

    # Combine all features: [rotary_patterns, coordinates, batch_idx]
    features = torch.cat([rotary_patterns, coordinates, batch_indices], dim=1).cpu()

    return features


def extract_class_patterns(class_batches, class_digit, device='cuda'):
    """Extract patterns from a specific digit class"""
    all_patterns = []

    print(f"\n{YELLOW}=== Extracting Patterns from {class_digit} Digits ==={ENDC}")

    with tqdm(class_batches, desc=f"{YELLOW}Processing {class_digit} batches{ENDC}", unit="batch") as pbar:
        for batch_idx, (batch, _) in enumerate(pbar):
            edges = fast_edge_detection(batch, device)
            patterns = extract_pattern_features(batch, edges, batch_idx * batch.shape[0], device)

            if len(patterns) > 0:
                all_patterns.append(patterns)

            pbar.set_postfix({"patterns": f"{sum([len(p) for p in all_patterns])}"})

    if not all_patterns:
        return torch.empty((0, 11))

    all_patterns = torch.cat(all_patterns)
    print(f"\n{GREEN}Total patterns extracted for {class_digit}: {len(all_patterns)}{ENDC}")

    return all_patterns


def get_zone_id(y, x, image_size=28, grid_size=4):
    """Calculate which zone (0-15) a coordinate belongs to in a 4x4 grid"""
    zone_height = image_size / grid_size
    zone_width = image_size / grid_size

    zone_y = int(y / zone_height)
    zone_x = int(x / zone_width)

    return zone_y * grid_size + zone_x


def cluster_patterns_simple(patterns, tolerance=0.2):
    """Cluster patterns using the simple method: same zone and similar sums"""
    print(f"{BLUE}=== Clustering Patterns with Tolerance {tolerance} ==={ENDC}")

    if len(patterns) == 0:
        return np.array([]), {}

    # Convert to numpy for easier processing
    patterns_np = patterns.numpy()

    # Add a count column at the end (initially 1 for each pattern)
    patterns_with_count = np.hstack([patterns_np, np.ones((len(patterns_np), 1))])

    # Create a dictionary to store cluster representatives for each zone
    zone_clusters = {zone_id: {} for zone_id in range(16)}
    cluster_id_counter = 0

    with tqdm(total=len(patterns_with_count), desc=f"{PINK}Clustering patterns{ENDC}", unit="pattern") as pbar:
        for i, pattern in enumerate(patterns_with_count):
            # Extract features for comparison
            rotary_sums = pattern[:8]
            y_coord = pattern[8]
            x_coord = pattern[9]

            # Determine which zone this pattern belongs to
            zone_id = get_zone_id(y_coord, x_coord)

            # Check if this pattern matches any existing cluster in the same zone
            matched = False
            for c_id, cluster_info in zone_clusters[zone_id].items():
                cluster_pattern = cluster_info['pattern']
                cluster_rotary = cluster_pattern[:8]

                # Check if rotary sums are within tolerance
                if np.all(np.abs(rotary_sums - cluster_rotary) <= tolerance):
                    # Match found, increment count
                    zone_clusters[zone_id][c_id]['count'] += 1
                    matched = True
                    break

            # If no match found, create a new cluster in this zone
            if not matched:
                zone_clusters[zone_id][cluster_id_counter] = {
                    'pattern': pattern.copy(),
                    'count': 1
                }
                cluster_id_counter += 1

            pbar.update(1)
            pbar.set_postfix({"clusters": f"{sum(len(z) for z in zone_clusters.values())}"})

    # Combine all clusters from all zones
    all_clusters = {}
    for zone_id, clusters_in_zone in zone_clusters.items():
        all_clusters.update(clusters_in_zone)

    # Create the final clustered patterns array
    clustered_patterns = []

    for c_id, cluster_info in all_clusters.items():
        # Update the pattern with the count
        pattern_with_count = cluster_info['pattern'].copy()
        pattern_with_count[-1] = cluster_info['count']
        clustered_patterns.append(pattern_with_count)

    clustered_patterns = np.array(clustered_patterns)

    return clustered_patterns, all_clusters


def analyze_clusters(clustered_patterns, clusters, class_digit):
    """Analyze and display cluster statistics"""
    print(f"\n{PINK}=== CLUSTER ANALYSIS FOR {class_digit} ==={ENDC}")

    if len(clustered_patterns) == 0:
        print(f"{YELLOW}No clusters found{ENDC}")
        return

    n_clusters = len(clusters)
    total_patterns = np.sum(clustered_patterns[:, -1])

    print(f"{PINK}Number of clusters: {n_clusters}{ENDC}")
    print(f"{PINK}Total patterns: {total_patterns}{ENDC}")

    # Cluster size distribution
    cluster_sizes = clustered_patterns[:, -1]
    print(f"\n{PINK}Cluster size statistics:{ENDC}")
    print(f"  Average: {np.mean(cluster_sizes):.1f}")
    print(f"  Median: {np.median(cluster_sizes):.1f}")
    print(f"  Minimum: {np.min(cluster_sizes)}")
    print(f"  Maximum: {np.max(cluster_sizes)}")
    print(f"  Standard deviation: {np.std(cluster_sizes):.1f}")

    # Top clusters
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    print(f"\n{PINK}Top 10 largest clusters:{ENDC}")
    for i, idx in enumerate(sorted_indices[:10]):
        print(f"  Cluster {i + 1}: {cluster_sizes[idx]:>8} patterns")

    # Zone distribution
    zone_counts = {zone_id: 0 for zone_id in range(16)}
    for pattern in clustered_patterns:
        y_coord = pattern[8]
        x_coord = pattern[9]
        zone_id = get_zone_id(y_coord, x_coord)
        zone_counts[zone_id] += pattern[-1]  # Add the count of patterns in this cluster

    print(f"\n{PINK}Pattern distribution across zones:{ENDC}")
    for zone_id in range(16):
        zone_y = zone_id // 4
        zone_x = zone_id % 4
        print(f"  Zone ({zone_y},{zone_x}): {zone_counts[zone_id]:>6} patterns")

    return cluster_sizes


def save_clustered_patterns(clustered_patterns, class_digit, filename_prefix="clustered"):
    """Save clustered patterns to a file with class name"""
    if len(clustered_patterns) == 0:
        print(f"{YELLOW}No patterns to save for {class_digit}!{ENDC}")
        return

    filename = f"{filename_prefix}_{class_digit}.pkl"

    # Save with pickle
    with open(filename, 'wb') as f:
        pickle.dump(clustered_patterns, f)

    print(f"\n{GREEN}Saved {len(clustered_patterns)} clusters for {class_digit} to {filename}{ENDC}")

    # Also save a text sample
    sample_filename = f"clustered_{class_digit}_samples.txt"
    with open(sample_filename, 'w') as f:
        f.write(f"Total clusters: {len(clustered_patterns)}\n")
        f.write(f"Total patterns: {np.sum(clustered_patterns[:, -1])}\n")
        f.write("Feature format: [rotary_sum1, ..., rotary_sum8, Y, X, batch_index, count]\n")
        f.write("Zone mapping: 0-3 (top row), 4-7, 8-11, 12-15 (bottom row)\n\n")
        for i in range(min(10, len(clustered_patterns))):
            y_coord = clustered_patterns[i][8]
            x_coord = clustered_patterns[i][9]
            zone_id = get_zone_id(y_coord, x_coord)
            f.write(f"Cluster {i}: {clustered_patterns[i].tolist()} (Zone: {zone_id})\n")
    print(f"{GREEN}Sample clusters saved to {sample_filename}{ENDC}")


def process_class(class_digit, device='cuda', tolerance=0.2):
    """Process a single digit class: extract patterns and cluster them"""
    print(f"\n{BLUE}=== PROCESSING CLASS {class_digit} ==={ENDC}")

    # Load data for this class
    class_batches = load_mnist_class_only(class_digit, max_samples=50000)

    # Extract patterns
    patterns = extract_class_patterns(class_batches, class_digit, device)

    # Cluster patterns
    clustered_patterns, clusters = cluster_patterns_simple(patterns, tolerance)

    # Analyze clusters
    analyze_clusters(clustered_patterns, clusters, class_digit)

    # Save results
    save_clustered_patterns(clustered_patterns, class_digit)

    return clustered_patterns


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tolerance = 0.2

    # Process all classes from 0 to 9
    all_clustered_patterns = {}
    for class_digit in range(10):
        start_time = time.time()
        clustered_patterns = process_class(class_digit, device, tolerance)
        all_clustered_patterns[class_digit] = clustered_patterns
        elapsed_time = time.time() - start_time
        print(f"{GREEN}Completed class {class_digit} in {elapsed_time:.2f} seconds{ENDC}")

    print(f"\n{YELLOW}=== PROCESSING COMPLETED ==={ENDC}")
    for class_digit in range(10):
        if class_digit in all_clustered_patterns and len(all_clustered_patterns[class_digit]) > 0:
            print(
                f"{YELLOW}Class {class_digit}: {len(all_clustered_patterns[class_digit])} clusters, {np.sum(all_clustered_patterns[class_digit][:, -1])} total patterns{ENDC}")
        else:
            print(f"{YELLOW}Class {class_digit}: No clusters found{ENDC}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()