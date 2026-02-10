import numpy as np
from sklearn.datasets import fetch_openml
from scipy import ndimage
import pickle


def load_mnist():
    """Load MNIST dataset - 50,000 samples only"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    return X[:50000], y[:50000]


def recenter_digit(digit, threshold=0.5):
    """Recenters the digit by calculating its center of mass"""
    digit_2d = digit.reshape(28, 28)
    binary_digit = (digit_2d > threshold * 255).astype(int)

    # Calculate center of mass
    cy, cx = ndimage.center_of_mass(binary_digit)

    # Calculate shift needed to center
    shift_y = 14 - int(cy)
    shift_x = 14 - int(cx)

    # Shift the digit
    shifted = np.roll(binary_digit, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)

    return shifted, (cy, cx)


def is_edge_pixel(digit, y, x):
    """Check if pixel is on the edge (has at least one zero neighbor)"""
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 28 and 0 <= nx < 28:
                if digit[ny, nx] == 0:
                    return True
            else:
                return True  # Border pixels are always edge
    return False


def find_starting_edge_pixel(digit):
    """Find a starting edge pixel on the right side"""
    # Start from right side and move left to find first edge pixel
    for x in range(27, -1, -1):
        for y in range(28):
            if digit[y, x] > 0 and is_edge_pixel(digit, y, x):
                return (y, x)
    return None


def get_clockwise_neighborhood(binary_image, center_y, center_x):
    """
    Extract the 24 neighborhood pixels in clockwise order starting from [-2, -2]
    """
    neighborhood = []

    # Define all positions in the correct clockwise order
    positions = [
        # Outer ring - top row
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        # Outer ring - right column
        (-1, 2), (0, 2), (1, 2), (2, 2),
        # Outer ring - bottom row
        (2, 1), (2, 0), (2, -1), (2, -2),
        # Outer ring - left column
        (1, -2), (0, -2), (-1, -2),
        # Inner ring - top row
        (-1, -1), (-1, 0), (-1, 1),
        # Inner ring - right column
        (0, 1), (1, 1),
        # Inner ring - bottom row
        (1, 0), (1, -1),
        # Inner ring - left column
        (0, -1)
    ]

    # Extract the values
    for dy, dx in positions:
        ny, nx = center_y + dy, center_x + dx

        # If the position is outside the image, consider it as background (0)
        if ny < 0 or ny >= binary_image.shape[0] or nx < 0 or nx >= binary_image.shape[1]:
            neighborhood.append(0)
        else:
            neighborhood.append(binary_image[ny, nx])

    return neighborhood


def get_zone(y, x):
    """Divide the 28x28 matrix into 16 zones (4x4 grid)"""
    zone_y = min(y // 7, 3)  # 28/4 = 7 pixels per zone vertically
    zone_x = min(x // 7, 3)  # 28/4 = 7 pixels per zone horizontally
    return zone_y * 4 + zone_x  # Zone number from 0 to 15


def process_digits(X, y):
    """Process all digits to extract edge patterns with zone information"""
    # Dictionary to store patterns by digit and zone
    digit_patterns = {d: {z: [] for z in range(16)} for d in range(10)}

    for i, (digit, label) in enumerate(zip(X, y)):
        if i % 5000 == 0:
            print(f"Processing digit {i + 1}/50000")

        # Recenter the digit
        recentered, _ = recenter_digit(digit)

        # Find starting edge pixel
        start_point = find_starting_edge_pixel(recentered)
        if start_point is None:
            continue

        # Find all edge pixels
        edge_pixels = set()
        for y in range(28):
            for x in range(28):
                if recentered[y, x] > 0 and is_edge_pixel(recentered, y, x):
                    edge_pixels.add((y, x))

        # Extract neighborhood patterns for each edge pixel
        for y, x in edge_pixels:
            # Get the zone for this pixel
            zone = get_zone(y, x)

            # Extract the 24-neighborhood pattern
            pattern = get_clockwise_neighborhood(recentered, y, x)

            # Store both pattern and zone information
            pattern_with_zone = {
                'pattern': pattern,
                'zone': zone,
                'position': (y, x)  # Also store the position for reference
            }

            # Add to the appropriate digit and zone
            digit_patterns[label][zone].append(pattern_with_zone)

    return digit_patterns


def save_patterns_with_zones(digit_patterns):
    """Save patterns with zone information for each digit"""
    for digit in range(10):
        # Create a list to store all patterns with their zone information
        all_patterns_with_zones = []

        # Also keep track of patterns by zone for statistics
        patterns_by_zone = {zone: [] for zone in range(16)}

        for zone in range(16):
            patterns_in_zone = digit_patterns[digit][zone]
            all_patterns_with_zones.extend(patterns_in_zone)
            patterns_by_zone[zone] = patterns_in_zone

        # Save to file
        filename = f"binary_{digit}_train_with_zones.pkl"
        with open(filename, 'wb') as f:
            pickle.dump({
                'digit': digit,
                'patterns_with_zones': all_patterns_with_zones,
                'patterns_by_zone': patterns_by_zone,
                'num_patterns': len(all_patterns_with_zones)
            }, f)
        print(f"Saved {len(all_patterns_with_zones)} patterns with zone info for digit {digit} to {filename}")


def main():
    # Load MNIST dataset (50,000 samples only)
    X, y = load_mnist()

    # Process all digits
    print("Processing digits to extract edge patterns with zone information...")
    digit_patterns = process_digits(X, y)

    # Save patterns with zone information
    print("Saving patterns with zone information to files...")
    save_patterns_with_zones(digit_patterns)

    # Print statistics
    print("\nPattern statistics:")
    for digit in range(10):
        total_patterns = sum(len(patterns) for patterns in digit_patterns[digit].values())
        print(f"Digit {digit}: {total_patterns} patterns")

        # Print per-zone statistics
        for zone in range(16):
            zone_patterns = len(digit_patterns[digit][zone])
            if zone_patterns > 0:
                print(f"  Zone {zone}: {zone_patterns} patterns")


if __name__ == "__main__":
    main()