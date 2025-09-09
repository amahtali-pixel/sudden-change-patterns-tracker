import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class PatternAnalyzer:
    def __init__(self, correct_patterns_file='correct_patterns.pkl', incorrect_patterns_file='incorrect_patterns.pkl'):
        self.correct_patterns_file = correct_patterns_file
        self.incorrect_patterns_file = incorrect_patterns_file
        self.correct_patterns = None
        self.incorrect_patterns = None
        self.filtered_incorrect_patterns = None

    def load_patterns(self):
        """Load patterns from files"""
        print("Loading patterns...")

        try:
            with open(self.correct_patterns_file, 'rb') as f:
                self.correct_patterns = pickle.load(f)
            print(f"Loaded {len(self.correct_patterns)} correct patterns")
        except FileNotFoundError:
            print(f"Warning: {self.correct_patterns_file} not found")
            self.correct_patterns = np.array([])

        try:
            with open(self.incorrect_patterns_file, 'rb') as f:
                self.incorrect_patterns = pickle.load(f)
            print(f"Loaded {len(self.incorrect_patterns)} incorrect patterns")
        except FileNotFoundError:
            print(f"Warning: {self.incorrect_patterns_file} not found")
            self.incorrect_patterns = np.array([])

        return len(self.correct_patterns), len(self.incorrect_patterns)

    def filter_pure_wrong_patterns(self, tolerance=1e-6):
        """
        Remove patterns that are involved in wrong predictions but not in correct predictions.

        Args:
            tolerance: Tolerance for considering patterns as similar

        Returns:
            Number of patterns removed, filtered incorrect patterns
        """
        print("\n=== FILTERING PURE WRONG PATTERNS ===")

        if len(self.correct_patterns) == 0 or len(self.incorrect_patterns) == 0:
            print("Need both correct and incorrect patterns for filtering")
            return 0, self.incorrect_patterns

        # Extract just the pattern features (excluding coordinates and labels)
        correct_features = self.correct_patterns[:, :8]
        incorrect_features = self.incorrect_patterns[:, :8]

        print(f"Comparing {len(correct_features)} correct vs {len(incorrect_features)} incorrect patterns...")

        # Create a KDTree for efficient nearest neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(correct_features)

        # Find distance to nearest correct pattern for each incorrect pattern
        distances, indices = tree.query(incorrect_features, k=1)

        # Patterns to keep: those that are NOT similar to any correct pattern
        keep_mask = distances > tolerance
        patterns_to_remove = len(incorrect_features) - np.sum(keep_mask)

        self.filtered_incorrect_patterns = self.incorrect_patterns[keep_mask]

        print(f"Removed {patterns_to_remove} incorrect patterns that were similar to correct patterns")
        print(f"Remaining incorrect patterns: {len(self.filtered_incorrect_patterns)}")
        print(f"Reduction: {patterns_to_remove / len(self.incorrect_patterns) * 100:.1f}%")

        return patterns_to_remove, self.filtered_incorrect_patterns

    def filter_by_feature_threshold(self, threshold_multiplier=1.2):
        """
        Filter patterns based on feature value thresholds.
        Removes incorrect patterns that have extreme feature values not seen in correct patterns.
        """
        print("\n=== FILTERING BY FEATURE THRESHOLDS ===")

        if len(self.correct_patterns) == 0 or len(self.incorrect_patterns) == 0:
            print("Need both correct and incorrect patterns for filtering")
            return self.incorrect_patterns

        correct_features = self.correct_patterns[:, :8]
        incorrect_features = self.incorrect_patterns[:, :8]

        # Calculate feature ranges from correct patterns
        feature_max = np.max(correct_features, axis=0)
        feature_min = np.min(correct_features, axis=0)

        # Set thresholds (patterns outside these ranges are considered abnormal)
        upper_threshold = feature_max * threshold_multiplier
        lower_threshold = feature_min / threshold_multiplier

        # Find incorrect patterns that fall outside normal ranges
        keep_mask = np.ones(len(incorrect_features), dtype=bool)

        for i in range(8):
            # Keep patterns that are within normal range for this feature
            in_range = (incorrect_features[:, i] >= lower_threshold[i]) & \
                       (incorrect_features[:, i] <= upper_threshold[i])
            keep_mask &= in_range

        patterns_to_remove = len(incorrect_features) - np.sum(keep_mask)
        filtered_patterns = self.incorrect_patterns[keep_mask]

        print(f"Removed {patterns_to_remove} incorrect patterns with abnormal feature values")
        print(f"Remaining incorrect patterns: {len(filtered_patterns)}")

        return filtered_patterns

    def filter_by_spatial_location(self, spatial_tolerance=1.0):
        """
        Filter patterns based on spatial location.
        Removes incorrect patterns that are in spatial regions not covered by correct patterns.
        """
        print("\n=== FILTERING BY SPATIAL LOCATION ===")

        if len(self.correct_patterns) == 0 or len(self.incorrect_patterns) == 0:
            print("Need both correct and incorrect patterns for filtering")
            return self.incorrect_patterns

        correct_coords = self.correct_patterns[:, 8:10]
        incorrect_coords = self.incorrect_patterns[:, 8:10]

        # Create spatial KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(correct_coords)

        # Find distance to nearest correct pattern for each incorrect pattern
        distances, indices = tree.query(incorrect_coords, k=1)

        # Keep patterns that are close to correct patterns spatially
        keep_mask = distances <= spatial_tolerance
        patterns_to_remove = len(incorrect_coords) - np.sum(keep_mask)
        filtered_patterns = self.incorrect_patterns[keep_mask]

        print(f"Removed {patterns_to_remove} incorrect patterns in unusual spatial locations")
        print(f"Remaining incorrect patterns: {len(filtered_patterns)}")

        return filtered_patterns

    def analyze_filtered_patterns(self, filtered_patterns):
        """Analyze the characteristics of filtered patterns"""
        print("\n=== ANALYSIS OF FILTERED PATTERNS ===")

        if filtered_patterns is None or len(filtered_patterns) == 0:
            print("No filtered patterns to analyze")
            return

        original_incorrect = len(self.incorrect_patterns)
        filtered_count = len(filtered_patterns)
        reduction = (original_incorrect - filtered_count) / original_incorrect * 100

        print(f"Original incorrect patterns: {original_incorrect:,}")
        print(f"Filtered incorrect patterns: {filtered_count:,}")
        print(f"Reduction: {reduction:.1f}%")

        # Analyze class distribution of remaining patterns
        true_labels = filtered_patterns[:, -2].astype(int)
        pred_labels = filtered_patterns[:, -1].astype(int)

        print("\nRemaining misclassifications by class:")
        for digit in range(10):
            count = np.sum(true_labels == digit)
            if count > 0:
                percentage = count / filtered_count * 100
                print(f"  True class {digit}: {count:>6,} patterns ({percentage:.1f}%)")

        # Most common remaining confusions
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for true_label, pred_label in zip(true_labels, pred_labels):
            confusion_matrix[true_label, pred_label] += 1

        print("\nTop remaining misclassifications:")
        confusions = []
        for i in range(10):
            for j in range(10):
                if i != j and confusion_matrix[i, j] > 0:
                    confusions.append((i, j, confusion_matrix[i, j]))

        confusions.sort(key=lambda x: x[2], reverse=True)
        for i, j, count in confusions[:10]:
            print(f"  {i} → {j}: {count:>6,} patterns")

    def save_filtered_patterns(self, filename='filtered_incorrect_patterns.pkl'):
        """Save the filtered patterns to a file"""
        if self.filtered_incorrect_patterns is not None:
            with open(filename, 'wb') as f:
                pickle.dump(self.filtered_incorrect_patterns, f)
            print(f"Filtered patterns saved to {filename}")
        else:
            print("No filtered patterns to save")

    def compare_pattern_characteristics(self, filtered_patterns):
        """Compare characteristics of original vs filtered incorrect patterns"""
        print("\n=== COMPARISON: ORIGINAL VS FILTERED PATTERNS ===")

        if filtered_patterns is None or len(filtered_patterns) == 0:
            print("No filtered patterns for comparison")
            return

        original_features = self.incorrect_patterns[:, :8]
        filtered_features = filtered_patterns[:, :8]

        print("Feature means comparison:")
        print("Feature | Original | Filtered | Difference")
        print("-" * 40)
        for i in range(8):
            orig_mean = np.mean(original_features[:, i])
            filt_mean = np.mean(filtered_features[:, i])
            diff = filt_mean - orig_mean
            print(f"F{i:6} | {orig_mean:.4f} | {filt_mean:.4f} | {diff:+.4f}")

    # [Keep all your existing methods here...]
    # basic_statistics, analyze_class_distribution, analyze_pattern_features,
    # analyze_spatial_distribution, compare_correct_vs_incorrect,
    # visualize_patterns, export_analysis_report, run_complete_analysis

    def run_filtering_analysis(self, tolerance=1e-6):
        """Run comprehensive filtering analysis"""
        print("=== PATTERN FILTERING ANALYSIS ===")

        # Load patterns first
        self.load_patterns()

        if len(self.correct_patterns) == 0 or len(self.incorrect_patterns) == 0:
            print("Need both correct and incorrect patterns for filtering")
            return

        # Run different filtering strategies
        print("\n1. Filtering pure wrong patterns (not similar to any correct patterns):")
        removed_count, filtered_patterns = self.filter_pure_wrong_patterns(tolerance)

        print("\n2. Analyzing filtered patterns:")
        self.analyze_filtered_patterns(filtered_patterns)

        print("\n3. Comparing characteristics:")
        self.compare_pattern_characteristics(filtered_patterns)

        # Save the filtered patterns
        self.save_filtered_patterns()

        return filtered_patterns


def main():
    """Main function to run pattern analysis and filtering"""
    analyzer = PatternAnalyzer()

    # Run complete analysis first
    analyzer.run_complete_analysis()

    # Then run filtering analysis
    print("\n" + "=" * 60)
    print("NOW RUNNING PATTERN FILTERING ANALYSIS")
    print("=" * 60)

    filtered_patterns = analyzer.run_filtering_analysis(tolerance=1e-6)


if __name__ == "__main__":
    main()