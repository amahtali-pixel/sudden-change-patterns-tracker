import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json  # Added missing import
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

    def basic_statistics(self):
        """Calculate basic statistics"""
        print("\n=== BASIC STATISTICS ===")

        total_patterns = len(self.correct_patterns) + len(self.incorrect_patterns)
        print(f"Total patterns: {total_patterns:,}")
        print(
            f"Correct patterns: {len(self.correct_patterns):,} ({len(self.correct_patterns) / total_patterns * 100:.1f}%)")
        print(
            f"Incorrect patterns: {len(self.incorrect_patterns):,} ({len(self.incorrect_patterns) / total_patterns * 100:.1f}%)")

        # Patterns per image (approximate)
        if len(self.correct_patterns) > 0:
            avg_patterns_correct = len(self.correct_patterns) / 40749  # 40749 correct predictions
            print(f"Average patterns per correct image: {avg_patterns_correct:.1f}")

        if len(self.incorrect_patterns) > 0:
            avg_patterns_incorrect = len(self.incorrect_patterns) / 9251  # 9251 incorrect predictions
            print(f"Average patterns per incorrect image: {avg_patterns_incorrect:.1f}")

    def analyze_class_distribution(self):
        """Analyze distribution across classes"""
        print("\n=== CLASS DISTRIBUTION ===")

        if len(self.correct_patterns) > 0:
            correct_true_labels = self.correct_patterns[:, -2].astype(int)
            correct_counts = Counter(correct_true_labels)

            print("Correct patterns by true class:")
            for digit in range(10):
                count = correct_counts.get(digit, 0)
                percentage = count / len(self.correct_patterns) * 100
                print(f"  Class {digit}: {count:>8,} patterns ({percentage:.1f}%)")

        if len(self.incorrect_patterns) > 0:
            incorrect_true_labels = self.incorrect_patterns[:, -2].astype(int)
            incorrect_pred_labels = self.incorrect_patterns[:, -1].astype(int)

            print("\nIncorrect patterns - True vs Predicted:")
            confusion_matrix = np.zeros((10, 10), dtype=int)

            for true_label, pred_label in zip(incorrect_true_labels, incorrect_pred_labels):
                confusion_matrix[true_label, pred_label] += 1

            # Print confusion matrix
            print("\nConfusion Matrix (True → Predicted):")
            print("T\\P", end="")
            for i in range(10):
                print(f"{i:>6}", end="")
            print()

            for i in range(10):
                print(f"{i:>3}", end="")
                for j in range(10):
                    print(f"{confusion_matrix[i, j]:>6}", end="")
                print()

            # Most common confusions
            print("\nTop misclassifications:")
            confusions = []
            for i in range(10):
                for j in range(10):
                    if i != j and confusion_matrix[i, j] > 0:
                        confusions.append((i, j, confusion_matrix[i, j]))

            confusions.sort(key=lambda x: x[2], reverse=True)
            for i, j, count in confusions[:10]:
                print(f"  {i} → {j}: {count:>6,} patterns")

            return confusion_matrix

    def analyze_pattern_features(self):
        """Analyze the pattern features themselves"""
        print("\n=== PATTERN FEATURE ANALYSIS ===")

        if len(self.correct_patterns) > 0:
            correct_features = self.correct_patterns[:, :8]  # Rotary patterns
            print("Correct patterns feature statistics:")
            print(f"  Mean values: {np.mean(correct_features, axis=0)}")
            print(f"  Std values: {np.std(correct_features, axis=0)}")
            print(f"  Min values: {np.min(correct_features, axis=0)}")
            print(f"  Max values: {np.max(correct_features, axis=0)}")

        if len(self.incorrect_patterns) > 0:
            incorrect_features = self.incorrect_patterns[:, :8]
            print("\nIncorrect patterns feature statistics:")
            print(f"  Mean values: {np.mean(incorrect_features, axis=0)}")
            print(f"  Std values: {np.std(incorrect_features, axis=0)}")

    def analyze_spatial_distribution(self):
        """Analyze spatial distribution of patterns"""
        print("\n=== SPATIAL DISTRIBUTION ===")

        if len(self.correct_patterns) > 0:
            correct_coords = self.correct_patterns[:, 8:10]  # x, y coordinates
            print("Correct patterns spatial distribution:")
            print(f"  Mean coordinates: ({np.mean(correct_coords[:, 0]):.2f}, {np.mean(correct_coords[:, 1]):.2f})")
            print(f"  Coordinate range: x[{np.min(correct_coords[:, 0]):.1f}-{np.max(correct_coords[:, 0]):.1f}], "
                  f"y[{np.min(correct_coords[:, 1]):.1f}-{np.max(correct_coords[:, 1]):.1f}]")

        if len(self.incorrect_patterns) > 0:
            incorrect_coords = self.incorrect_patterns[:, 8:10]
            print("\nIncorrect patterns spatial distribution:")
            print(f"  Mean coordinates: ({np.mean(incorrect_coords[:, 0]):.2f}, {np.mean(incorrect_coords[:, 1]):.2f})")

    def compare_correct_vs_incorrect(self):
        """Compare features between correct and incorrect patterns"""
        print("\n=== CORRECT VS INCORRECT COMPARISON ===")

        if len(self.correct_patterns) > 0 and len(self.incorrect_patterns) > 0:
            correct_features = self.correct_patterns[:, :8]
            incorrect_features = self.incorrect_patterns[:, :8]

            print("Feature comparison (correct vs incorrect):")
            for i in range(8):
                correct_mean = np.mean(correct_features[:, i])
                incorrect_mean = np.mean(incorrect_features[:, i])
                diff = incorrect_mean - correct_mean
                print(f"  Feature {i}: {correct_mean:.3f} vs {incorrect_mean:.3f} (diff: {diff:+.3f})")

            # Compare coordinate distributions
            correct_coords = self.correct_patterns[:, 8:10]
            incorrect_coords = self.incorrect_patterns[:, 8:10]

            print(f"\nSpatial comparison:")
            print(f"  Correct mean: ({np.mean(correct_coords[:, 0]):.2f}, {np.mean(correct_coords[:, 1]):.2f})")
            print(f"  Incorrect mean: ({np.mean(incorrect_coords[:, 0]):.2f}, {np.mean(incorrect_coords[:, 1]):.2f})")

    def visualize_patterns(self, n_samples=1000):
        """Create visualizations of the patterns"""
        print("\n=== CREATING VISUALIZATIONS ===")

        # Sample patterns for visualization (to avoid memory issues)
        if len(self.correct_patterns) > n_samples:
            correct_sample = self.correct_patterns[
                np.random.choice(len(self.correct_patterns), n_samples, replace=False)]
        else:
            correct_sample = self.correct_patterns

        if len(self.incorrect_patterns) > n_samples:
            incorrect_sample = self.incorrect_patterns[
                np.random.choice(len(self.incorrect_patterns), n_samples, replace=False)]
        else:
            incorrect_sample = self.incorrect_patterns

        # 1. Feature distribution comparison
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        if len(correct_sample) > 0:
            plt.hist(correct_sample[:, 0], alpha=0.7, bins=30, label='Correct', density=True)
        if len(incorrect_sample) > 0:
            plt.hist(incorrect_sample[:, 0], alpha=0.7, bins=30, label='Incorrect', density=True)
        plt.title('Feature 0 Distribution')
        plt.legend()

        plt.subplot(2, 3, 2)
        if len(correct_sample) > 0:
            plt.hist(correct_sample[:, 4], alpha=0.7, bins=30, label='Correct', density=True)
        if len(incorrect_sample) > 0:
            plt.hist(incorrect_sample[:, 4], alpha=0.7, bins=30, label='Incorrect', density=True)
        plt.title('Feature 4 Distribution')
        plt.legend()

        # 2. Spatial distribution
        plt.subplot(2, 3, 3)
        if len(correct_sample) > 0:
            plt.scatter(correct_sample[:, 8], correct_sample[:, 9], alpha=0.3, label='Correct', s=1)
        if len(incorrect_sample) > 0:
            plt.scatter(incorrect_sample[:, 8], incorrect_sample[:, 9], alpha=0.3, label='Incorrect', s=1)
        plt.title('Spatial Distribution')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()

        # 3. Class distribution
        plt.subplot(2, 3, 4)
        if len(self.correct_patterns) > 0:
            correct_labels = self.correct_patterns[:, -2].astype(int)
            correct_counts = [np.sum(correct_labels == i) for i in range(10)]
            plt.bar(range(10), correct_counts, alpha=0.7, label='Correct')

        if len(self.incorrect_patterns) > 0:
            incorrect_labels = self.incorrect_patterns[:, -2].astype(int)  # FIXED: Changed ast to astype
            incorrect_counts = [np.sum(incorrect_labels == i) for i in range(10)]
            plt.bar(range(10), incorrect_counts, alpha=0.7, label='Incorrect', bottom=correct_counts)

        plt.title('Patterns by True Class')
        plt.xlabel('Digit Class')
        plt.ylabel('Number of Patterns')
        plt.legend()

        # 4. Feature correlation
        plt.subplot(2, 3, 5)
        if len(correct_sample) > 0:
            corr_matrix = np.corrcoef(correct_sample[:, :8].T)
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        xticklabels=[f'F{i}' for i in range(8)],
                        yticklabels=[f'F{i}' for i in range(8)])
            plt.title('Feature Correlation (Correct)')

        plt.tight_layout()
        plt.savefig('pattern_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved to 'pattern_analysis_visualizations.png'")

        # 5. t-SNE visualization (if enough samples)
        if len(correct_sample) > 100 and len(incorrect_sample) > 100:
            try:
                print("Creating t-SNE visualization...")
                combined_samples = np.vstack([correct_sample[:500], incorrect_sample[:500]])
                labels = ['Correct'] * min(500, len(correct_sample)) + ['Incorrect'] * min(500, len(incorrect_sample))

                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                tsne_results = tsne.fit_transform(combined_samples[:, :8])

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                                      c=[0 if l == 'Correct' else 1 for l in labels],
                                      alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, label='Pattern Type (0=Correct, 1=Incorrect)')
                plt.title('t-SNE Visualization of Patterns')
                plt.savefig('pattern_tsne_visualization.png', dpi=300, bbox_inches='tight')
                print("t-SNE visualization saved to 'pattern_tsne_visualization.png'")
            except Exception as e:
                print(f"t-SNE failed: {e}")

    def export_analysis_report(self):
        """Export a comprehensive analysis report"""
        print("\n=== EXPORTING ANALYSIS REPORT ===")

        report = {
            'total_patterns': len(self.correct_patterns) + len(self.incorrect_patterns),
            'correct_patterns_count': len(self.correct_patterns),
            'incorrect_patterns_count': len(self.incorrect_patterns),
            'correct_percentage': len(self.correct_patterns) / (
                        len(self.correct_patterns) + len(self.incorrect_patterns)) * 100,
            'class_distribution_correct': {},
            'class_distribution_incorrect': {}
        }

        if len(self.correct_patterns) > 0:
            correct_labels = self.correct_patterns[:, -2].astype(int)
            for digit in range(10):
                report['class_distribution_correct'][digit] = int(np.sum(correct_labels == digit))

        if len(self.incorrect_patterns) > 0:
            incorrect_labels = self.incorrect_patterns[:, -2].astype(int)
            for digit in range(10):
                report['class_distribution_incorrect'][digit] = int(np.sum(incorrect_labels == digit))

        # Save report
        with open('pattern_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("Analysis report saved to 'pattern_analysis_report.json'")
        return report

    def run_complete_analysis(self):
        """Run all analysis steps"""
        print("=== PATTERN ANALYSIS STARTED ===")

        # Load patterns
        correct_count, incorrect_count = self.load_patterns()

        if correct_count == 0 and incorrect_count == 0:
            print("No patterns found to analyze!")
            return

        # Run all analysis steps
        self.basic_statistics()
        confusion_matrix = self.analyze_class_distribution()
        self.analyze_pattern_features()
        self.analyze_spatial_distribution()
        self.compare_correct_vs_incorrect()
        self.visualize_patterns()
        report = self.export_analysis_report()

        print("\n=== ANALYSIS COMPLETED ===")
        return report, confusion_matrix


def main():
    """Main function to run pattern analysis"""
    analyzer = PatternAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()