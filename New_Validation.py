import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
import json
import os
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')


class ConflictPatternManager:
    def __init__(self, patterns_file='incorrect_patterns.pkl', output_dir='conflict_patterns'):
        self.patterns_file = patterns_file
        self.output_dir = output_dir
        self.patterns = None
        self.confusion_matrix = None
        self.conflicting_pairs = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_patterns(self):
        """Load incorrect patterns from file"""
        print("Loading patterns...")
        try:
            with open(self.patterns_file, 'rb') as f:
                self.patterns = pickle.load(f)
            print(f"Loaded {len(self.patterns)} incorrect patterns")
            return True
        except FileNotFoundError:
            print(f"Error: {self.patterns_file} not found")
            return False
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return False

    def build_confusion_matrix(self):
        """Build confusion matrix from incorrect patterns"""
        if self.patterns is None:
            print("Please load patterns first")
            return

        true_labels = self.patterns[:, -2].astype(int)
        pred_labels = self.patterns[:, -1].astype(int)

        self.confusion_matrix = np.zeros((10, 10), dtype=int)
        for true, pred in zip(true_labels, pred_labels):
            if true != pred:  # Only count misclassifications
                self.confusion_matrix[true, pred] += 1

        print("Confusion matrix built successfully")

    def find_top_confusing_pairs(self, min_count=10, top_k=10):
        """Find the most confusing class pairs"""
        if self.confusion_matrix is None:
            self.build_confusion_matrix()

        confusing_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and self.confusion_matrix[i, j] >= min_count:
                    confusing_pairs.append((i, j, self.confusion_matrix[i, j]))

        # Sort by frequency descending
        confusing_pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop {min(top_k, len(confusing_pairs))} most confusing class pairs:")
        for i, (true_class, pred_class, count) in enumerate(confusing_pairs[:top_k]):
            print(f"{i + 1}. {true_class} → {pred_class}: {count} misclassifications")

        return confusing_pairs[:top_k]

    def extract_conflict_patterns(self, class_n, class_m, max_patterns_per_class=1000):
        """
        Extract patterns that cause confusion between class n and class m

        Returns:
            dict: Contains patterns from both directions and analysis results
        """
        if self.patterns is None:
            print("Please load patterns first")
            return None

        true_labels = self.patterns[:, -2].astype(int)
        pred_labels = self.patterns[:, -1].astype(int)

        # Patterns where n is misclassified as m
        n_to_m_mask = (true_labels == class_n) & (pred_labels == class_m)
        n_to_m_patterns = self.patterns[n_to_m_mask]

        # Patterns where m is misclassified as n
        m_to_n_mask = (true_labels == class_m) & (pred_labels == class_n)
        m_to_n_patterns = self.patterns[m_to_n_mask]

        # Limit to maximum patterns
        n_to_m_patterns = n_to_m_patterns[:max_patterns_per_class]
        m_to_n_patterns = m_to_n_patterns[:max_patterns_per_class]

        print(f"Extracted {len(n_to_m_patterns)} {class_n}→{class_m} patterns")
        print(f"Extracted {len(m_to_n_patterns)} {class_m}→{class_n} patterns")

        # Analyze what makes these patterns confusing
        analysis = self._analyze_confusion_patterns(n_to_m_patterns, m_to_n_patterns, class_n, class_m)

        conflict_data = {
            'class_pair': (class_n, class_m),
            'n_to_m_patterns': n_to_m_patterns,
            'm_to_n_patterns': m_to_n_patterns,
            'analysis': analysis,
            'total_patterns': len(n_to_m_patterns) + len(m_to_n_patterns),
            'extraction_date': pd.Timestamp.now().isoformat()
        }

        self.conflicting_pairs[(class_n, class_m)] = conflict_data
        return conflict_data

    def _analyze_confusion_patterns(self, n_to_m_patterns, m_to_n_patterns, class_n, class_m):
        """Analyze what features cause confusion between the two classes"""
        analysis = {
            'feature_overlaps': [],
            'spatial_proximity': None,
            'confidence_scores': None,
            'recommended_filters': [],
            'error': None
        }

        # Check if we have enough patterns for analysis
        if len(n_to_m_patterns) == 0 or len(m_to_n_patterns) == 0:
            analysis['error'] = "Not enough patterns for analysis (one direction missing)"
            return analysis

        # Analyze feature overlaps
        n_to_m_features = n_to_m_patterns[:, :8]
        m_to_n_features = m_to_n_patterns[:, :8]

        for feature_idx in range(8):
            n_m_mean = np.mean(n_to_m_features[:, feature_idx])
            m_n_mean = np.mean(m_to_n_features[:, feature_idx])
            n_m_std = np.std(n_to_m_features[:, feature_idx])
            m_n_std = np.std(m_to_n_features[:, feature_idx])

            # Calculate overlap score (0-1, higher = more overlap)
            overlap_score = 1 - abs(n_m_mean - m_n_mean) / (n_m_std + m_n_std + 1e-10)
            overlap_score = max(0, min(1, overlap_score))

            analysis['feature_overlaps'].append({
                'feature_index': feature_idx,
                'overlap_score': float(overlap_score),
                'n_to_m_mean': float(n_m_mean),
                'm_to_n_mean': float(m_n_mean),
                'n_to_m_std': float(n_m_std),
                'm_to_n_std': float(m_n_std)
            })

        # Sort by overlap score (most overlapping features first)
        analysis['feature_overlaps'].sort(key=lambda x: x['overlap_score'], reverse=True)

        # Analyze spatial proximity
        n_to_m_coords = n_to_m_patterns[:, 8:10]
        m_to_n_coords = m_to_n_patterns[:, 8:10]

        if len(n_to_m_coords) > 0 and len(m_to_n_coords) > 0:
            tree = cKDTree(n_to_m_coords)
            distances, _ = tree.query(m_to_n_coords, k=1)
            avg_distance = float(np.mean(distances))
            analysis['spatial_proximity'] = avg_distance

        return analysis

    def save_conflict_patterns(self, class_n, class_m, filename=None):
        """Save conflict patterns to file"""
        if (class_n, class_m) not in self.conflicting_pairs:
            print(f"No conflict patterns extracted for {class_n} vs {class_m}")
            return False

        if filename is None:
            filename = f"conflict_{class_n}_vs_{class_m}.pkl"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.conflicting_pairs[(class_n, class_m)], f)
            print(f"Conflict patterns saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving conflict patterns: {e}")
            return False

    def save_all_conflicts(self):
        """Save all extracted conflict patterns"""
        success_count = 0
        for (class_n, class_m) in self.conflicting_pairs:
            if self.save_conflict_patterns(class_n, class_m):
                success_count += 1

        print(f"Saved {success_count} conflict pattern files")
        return success_count

    def load_conflict_patterns(self, class_n, class_m, filename=None):
        """Load previously saved conflict patterns"""
        if filename is None:
            filename = f"conflict_{class_n}_vs_{class_m}.pkl"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'rb') as f:
                conflict_data = pickle.load(f)

            self.conflicting_pairs[(class_n, class_m)] = conflict_data
            print(f"Loaded conflict patterns from {filepath}")
            return conflict_data
        except FileNotFoundError:
            print(f"Conflict patterns file not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error loading conflict patterns: {e}")
            return None

    def generate_conflict_resolution_rules(self, class_n, class_m, confidence_threshold=0.7):
        """Generate rules to resolve conflicts between two classes"""
        if (class_n, class_m) not in self.conflicting_pairs:
            print(f"No conflict patterns available for {class_n} vs {class_m}")
            return None

        conflict_data = self.conflicting_pairs[(class_n, class_m)]
        analysis = conflict_data['analysis']

        # Check if analysis contains an error
        if analysis.get('error'):
            print(f"Skipping rule generation for {class_n} vs {class_m}: {analysis['error']}")
            return None

        rules = {
            'class_pair': (class_n, class_m),
            'rules': [],
            'confidence_threshold': confidence_threshold,
            'generation_date': pd.Timestamp.now().isoformat()
        }

        # Create rules based on feature analysis
        for feature_info in analysis['feature_overlaps']:
            if feature_info['overlap_score'] > confidence_threshold:
                # This feature strongly contributes to confusion
                threshold = (feature_info['n_to_m_mean'] + feature_info['m_to_n_mean']) / 2

                rule = {
                    'feature': feature_info['feature_index'],
                    'threshold': float(threshold),
                    'overlap_score': feature_info['overlap_score'],
                    'description': f"If feature {feature_info['feature_index']} is near {threshold:.3f}, be cautious of {class_n}↔{class_m} confusion"
                }
                rules['rules'].append(rule)

        # Save rules to JSON file
        rules_filename = f"resolution_rules_{class_n}_vs_{class_m}.json"
        rules_path = os.path.join(self.output_dir, rules_filename)

        try:
            with open(rules_path, 'w') as f:
                json.dump(rules, f, indent=2)
            print(f"Conflict resolution rules saved to {rules_path}")
        except Exception as e:
            print(f"Error saving rules: {e}")

        return rules

    def run_complete_analysis(self, top_k=5, min_pattern_count=20):
        """Run complete conflict pattern analysis"""
        print("=== COMPLETE CONFLICT PATTERN ANALYSIS ===")

        if not self.load_patterns():
            return

        self.build_confusion_matrix()

        # Find top confusing pairs
        confusing_pairs = self.find_top_confusing_pairs(min_count=min_pattern_count, top_k=top_k)

        # Extract and analyze patterns for each confusing pair
        for class_n, class_m, count in confusing_pairs:
            print(f"\n{'=' * 50}")
            print(f"Analyzing conflict: {class_n} ↔ {class_m} ({count} misclassifications)")
            print(f"{'=' * 50}")

            conflict_data = self.extract_conflict_patterns(class_n, class_m)

            if conflict_data:
                # Save patterns
                self.save_conflict_patterns(class_n, class_m)

                # Generate resolution rules (only if we have patterns in both directions)
                if len(conflict_data['n_to_m_patterns']) > 0 and len(conflict_data['m_to_n_patterns']) > 0:
                    self.generate_conflict_resolution_rules(class_n, class_m)
                else:
                    print(
                        f"Skipping rule generation for {class_n} vs {class_m}: Not enough patterns in both directions")

        print(f"\nAnalysis complete! Processed {len(confusing_pairs)} class pairs")

        # Save summary report
        self.save_summary_report(confusing_pairs)

    def save_summary_report(self, confusing_pairs):
        """Save analysis summary report"""
        summary = {
            'total_incorrect_patterns': len(self.patterns) if self.patterns is not None else 0,
            'analyzed_pairs': [],
            'timestamp': pd.Timestamp.now().isoformat()
        }

        for class_n, class_m, count in confusing_pairs:
            if (class_n, class_m) in self.conflicting_pairs:
                pair_data = {
                    'classes': (class_n, class_m),
                    'misclassification_count': count,
                    'extracted_patterns': self.conflicting_pairs[(class_n, class_m)]['total_patterns'],
                    'n_to_m_count': len(self.conflicting_pairs[(class_n, class_m)]['n_to_m_patterns']),
                    'm_to_n_count': len(self.conflicting_pairs[(class_n, class_m)]['m_to_n_patterns']),
                    'analysis_available': True
                }
                summary['analyzed_pairs'].append(pair_data)

        summary_path = os.path.join(self.output_dir, 'analysis_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Analysis summary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving summary: {e}")

    def validate_conflict_patterns(self, validation_patterns_file='validation_patterns.pkl'):
        """
        Second stage: Validate the conflict patterns using a separate validation dataset

        Args:
            validation_patterns_file: Path to validation patterns file
        """
        print("\n" + "=" * 60)
        print("SECOND STAGE: VALIDATING CONFLICT PATTERNS")
        print("=" * 60)

        # Load validation patterns
        try:
            with open(validation_patterns_file, 'rb') as f:
                validation_patterns = pickle.load(f)
            print(f"Loaded {len(validation_patterns)} validation patterns")
        except FileNotFoundError:
            print(f"Error: {validation_patterns_file} not found")
            return
        except Exception as e:
            print(f"Error loading validation patterns: {e}")
            return

        # Load all saved conflict patterns
        self.load_all_saved_conflicts()

        if not self.conflicting_pairs:
            print("No conflict patterns found. Please run analysis first.")
            return

        validation_results = {}

        # Validate each conflict pair
        for class_pair, conflict_data in self.conflicting_pairs.items():
            class_n, class_m = class_pair

            # Extract patterns from validation set for this class pair
            true_labels = validation_patterns[:, -2].astype(int)
            pred_labels = validation_patterns[:, -1].astype(int)

            # Patterns where n is misclassified as m
            n_to_m_mask = (true_labels == class_n) & (pred_labels == class_m)
            n_to_m_validation = validation_patterns[n_to_m_mask]

            # Patterns where m is misclassified as n
            m_to_n_mask = (true_labels == class_m) & (pred_labels == class_n)
            m_to_n_validation = validation_patterns[m_to_n_mask]

            # Calculate validation metrics
            total_validation = len(n_to_m_validation) + len(m_to_n_validation)
            training_count = conflict_data['total_patterns']

            # Calculate similarity between training and validation patterns
            similarity_score = self._calculate_pattern_similarity(
                conflict_data['n_to_m_patterns'],
                conflict_data['m_to_n_patterns'],
                n_to_m_validation,
                m_to_n_validation
            )

            validation_results[class_pair] = {
                'training_patterns': training_count,
                'validation_patterns': total_validation,
                'similarity_score': similarity_score,
                'n_to_m_validation': len(n_to_m_validation),
                'm_to_n_validation': len(m_to_n_validation)
            }

            print(f"Class pair {class_n} vs {class_m}: "
                  f"{training_count} training patterns, "
                  f"{total_validation} validation patterns, "
                  f"similarity: {similarity_score:.3f}")

        # Save validation results
        validation_path = os.path.join(self.output_dir, 'validation_results.json')
        try:
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            print(f"Validation results saved to {validation_path}")
        except Exception as e:
            print(f"Error saving validation results: {e}")

        return validation_results

    def _calculate_pattern_similarity(self, n_to_m_train, m_to_n_train, n_to_m_val, m_to_n_val):
        """Calculate similarity between training and validation patterns"""
        if len(n_to_m_train) == 0 or len(n_to_m_val) == 0 or len(m_to_n_train) == 0 or len(m_to_n_val) == 0:
            return 0.0

        # Calculate feature distribution similarity
        n_to_m_similarity = self._calculate_feature_similarity(
            n_to_m_train[:, :8], n_to_m_val[:, :8]
        )

        m_to_n_similarity = self._calculate_feature_similarity(
            m_to_n_train[:, :8], m_to_n_val[:, :8]
        )

        # Average the similarity scores
        return (n_to_m_similarity + m_to_n_similarity) / 2

    def _calculate_feature_similarity(self, train_features, val_features):
        """Calculate similarity between feature distributions"""
        if len(train_features) == 0 or len(val_features) == 0:
            return 0.0

        # Calculate mean and std for each feature
        train_means = np.mean(train_features, axis=0)
        train_stds = np.std(train_features, axis=0)
        val_means = np.mean(val_features, axis=0)
        val_stds = np.std(val_features, axis=0)

        # Calculate similarity score for each feature
        similarity_scores = []
        for i in range(len(train_means)):
            # Use Bhattacharyya distance to measure similarity between distributions
            mean_diff = train_means[i] - val_means[i]
            std_avg = (train_stds[i] + val_stds[i]) / 2
            bhattacharyya = 0.25 * (mean_diff ** 2) / (std_avg ** 2) + 0.5 * np.log(
                (std_avg ** 2) / (train_stds[i] * val_stds[i]))

            # Convert to similarity score (0-1)
            similarity = np.exp(-bhattacharyya)
            similarity_scores.append(similarity)

        return np.mean(similarity_scores)

    def load_all_saved_conflicts(self):
        """Load all saved conflict patterns from the output directory"""
        print("Loading all saved conflict patterns...")

        # Get all conflict pattern files
        pattern_files = [f for f in os.listdir(self.output_dir) if f.startswith('conflict_') and f.endswith('.pkl')]

        for pattern_file in pattern_files:
            # Extract class numbers from filename
            try:
                parts = pattern_file.split('_')
                class_n = int(parts[1])
                class_m = int(parts[3].split('.')[0])

                # Load the conflict patterns
                self.load_conflict_patterns(class_n, class_m, pattern_file)
            except (ValueError, IndexError) as e:
                print(f"Error parsing filename {pattern_file}: {e}")
                continue

        print(f"Loaded {len(self.conflicting_pairs)} conflict patterns")


# Example usage and main function
def main():
    """Main function to demonstrate conflict pattern analysis"""
    manager = ConflictPatternManager(
        patterns_file='incorrect_patterns.pkl',
        output_dir='conflict_patterns'
    )

    # Run complete analysis
    manager.run_complete_analysis(top_k=8, min_pattern_count=15)

    # Second stage: Validate the conflict patterns
    manager.validate_conflict_patterns(validation_patterns_file='validation_patterns.pkl')

    # Example: How to use stored patterns later
    print("\n" + "=" * 60)
    print("EXAMPLE: LOADING AND USING STORED CONFLICT PATTERNS")
    print("=" * 60)

    # You can later load specific conflict patterns when needed
    # conflict_data = manager.load_conflict_patterns(3, 8)
    # if conflict_data:
    #     print(f"Loaded {conflict_data['total_patterns']} patterns for 3 vs 8 conflict")
    #     # Use these patterns for specialized training or analysis


if __name__ == "__main__":
    main()