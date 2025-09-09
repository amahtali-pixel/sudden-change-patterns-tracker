import torch
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import os
import pickle
import time
import json
from collections import defaultdict

"""
SYSTÈME INNOVANT DE RECONNAISSANCE DE CHIFFRES MANUSCRITS
PAR EXTRACTION DE MOTIFS ROTATIFS ET VALIDATION TRIPLE

Copyright (c) 2024 [Votre Nom]. Tous droits réservés.

INVENTIONS PROTÉGÉES :
- Extraction de motifs rotatifs (Rotary Pattern Extraction)
- Validation en trois étapes (Three-Stage Validation) 
- Matching zoné avec grille 4x4 (Zone-Based Matching)
- Détection d'arêtes optimisée GPU (GPU Edge Detection)

Protégé sous le droit d'auteur français (Code de la propriété intellectuelle)
et les conventions internationales (Convention de Berne).
"""




# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Color codes for better output visualization
YELLOW = '\033[93m'
PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'


class ThreeStageValidator:
    def __init__(self):
        # Load pixel density statistics
        self.pixel_stats = self.load_pixel_statistics()
        self.unique_patterns = self.load_unique_patterns()

    def load_pixel_statistics(self):
        """Load pixel density statistics"""
        try:
            with open('pixel_density_statistics.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"{RED}Warning: pixel_density_statistics.json not found{ENDC}")
            return None

    def load_unique_patterns(self):
        """Load unique patterns for tie-breaking"""
        try:
            with open('unique_patterns.pkl', 'rb') as f:
                unique_patterns = pickle.load(f)
                # Organize by class
                unique_by_class = {}
                for pattern in unique_patterns:
                    class_id = int(pattern[-1])
                    if class_id not in unique_by_class:
                        unique_by_class[class_id] = []
                    unique_by_class[class_id].append(pattern[:-1])
                return unique_by_class
        except FileNotFoundError:
            print(f"{RED}Warning: unique_patterns.pkl not found{ENDC}")
            return None

    def count_active_pixels(self, image):
        """Count non-zero pixels in an image"""
        return torch.count_nonzero(image).item()

    def stage2_unique_patterns(self, image_patterns, tied_classes):
        """Stage 2: Unique pattern matching"""
        if self.unique_patterns is None:
            return tied_classes

        unique_matches = {}
        for class_id in tied_classes:
            if class_id in self.unique_patterns:
                ref_patterns = self.unique_patterns[class_id]
                # Simple matching - count close patterns
                matches = 0
                for img_pattern in image_patterns:
                    distances = np.linalg.norm(ref_patterns - img_pattern[:8], axis=1)
                    matches += np.sum(distances < 0.1)
                unique_matches[class_id] = matches
            else:
                unique_matches[class_id] = 0

        if not unique_matches:
            return tied_classes

        max_matches = max(unique_matches.values())
        return [cls for cls, count in unique_matches.items() if count == max_matches]

    def stage3_pixel_density(self, tied_classes, image_pixel_count):
        """Stage 3: Pixel density validation"""
        if self.pixel_stats is None:
            return tied_classes[0] if tied_classes else None

        best_match = None
        smallest_diff = float('inf')

        for class_id in tied_classes:
            class_str = str(class_id)
            if class_str in self.pixel_stats['density_profiles']:
                stats = self.pixel_stats['density_profiles'][class_str]
                avg_pixels = stats['avg_pixels']
                diff = abs(avg_pixels - image_pixel_count)

                if diff < smallest_diff:
                    smallest_diff = diff
                    best_match = class_id

        return best_match


class PatternMatcher:
    def __init__(self, device='cuda', tolerance=0.298, batch_size=32):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.ref_arrays = {}
        self.ref_zones = {}
        self.stats = {'total_processed': 0, 'correct_predictions': 0}
        self.validator = ThreeStageValidator()

        # New: Collections for patterns that cause correct and incorrect predictions
        self.correct_patterns = []  # (pattern, true_label)
        self.incorrect_patterns = []  # (pattern, predicted_label, true_label)

        # Precompute constants
        self.shifts = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                                    [0, 1], [1, -1], [1, 0], [1, 1]],
                                   dtype=torch.long)

        print(f"{GREEN}PatternMatcher initialized on device: {self.device}{ENDC}")

    def fast_edge_detection(self, batch):
        """Detect edges in images using optimized convolution"""
        batch = batch.to(self.device)

        if batch.dim() == 3:
            batch = batch.unsqueeze(1)

        padded = F.pad(batch, (1, 1, 1, 1), mode='constant', value=0)

        # Create and configure kernel
        kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]],
                              dtype=torch.float32, device=self.device)
        kernel = kernel.repeat(batch.size(1), 1, 1, 1)

        neighbor_counts = F.conv2d(padded, kernel)
        edges = ((batch > 0) & (neighbor_counts < 8) & (neighbor_counts > 0))
        return edges.squeeze(1) if edges.size(1) == 1 else edges

    def extract_pattern_features_batch(self, images_batch):
        """Extract patterns from a batch of images"""
        batch_size = images_batch.size(0)
        images_batch = images_batch.to(self.device)

        edges = self.fast_edge_detection(images_batch)
        all_features = []

        for i in range(batch_size):
            image_edges = edges[i]
            edge_mask = image_edges > 0

            if not torch.any(edge_mask):
                continue

            edge_coords = torch.stack(torch.where(edge_mask), dim=1)
            padded_img = F.pad(images_batch[i].unsqueeze(0), (1, 1, 1, 1),
                               mode='constant', value=0).squeeze(0)

            # Extract neighbor values
            y_coords = edge_coords[:, 0].unsqueeze(1) + self.shifts[:, 0].to(self.device) + 1
            x_coords = edge_coords[:, 1].unsqueeze(1) + self.shifts[:, 1].to(self.device) + 1

            y_coords = torch.clamp(y_coords, 0, padded_img.size(0) - 1)
            x_coords = torch.clamp(x_coords, 0, padded_img.size(1) - 1)

            neighbor_values = padded_img[y_coords, x_coords]
            rotary_patterns = neighbor_values + torch.roll(neighbor_values, shifts=1, dims=1)

            coordinates = edge_coords[:, :2].float()
            batch_indices = torch.full((len(edge_coords), 1), i,
                                       dtype=torch.float32, device=self.device)

            features = torch.cat([rotary_patterns, coordinates, batch_indices], dim=1)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy() if all_features else np.array([])

    def load_reference_patterns(self):
        """Load reference patterns from class files"""
        self.ref_arrays = {}
        self.ref_zones = {}

        for class_digit in range(10):
            filename = f"clustered_{class_digit}.pkl"
            try:
                with open(filename, 'rb') as f:
                    patterns = pickle.load(f)

                if isinstance(patterns, np.ndarray) and patterns.size > 0:
                    self.ref_arrays[class_digit] = patterns
                    self.ref_zones[class_digit] = self.get_zone_id_batch(patterns[:, 8:10])
                elif isinstance(patterns, list) and len(patterns) > 0:
                    ref_array = np.array(patterns)
                    self.ref_arrays[class_digit] = ref_array
                    self.ref_zones[class_digit] = self.get_zone_id_batch(ref_array[:, 8:10])
                else:
                    self.ref_arrays[class_digit] = None
                    self.ref_zones[class_digit] = None

                count = len(patterns) if hasattr(patterns, '__len__') else 0
                print(f"{GREEN}Loaded {count} patterns for class {class_digit}{ENDC}")

            except FileNotFoundError:
                print(f"{RED}Warning: {filename} not found{ENDC}")
                self.ref_arrays[class_digit] = None
                self.ref_zones[class_digit] = None

    def get_zone_id_batch(self, coords, image_size=28, grid_size=4):
        """Calculate zone IDs for coordinates"""
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()

        y, x = coords[:, 0], coords[:, 1]
        zone_size = image_size / grid_size

        zone_y = (y / zone_size).astype(int)
        zone_x = (x / zone_size).astype(int)

        return zone_y * grid_size + zone_x

    def count_pattern_matches(self, new_patterns):
        """Count pattern matches with reference classes"""
        match_counts = {i: 0 for i in range(10)}

        if len(new_patterns) == 0:
            return match_counts

        new_coords = new_patterns[:, 8:10]
        new_zones = self.get_zone_id_batch(new_coords)
        new_rotary = new_patterns[:, :8]
        unique_zones = np.unique(new_zones)

        for class_digit in range(10):
            if self.ref_arrays[class_digit] is None or len(self.ref_arrays[class_digit]) == 0:
                continue

            ref_rotary = self.ref_arrays[class_digit][:, :8]
            ref_zones_class = self.ref_zones[class_digit]

            for zone in unique_zones:
                new_in_zone = new_zones == zone
                if not np.any(new_in_zone):
                    continue

                ref_in_zone = ref_zones_class == zone
                if not np.any(ref_in_zone):
                    continue

                new_zone_patterns = new_rotary[new_in_zone]
                ref_zone_patterns = ref_rotary[ref_in_zone]

                # Vectorized distance calculation
                diff = new_zone_patterns[:, np.newaxis, :] - ref_zone_patterns[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                min_distances = np.min(distances, axis=1)

                match_counts[class_digit] += np.sum(min_distances <= self.tolerance)

        return match_counts

    def three_stage_prediction(self, match_counts, image_patterns, image_data, true_label):
        """Three-stage prediction with tie-breaking"""
        # Stage 1: Regular pattern matching
        max_matches = max(match_counts.values())
        top_classes = [cls for cls, count in match_counts.items() if count == max_matches]

        if len(top_classes) == 1:
            # Clear winner from stage 1
            confidence = self.calculate_confidence(match_counts, top_classes[0])
            return top_classes[0], confidence

        print(f"{YELLOW}⚡ Stage 1 Tie: {top_classes}{ENDC}")

        # Stage 2: Unique pattern matching
        stage2_classes = self.validator.stage2_unique_patterns(image_patterns, top_classes)

        if len(stage2_classes) == 1:
            # Resolved by unique patterns
            confidence = 0.7  # Medium confidence for stage 2 resolution
            print(f"{GREEN}✅ Stage 2 resolved: {stage2_classes[0]}{ENDC}")
            return stage2_classes[0], confidence

        print(f"{YELLOW}⚡ Stage 2 Tie: {stage2_classes}{ENDC}")

        # Stage 3: Pixel density validation
        image_pixels = self.validator.count_active_pixels(image_data)
        final_prediction = self.validator.stage3_pixel_density(stage2_classes, image_pixels)

        if final_prediction is not None:
            confidence = 0.6  # Lower confidence for stage 3 resolution
            print(f"{GREEN}✅ Stage 3 resolved: {final_prediction} (Pixels: {image_pixels}){ENDC}")
            return final_prediction, confidence

        # Fallback: return first class if all stages fail
        return top_classes[0], 0.5

    def calculate_confidence(self, match_counts, predicted_class):
        """Calculate confidence score based on match distribution"""
        total_matches = sum(match_counts.values())
        if total_matches == 0:
            return 0.0

        predicted_matches = match_counts[predicted_class]

        # Get second highest matches
        other_matches = [count for cls, count in match_counts.items() if cls != predicted_class]
        second_best = max(other_matches) if other_matches else 0

        # Confidence based on margin
        confidence = (predicted_matches - second_best) / total_matches
        return max(0.0, min(1.0, confidence))

    def analyze_results(self, match_counts, true_label, image_patterns, image_data):
        """Analyze and display match results with three-stage validation"""
        total_matches = sum(match_counts.values())

        if total_matches == 0:
            print(f"{YELLOW}No pattern matches found{ENDC}")
            return None, None

        print(f"{PINK}Total matches: {total_matches}{ENDC}")
        print(f"{PINK}Match distribution:{ENDC}")

        sorted_classes = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

        for digit, count in sorted_classes:
            percentage = (count / total_matches * 100) if total_matches > 0 else 0
            print(f"  Class {digit}: {count:>6} matches ({percentage:.1f}%)")

        # Three-stage prediction
        predicted_class, confidence = self.three_stage_prediction(
            match_counts, image_patterns, image_data, true_label
        )

        print(f"\n{PINK}Predicted: {predicted_class}, Confidence: {confidence:.3f}{ENDC}")

        if true_label is not None:
            correct = predicted_class == true_label
            color = GREEN if correct else RED
            status = "CORRECT" if correct else "INCORRECT"
            print(f"{color}True label: {true_label} ({status}){ENDC}")

            # Update statistics
            self.stats['total_processed'] += 1
            if correct:
                self.stats['correct_predictions'] += 1

            # Show instant accuracy
            if self.stats['total_processed'] > 0:
                accuracy = self.stats['correct_predictions'] / self.stats['total_processed']
                print(f"{BLUE}Instant Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%) - "
                      f"{self.stats['correct_predictions']}/{self.stats['total_processed']}{ENDC}")

        return predicted_class, confidence

    def collect_patterns_for_analysis(self, image_patterns, true_label, predicted_label):
        """Collect patterns for later analysis of correct and incorrect predictions"""
        if image_patterns is None or len(image_patterns) == 0:
            return

        # Add the true label to each pattern
        patterns_with_label = np.hstack([
            image_patterns,
            np.full((len(image_patterns), 1), true_label)
        ])

        # Add the predicted label to each pattern
        patterns_with_both_labels = np.hstack([
            patterns_with_label,
            np.full((len(image_patterns), 1), predicted_label)
        ])

        # Store patterns based on prediction correctness
        if predicted_label == true_label:
            # Correct prediction - store pattern with true label
            self.correct_patterns.extend(patterns_with_both_labels)
        else:
            # Incorrect prediction - store pattern with both predicted and true labels
            self.incorrect_patterns.extend(patterns_with_both_labels)

    def save_pattern_analysis(self):
        """Save the collected patterns for correct and incorrect predictions"""
        if self.correct_patterns:
            with open('correct_patterns.pkl', 'wb') as f:
                pickle.dump(np.array(self.correct_patterns), f)
            print(f"{GREEN}Saved {len(self.correct_patterns)} patterns from correct predictions{ENDC}")

        if self.incorrect_patterns:
            with open('incorrect_patterns.pkl', 'wb') as f:
                pickle.dump(np.array(self.incorrect_patterns), f)
            print(f"{GREEN}Saved {len(self.incorrect_patterns)} patterns from incorrect predictions{ENDC}")

    def process_single_image(self, image, true_label, image_idx):
        """Process a single image with three-stage validation"""
        print(f"\n{YELLOW}=== Processing Image {image_idx} (True class: {true_label}) ==={ENDC}")

        # Add batch dimension and process
        image_batch = image.unsqueeze(0)
        patterns_batch = self.extract_pattern_features_batch(image_batch)

        if len(patterns_batch) == 0:
            print(f"{YELLOW}No patterns found{ENDC}")
            return {
                'image_idx': image_idx,
                'true_label': true_label,
                'predicted_class': None,
                'confidence': 0,
                'match_counts': {i: 0 for i in range(10)},
                'correct': False
            }

        print(f"{GREEN}Extracted {len(patterns_batch)} patterns{ENDC}")

        # Count matches and analyze with three-stage validation
        match_counts = self.count_pattern_matches(patterns_batch)
        predicted_class, confidence = self.analyze_results(
            match_counts, true_label, patterns_batch, image
        )

        # Check if prediction is correct
        is_correct = predicted_class == true_label if predicted_class is not None else False

        # Collect patterns for analysis if we've processed less than 1000 images
        if self.stats['total_processed'] <= 1000 and predicted_class is not None:
            self.collect_patterns_for_analysis(patterns_batch, true_label, predicted_class)

        return {
            'image_idx': image_idx,
            'true_label': true_label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'match_counts': match_counts,
            'correct': is_correct,
            'total_patterns': len(patterns_batch)
        }

    def run(self, num_images=9900):
        """Main execution method"""
        print(f"{BLUE}Loading test images...{ENDC}")
        test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        test_images = test_dataset.data[:num_images].float() / 255.0
        test_labels = test_dataset.targets[:num_images]

        print(f"{BLUE}Loading reference patterns...{ENDC}")
        self.load_reference_patterns()

        all_results = []
        start_time = time.time()

        # Process images
        for i in range(len(test_images)):
            result = self.process_single_image(test_images[i], test_labels[i].item(), i)
            all_results.append(result)

            # Show progress every 100 images
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"{BLUE}Processed {i + 1}/{len(test_images)} images "
                      f"({elapsed:.2f} seconds elapsed){ENDC}")

            # Save pattern analysis after processing 1000 images
            if self.stats['total_processed'] == 1000:
                self.save_pattern_analysis()

        # Final results
        self.print_final_results(all_results, start_time)
        return all_results

    def print_final_results(self, all_results, start_time):
        """Print final accuracy and statistics"""
        elapsed = time.time() - start_time

        print(f"\n{PINK}=== PROCESSING COMPLETED ==={ENDC}")
        print(f"{PINK}Time: {elapsed:.2f} seconds{ENDC}")
        print(f"{PINK}Images processed: {self.stats['total_processed']}{ENDC}")

        if self.stats['total_processed'] > 0:
            accuracy = self.stats['correct_predictions'] / self.stats['total_processed']
            print(f"{PINK}Final Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%){ENDC}")
            print(f"{PINK}Correct predictions: {self.stats['correct_predictions']}/"
                  f"{self.stats['total_processed']}{ENDC}")

            # Per-class accuracy
            print(f"\n{PINK}=== PER-CLASS ACCURACY ==={ENDC}")
            class_stats = {i: {'correct': 0, 'total': 0} for i in range(10)}

            for result in all_results:
                if result['predicted_class'] is not None:
                    true_class = result['true_label']
                    class_stats[true_class]['total'] += 1
                    if result['correct']:
                        class_stats[true_class]['correct'] += 1

            for digit in range(10):
                correct = class_stats[digit]['correct']
                total = class_stats[digit]['total']
                if total > 0:
                    acc = correct / total
                    print(f"  Class {digit}: {correct}/{total} ({acc:.3f})")

        # Save results
        with open('pattern_match_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print(f"{GREEN}Results saved to pattern_match_results.pkl{ENDC}")


def main():
    """Main function"""
    # Initialize pattern matcher
    matcher = PatternMatcher(device='cuda', tolerance=0.298, batch_size=32)

    # Run the pattern matching
    results = matcher.run(num_images=9900)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
