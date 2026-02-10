import pickle
import os
from collections import defaultdict


def load_clustered_patterns(digit):
    """Load clustered patterns for a specific digit"""
    filename = f"clustered_digit_{digit}.pkl"
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return None

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data['clustered_data']


def extract_zoned_pattern_tuples(clustered_data):
    """Extract pattern tuples with zone information"""
    zoned_patterns = defaultdict(set)  # zone -> set of patterns

    for zone, zone_data in clustered_data.items():
        for cluster in zone_data['clusters']:
            zoned_patterns[zone].add(cluster['pattern'])

    return zoned_patterns


def find_unique_patterns_with_zones(digit1, digit2):
    """Find patterns that are unique to digit1 compared to digit2, considering zones"""
    # Load clustered patterns for both digits
    clustered_data1 = load_clustered_patterns(digit1)
    clustered_data2 = load_clustered_patterns(digit2)

    if clustered_data1 is None or clustered_data2 is None:
        return {}

    # Extract patterns with zone information
    patterns1 = extract_zoned_pattern_tuples(clustered_data1)
    patterns2 = extract_zoned_pattern_tuples(clustered_data2)

    # Find unique patterns per zone
    unique_patterns_by_zone = {}

    for zone in patterns1.keys():
        if zone in patterns2:
            # Patterns unique to digit1 in this specific zone
            unique_patterns = patterns1[zone] - patterns2[zone]
        else:
            # If zone doesn't exist in digit2, all patterns are unique
            unique_patterns = patterns1[zone]

        unique_patterns_by_zone[zone] = unique_patterns

    return unique_patterns_by_zone


def save_unique_patterns_with_zones(digit1, digit2, unique_patterns_by_zone):
    """Save unique patterns with zone information to file"""
    filename = f"unique_patterns_digit_{digit1}_vs_{digit2}_with_zones.pkl"

    total_unique = sum(len(patterns) for patterns in unique_patterns_by_zone.values())

    with open(filename, 'wb') as f:
        pickle.dump({
            'digit1': digit1,
            'digit2': digit2,
            'unique_patterns_count': total_unique,
            'unique_patterns_by_zone': dict(unique_patterns_by_zone),
            'zone_summary': {zone: len(patterns) for zone, patterns in unique_patterns_by_zone.items()}
        }, f)

    return total_unique, unique_patterns_by_zone


def main():
    """Main function to perform cross-digit unique pattern search with zones"""
    print("Starting cross-digit unique pattern search with zones...")
    print("=" * 60)

    results = {}
    zone_results = defaultdict(lambda: defaultdict(int))
    detailed_results = {}

    # Compare each digit against all other digits
    for digit1 in range(10):
        for digit2 in range(10):
            if digit1 == digit2:
                continue  # Skip comparing digit to itself

            print(f"Comparing digit {digit1} vs digit {digit2}...")

            # Find unique patterns with zone information
            unique_patterns_by_zone = find_unique_patterns_with_zones(digit1, digit2)

            # Save results
            unique_count, patterns_data = save_unique_patterns_with_zones(digit1, digit2, unique_patterns_by_zone)

            # Store results
            if digit1 not in results:
                results[digit1] = {}
            results[digit1][digit2] = unique_count

            # Store detailed results for zone-level analysis
            detailed_results[(digit1, digit2)] = patterns_data

            # Accumulate zone-level statistics
            for zone, patterns in unique_patterns_by_zone.items():
                zone_results[digit1][zone] += len(patterns)

            print(f"  Found {unique_count} unique patterns across {len(unique_patterns_by_zone)} zones")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-DIGIT UNIQUE PATTERN SUMMARY (WITH ZONES)")
    print("=" * 60)

    # Overall summary
    for digit1 in range(10):
        print(f"\nDigit {digit1} unique patterns compared to other digits:")
        total_unique = 0

        for digit2 in range(10):
            if digit1 == digit2:
                continue

            unique_count = results[digit1][digit2]
            total_unique += unique_count
            print(f"  vs digit {digit2}: {unique_count} unique patterns")

        print(f"  Total unique patterns for digit {digit1}: {total_unique}")

    # Zone-level summary
    print("\n" + "=" * 60)
    print("ZONE-LEVEL SUMMARY")
    print("=" * 60)

    for digit in range(10):
        print(f"\nDigit {digit} - Unique patterns by zone:")
        for zone in sorted(zone_results[digit].keys()):
            count = zone_results[digit][zone]
            print(f"  Zone {zone}: {count} unique patterns")

    # Save comprehensive results
    with open("comprehensive_unique_patterns_summary.pkl", 'wb') as f:
        pickle.dump({
            'overall_results': dict(results),
            'zone_results': dict(zone_results),
            'detailed_results': detailed_results
        }, f)

    print("\nSearch completed. Results saved to:")
    print("- 'unique_patterns_digit_X_vs_Y_with_zones.pkl' (individual comparisons)")
    print("- 'comprehensive_unique_patterns_summary.pkl' (complete summary)")


if __name__ == "__main__":
    main()