import pstats
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_profile.py <stats_file> [filter_string]")
        print("Example: python scripts/analyze_profile.py profile.stats tropt")
        return

    stats_file = sys.argv[1]
    filter_str = sys.argv[2] if len(sys.argv) > 2 else ""
    
    if not os.path.exists(stats_file):
        print(f"Error: File '{stats_file}' not found.")
        return

    p = pstats.Stats(stats_file)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS OF: {stats_file}")
    if filter_str:
        print(f"FILTERING BY: '{filter_str}'")
    print(f"{'='*60}")
    
    # Strip dirs helps readability, but we need full paths if we want to filter by them effectively
    # explicitly, usually. However, pstats filtering works on the full filename even if stripped?
    # Actually p.strip_dirs() modifies the object in place and removes directory info forever.
    # We should NOT strip dirs if we want to see where files are, OR we trust the filter.
    # Usually for "repository level" we want to see 'tropt/optimizer/...' vs 'site-packages/...'.
    # So we probably shouldn't strip dirs if we want to distinguish, 
    # BUT pstats output is ugly with full paths.
    # Compromise: We filter first, then maybe print? 
    # pstats doesn't let you filter then strip easily in one chain if strip is destructive.
    # Let's just use the filter arg of print_stats.

    print("\n--- TOP 20 FUNCTIONS BY CUMULATIVE TIME ---")
    print("(Includes time spent in sub-calls)")
    p.sort_stats("cumtime").print_stats(filter_str, 20)
    
    print("\n--- TOP 20 FUNCTIONS BY INTERNAL TIME ---")
    print("(Excludes time spent in sub-calls - finding the 'heavy lifters')")
    p.sort_stats("tottime").print_stats(filter_str, 20)

if __name__ == "__main__":
    main()

