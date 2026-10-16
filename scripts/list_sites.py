"""Print site names from sites.yaml, one for each line, for the shell scripts.

Usage:
    python scripts/list_sites.py                  # the active sites
    python scripts/list_sites.py --all            # all the sites
    python scripts/list_sites.py --batch batch_2  # the sites of one batch
"""

import argparse

from cfm_config import batches, sites

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="All the sites, also the sites with 'active: false'")
    group.add_argument("--batch", type=str, help="A batch name in sites.yaml")
    args = parser.parse_args()

    if args.batch:
        all_batches = batches()
        if args.batch not in all_batches:
            parser.error(f"Batch {args.batch!r} is not in sites.yaml. Known batches: {', '.join(all_batches)}")
        names = all_batches[args.batch]
    else:
        names = list(sites(active_only=not args.all))
    print("\n".join(names))
