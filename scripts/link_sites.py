"""Make the site links: {CFM_data_path}/sites/<site>/<rcm> -> ../../runs/<model point> (FEAT-001).

The links give a view of the runs for each site. Two sites in one grid cell link to the same model point folder. The
script makes a link only if the model point has a runs/ folder. It deletes each link that is not correct, and makes it
again. It does not delete or change a file that is not a symbolic link.

Do not make or change a link by hand. The links are correct only because the name and the coordinates of a site do not
change (FEAT-002).

Usage:
    python scripts/link_sites.py              # all the sites in sites.yaml, for each RCM that has runs
    python scripts/link_sites.py --dry-run    # print the changes only
    python scripts/link_sites.py --rcm RACMO  # only these RCMs
"""

import argparse
import logging
import os
from typing import Dict, Iterable, List

from cfm_config import config, sites
from sim.manifest import runs_root

logging.basicConfig(level=logging.INFO, format="%(message)s")


def sites_root() -> str:
    return os.path.join(config["CFM_data_path"], "sites")


def rcm_of_tag(tag: str) -> str:
    """The RCM of a ModelPoint.tag: RACMO23_101_222 -> RACMO23."""
    return tag.rsplit("_", 2)[0]


def rcms_with_runs(runs: str) -> List[str]:
    """The RCMs that have a model point folder in runs/."""
    if not os.path.isdir(runs):
        return []
    return sorted({rcm_of_tag(tag) for tag in os.listdir(runs) if os.path.isdir(os.path.join(runs, tag))})


def expected_links(site_tags: Dict[str, Dict[str, str]], runs: str, links: str) -> Dict[str, str]:
    """{link path: link target} for {site: {rcm: model point tag}}, only for model points with a runs/ folder."""
    expected = {}
    for site, tags in site_tags.items():
        for rcm, tag in tags.items():
            if os.path.isdir(os.path.join(runs, tag)):
                link = os.path.join(links, site, rcm)
                expected[link] = os.path.relpath(os.path.join(runs, tag), os.path.dirname(link))
    return expected


def sync_links(expected: Dict[str, str], links: str, rcms: Iterable[str], dry_run: bool = False) -> List[str]:
    """Make the links in expected, and delete the other links for these rcms in links/. Returns the changes.

    Stops with an error if a path in links/ is not a symbolic link, as the script must not delete data.
    """
    rcms = set(rcms)
    expected = {path: target for path, target in expected.items() if os.path.basename(path) in rcms}
    changes = []
    if os.path.isdir(links):
        for site in sorted(os.listdir(links)):
            for rcm in sorted(os.listdir(os.path.join(links, site))):
                path = os.path.join(links, site, rcm)
                if not os.path.islink(path):
                    raise RuntimeError(f"{path} is not a symbolic link. Move it out of {links} first.")
                if rcm not in rcms:
                    continue
                if expected.get(path) != os.readlink(path):
                    changes.append(f"delete {path} -> {os.readlink(path)}")
                    if not dry_run:
                        os.remove(path)
    for path, target in sorted(expected.items()):
        if os.path.islink(path) and os.readlink(path) == target:
            continue
        changes.append(f"link {path} -> {target}")
        if not dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.symlink(target, path)
    if not dry_run and os.path.isdir(links):
        for site in os.listdir(links):
            if not os.listdir(os.path.join(links, site)):
                os.rmdir(os.path.join(links, site))
    return changes


if __name__ == "__main__":
    from force.model_point import model_point

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rcm", nargs="+", help="RCMs to link (default: each RCM that has runs)")
    parser.add_argument("--dry-run", action="store_true", help="Print the changes only")
    args = parser.parse_args()

    runs, links = runs_root(), sites_root()
    rcms = args.rcm or rcms_with_runs(runs)
    if not rcms:
        logging.info(f"No runs in {runs}. Nothing to link.")
    # all the sites, also the inactive ones: their runs exist
    site_tags = {name: {rcm: model_point(site, rcm).tag for rcm in rcms} for name, site in sites(active_only=False).items()}
    changes = sync_links(expected_links(site_tags, runs, links), links, rcms, dry_run=args.dry_run)
    for change in changes:
        logging.info(("(dry run) " if args.dry_run else "") + change)
    logging.info(f"{len(changes)} changes in {links}.")
