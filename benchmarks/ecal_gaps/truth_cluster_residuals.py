"""ECAL reconstructed-minus-TC residuals; shared hits correlate the estimators.

The ecal_gaps workflow runs this analysis for all three ECALs. For MCParticles
index zero, it requires exactly one associated truth cluster and selects the
nearest electron-associated standard cluster on the subsystem surface.
Energy residuals are (E_reco-E_TC)/E_TC; endcap position residuals are
reconstructed minus TC x/y. Barrel residuals are TC radius times wrapped
reco-minus-TC azimuth and reco-minus-TC z, with matching in these two surface
coordinates rather than endcap x/y.

Default matching cuts are 50 mm backward/barrel and 200 mm forward when
analyzing all subsystems. These detector-specific cuts are provisional
matching-quality selections, not definitions of intrinsic resolution. The
forward choice is motivated by the 5 GeV generated-eta cut scan, where remaining
distance failures are concentrated toward the transition region; 500 MeV
results require separate interpretation.

Central-68% half-widths and all exclusion counts are saved. Shared hits and
different log-weight bases make these agreement widths, not independent
detector resolutions. The matching fraction includes tiny incidental TC
deposits and is not an electron efficiency. The barrel truth-guided merger is
not an independent hit-level truth clustering algorithm, so correlations can
produce very narrow or exactly zero residuals. Results are labelled separately.

Configure matching cuts with Snakemake, for example:
    --config ecal_gaps_match_radius_mm=30 ecal_gaps_forward_match_radius_mm=150
or with --match-radius-mm and --forward-match-radius-mm when analyzing all
subsystems. With --inputs, --match-radius-mm sets the cut for the selected
single subsystem (default 50 mm). Changing cuts only requires analysis to be
rerun, not simulation or reconstruction.
"""
import argparse
import glob
import json
from pathlib import Path

import awkward as ak
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def links(event, name, particle, count):
    rec = ak.to_list(event[f"_{name}_rec.index"])
    sim = ak.to_list(event[f"_{name}_sim.index"])
    if len(rec) != len(sim):
        raise ValueError("Association arrays differ in length")
    indices = sorted({r for r, s in zip(rec, sim) if s == particle})
    if any(i < 0 or i >= count for i in indices):
        raise ValueError("Invalid cluster association index")
    return indices


def position_residual(reco, truth, barrel=False):
    if not barrel:
        return np.asarray(reco[:2]) - np.asarray(truth[:2])
    dphi = np.arctan2(reco[1], reco[0]) - np.arctan2(truth[1], truth[0])
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.asarray([np.hypot(*truth[:2]) * dphi, reco[2] - truth[2]])


def analyze(paths, radius, subsystem):
    reco, truth = f"{subsystem}Clusters", f"{subsystem}TruthClusters"
    counts = dict(events=0, zero_truth=0, multiple_truth=0, unique_truth=0,
                  no_associated_reco=0, outside_radius=0, nonpositive_truth_energy=0, matched=0)
    rows = []
    for events in uproot.iterate({path: "events" for path in paths}, filter_name=[
        f"{reco}.*", f"{truth}.*", f"_{subsystem}ClusterAssociations*",
        f"_{subsystem}TruthClusterAssociations*"], step_size="100 MB"):
        for event in events:
            counts["events"] += 1
            ti = links(event, f"{subsystem}TruthClusterAssociations", 0, len(event[f"{truth}.energy"]))
            if len(ti) != 1:
                counts["zero_truth" if not ti else "multiple_truth"] += 1
                continue
            counts["unique_truth"] += 1
            ri = links(event, f"{subsystem}ClusterAssociations", 0, len(event[f"{reco}.energy"]))
            if not ri:
                counts["no_associated_reco"] += 1
                continue
            t = ti[0]
            tp = [float(event[f"{truth}.position.{c}"][t]) for c in "xyz"]
            delta = np.asarray([position_residual(
                [float(event[f"{reco}.position.{c}"][r]) for c in "xyz"],
                tp, subsystem == "EcalBarrel") for r in ri])
            distances = np.linalg.norm(delta, axis=1)
            best = int(np.argmin(distances))
            if distances[best] >= radius:
                counts["outside_radius"] += 1
                continue
            et = float(event[f"{truth}.energy"][t])
            if et <= 0:
                counts["nonpositive_truth_energy"] += 1
                continue
            er = float(event[f"{reco}.energy"][ri[best]])
            rows.append([(er-et)/et, *delta[best], distances[best]])
            counts["matched"] += 1
    return np.asarray(rows).reshape((-1, 4)), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, help="500MeV, 5GeV, 20GeV files (single subsystem)")
    parser.add_argument("--detector-config", help="Discover all nine benchmark inputs")
    parser.add_argument("--subsystem", choices=["EcalEndcapN", "EcalBarrel", "EcalEndcapP"], default="EcalEndcapN")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-radius-mm", type=float, default=50,
                        help="Backward/barrel cut, and single-subsystem cut [mm] (default: 50)")
    parser.add_argument("--forward-match-radius-mm", type=float, default=200,
                        help="Forward cut when analyzing all subsystems [mm] (default: 200)")
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.detector_config):
        parser.error("Supply exactly one of --inputs and --detector-config")
    if args.detector_config:
        # Same implementation, separately labelled geometry-aware results.
        for subsystem in ["EcalEndcapN", "EcalBarrel", "EcalEndcapP"]:
            paths = []
            for energy in ["500MeV", "5GeV", "20GeV"]:
                files = sorted(glob.glob(f"sim_output/ecal_gaps/{args.detector_config}/e-/{energy}/*/*.eicrecon.edm4eic.root"))
                if len(files) != 3:
                    raise RuntimeError(f"Expected three angular samples for {energy}, got {len(files)}")
                paths.append(files)
            radius = args.forward_match_radius_mm if subsystem == "EcalEndcapP" else args.match_radius_mm
            run_subsystem(paths, subsystem, args.output_dir, radius)
        return
    run_subsystem([[p] for p in args.inputs], args.subsystem, args.output_dir, args.match_radius_mm)


def run_subsystem(inputs, subsystem, output_dir, radius):
    from types import SimpleNamespace
    args = SimpleNamespace(output_dir=output_dir, match_radius_mm=radius)
    coordinates = ["R TC * delta phi", "delta z"] if subsystem == "EcalBarrel" else ["delta x", "delta y"]
    args.output_dir = args.output_dir / subsystem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    widths = []
    for label, paths in zip(["500MeV", "5GeV", "20GeV"], inputs):
        rows, counts = analyze(paths, args.match_radius_mm, subsystem)
        np.savez_compressed(args.output_dir / f"tc_residuals_{label}.npz", residuals=rows)
        width = [float(np.diff(np.quantile(rows[:,i], [.16,.84]))[0]/2)
                 if len(rows) >= 20 else None for i in range(3)]
        widths.append([np.nan if x is None else x for x in width])
        summary[label] = {"counts":counts, "central_68_half_width_energy_position":width,
                          "match_fraction_of_unique_truth": counts["matched"]/counts["unique_truth"]
                          if counts["unique_truth"] else None}
        fig, axes = plt.subplots(1,3,figsize=(12,3.5))
        for i, axis in enumerate(axes):
            axis.hist(rows[:,i],bins=100,histtype="step")
            axis.set_xlabel(["(E reco - E TC) / E TC", coordinates[0]+" [mm]", coordinates[1]+" [mm]"][i])
            axis.set_ylabel("Matched events")
        fig.suptitle(f"{subsystem} {label}: reconstructed/TC agreement")
        fig.tight_layout(); fig.savefig(args.output_dir/f"tc_residuals_{label}.png",dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4), layout="constrained"); widths=np.asarray(widths)
    axes[0].plot([.5,5,20],100*widths[:,0],"o-"); axes[0].set_ylabel("Energy residual central 68% half-width [%]")
    for i,c in enumerate(coordinates): axes[1].plot([.5,5,20],widths[:,i+1],"o-",label=c)
    axes[1].set_ylabel("Position residual central 68% half-width [mm]"); axes[1].legend()
    for axis in axes: axis.set_xlabel("Thrown electron energy [GeV]")
    fig.savefig(args.output_dir/"tc_agreement_widths.png",dpi=150); plt.close(fig)
    with (args.output_dir/"tc_residuals_summary.json").open("w") as f:
        json.dump({"match_radius_mm":args.match_radius_mm,"particle_index":0,
                   "subsystem":subsystem, "position_coordinates":coordinates,
                   "convention":"unique associated TC, nearest associated reco on subsystem surface; shared hits correlate quantities",
                   "results":summary},f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
