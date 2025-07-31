#!/usr/bin/env python3

import os
import sys

import argparse
import glob
import multiprocessing
import subprocess

from radarize.config import cfg, update_config

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        default="configs/default.yaml",
        type=str,
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=1,
        help="Number of processes to use for parallel processing.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def run_commands(cmds, n_proc):
    print(f"Running {len(cmds)} commands in parallel using {n_proc} processes")
    
    def run_with_output(cmd):
        proc = subprocess.run(cmd, check=True, capture_output=True)
        return proc
    
    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(run_with_output, cmds)
    
    return results


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)
    
    print(f"\nRunning with {args.n_proc} processes (out of {multiprocessing.cpu_count()} available cores)")

    # Preprocess datasets. (bag -> npz)
    print("Preprocessing datasets...")
    all_bag_paths = sorted(glob.glob(os.path.join(cfg["DATASET"]["PATH"], "*.bag")))
    print(f"Found {len(all_bag_paths)} .bag files in total")
    
    bag_paths = [x for x in all_bag_paths if not os.path.exists(x.replace(".bag", ".npz"))]
    skipped_paths = [x for x in all_bag_paths if os.path.exists(x.replace(".bag", ".npz"))]
    
    print(f"\nSkipping {len(skipped_paths)} already processed files:")
    for skip_file in skipped_paths:
        print(f"  - Skipping: {os.path.basename(skip_file)} (already has .npz)")
    
    print(f"\nProcessing {len(bag_paths)} new files:")
    for process_file in bag_paths:
        print(f"  - Will process: {os.path.basename(process_file)}")
    
    if bag_paths:  # Only run if there are files to process
        print(f"\nStarting conversion of .bag to .npz with {args.n_proc} parallel processes...")
        run_commands(
            [
                [f"tools/create_dataset.py", f"--cfg={args.cfg}", f"--bag_path={x}"]
                for x in bag_paths
            ],
            args.n_proc,
        )
    else:
        print("\nNo new files to process!")
    
    print("Dataset preprocessing completed.")

    print("Preparing training and testing datasets...")
    train_npz_paths = sorted(
        [
            os.path.join(cfg["DATASET"]["PATH"], os.path.basename(x) + ".npz")
            for x in cfg["DATASET"]["TRAIN_SPLIT"]
        ]
    )
    test_npz_paths = sorted(
        [
            os.path.join(cfg["DATASET"]["PATH"], os.path.basename(x) + ".npz")
            for x in cfg["DATASET"]["TEST_SPLIT"]
        ]
    )
    print("prepare training and testing datasets completed.")

    # Extract ground truth.
    print(f"Extracting ground truth with {args.n_proc} parallel processes...")
    run_commands(
        [
            ["tools/extract_gt.py", f"--cfg={args.cfg}", f"--npz_path={x}"]
            for x in test_npz_paths
        ],
        args.n_proc,
    )
    print("Ground truth extraction completed.")


    # Train flow models.
    print("Training flow models...")
    subprocess.run(["tools/train_flow.py", f"--cfg={args.cfg}", f"--n_proc={args.n_proc}"], check=True)
    subprocess.run(["tools/test_flow.py", f"--cfg={args.cfg}", f"--n_proc={args.n_proc}"], check=True)

    # Train rotnet models.
    print("Training rotnet models...")
    subprocess.run(["tools/train_rot.py", f"--cfg={args.cfg}", f"--n_proc={args.n_proc}"], check=True)
    subprocess.run(["tools/test_rot.py", f"--cfg={args.cfg}", f"--n_proc={args.n_proc}"], check=True)

    # Extract odometry.
    print(f"Extracting odometry with {args.n_proc} parallel processes...")
    run_commands(
        [
            ["tools/test_odom.py", f"--cfg={args.cfg}", f"--npz_path={x}"]
            for x in test_npz_paths
        ],
        args.n_proc,
    )

    # Train UNet
    print("Training UNet...")
    subprocess.run(["tools/train_unet.py", f"--cfg={args.cfg}", f"--n_proc={args.n_proc}"], check=True)
    print(f"Testing UNet with {args.n_proc} parallel processes...")
    run_commands(
        [
            ["tools/test_unet.py", f"--cfg={args.cfg}", f"--npz_path={x}"]
            for x in test_npz_paths
        ],
        args.n_proc,
    )

    ### Run Cartographer with multiple processes
    # Get ground truth.
    print("Running Cartographer with ground truth...")
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc={args.n_proc}",
            f"--odom=gt",
            f"--scan=gt",
            f"--params=default",
        ],
        check=True,
    )

    # RadarHD baseline.
    print("Running Cartographer with RadarHD baseline...")
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc={args.n_proc}",
            f"--odom=gt",
            f"--scan=radarhd",
            f"--params=scan_only",
        ],
        check=True,
    )

    # RNIN + RadarHD baseline.
    print("Running Cartographer with RNIN + RadarHD baseline...")
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc={args.n_proc}",
            f"--odom=rnin",
            f"--scan=radarhd",
            f"--params=default",
        ],
        check=True,
    )

    # milliEgo + RadarHD baseline.
    print("Running Cartographer with milliEgo + RadarHD baseline...")
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc={args.n_proc}",
            f"--odom=milliego",
            f"--scan=radarhd",
            f"--params=default",
        ],
        check=True,
    )

    # Our odometry + RadarHD baseline.
    print("Running Cartographer with our odometry + RadarHD baseline...")
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc={args.n_proc}",
            f"--odom=odometry",
            f"--scan=radarhd",
            f"--params=radar",
        ],
        check=True,
    )

    # Run radarize.
    print("Running Cartographer with Radarize...")
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc={args.n_proc}",
            f"--odom=odometry",
            f"--scan=unet",
            f"--params=radar",
        ],
        check=True,
    )