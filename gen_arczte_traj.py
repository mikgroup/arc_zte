import argparse
import numpy as np

from arc_zte_sim.metrics import percentage_TRs_with_refocusing_metric, cov_uniformity_metric
from arc_zte_sim.theta_i_schemes import FurthestDist_CostFunction 
from arc_zte_sim.rotate_spokes import save_Rs_txt
from arc_zte_sim.rotate_segments import read_seg_rotmats_from_txt

GOLDEN_ANGLE_3D_ROTS = './rot_txt_files/seg_golden3d_rotMats.txt'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate trajectory for Arc-ZTE segment using optimization scheme"
    )
    # Required arguments
    parser.add_argument(
        "--arc_angle", type=float, required=True, help="Arc angle (deg)"
    )
    parser.add_argument(
        "--spokes_per_seg", type=int, required=True, help="Number of spokes in segment"
    )
    parser.add_argument(
        "--num_segs", type=int, required=True, help="Number of golden-angle rotated segments"
    )
    
    # Optional arguments
    parser.add_argument(
        "--nReadout", type=int, default=256, required=False, help="Number of readout points per spoke"
    )
    # Optional arguments
    parser.add_argument(
        "--TR", type=float, default=2.3e-3, required=False, help="TR in s"
    )
    # Optional arguments
    parser.add_argument(
        "--grad_dt", type=float, default=8e-6, required=False, help="Sampling dwell time in s"
    )
    parser.add_argument(
        "--nTestAngles", type=int, default=200, required=False, 
        help="Number of test angles for discretized theta space"
    )
    parser.add_argument(
        "--lambdas_for_grid_search", type=int, nargs="+", 
        default=np.arange(1, 6.5, 0.5), required=False, 
        help="Number of test angles for discretized theta space"
    )
    parser.add_argument(
        "--out_rotmat_txt_path", type=str, required=False, default=None, 
        help="Path of output text file to save rotations"
    )
    parser.add_argument(
        "--save_coords", type=bool, required=False, default=True, 
        help="Set true to also save traj coordinates"
    )
    parser.add_argument(
        "--out_coords_npy_path", type=str, required=False, default=None, 
        help="Path of output npy file to save coordinates"
    )
    parser.add_argument(
        "--seg_rot_txt_file", type=str, required=False, 
        default=GOLDEN_ANGLE_3D_ROTS, 
        help="Txt file for rotations of whole segments (groups of continuous TRs)"
    )
    return parser.parse_args()


def main():
    """
    Run discrete optimization to calculate rotation angles for Arc-ZTE
    with grid search for lamda to choose smallest value with no refocusing
    """
    args = parse_args()

    # Output save paths
    if args.out_rotmat_txt_path is None:
        args.out_rotmat_txt_path = f"rotmats_1seg_{args.arc_angle}deg_{args.spokes_per_seg}spokes.txt"

    if args.out_coords_npy_path is None:
        args.out_coords_npy_path = f"coords_traj_{args.arc_angle}deg_{args.spokes_per_seg}spokes.npy"

    # Calculate coords for single segment
    print(f"Calculating trajectory for arc angle {args.arc_angle} for segment with {args.spokes_per_seg} spokes")
    for lamda in args.lambdas_for_grid_search: 

        # Run optimization
        scheme = FurthestDist_CostFunction(lamda, args.arc_angle, args.nReadout, 
                                           args.spokes_per_seg, args.grad_dt, 
                                           args.TR, nTestAngles=args.nTestAngles)
        scheme.rotate()

        # Dims [nSpokes, nReadout, 3]
        coords_single_segment = scheme.spoke_arr.transpose(0,2,1)[:, 0:args.nReadout]
        
        # Calculate instances of refocusing
        refocus_metric = percentage_TRs_with_refocusing_metric(scheme.spoke_arr.transpose(0,2,1), 
                                                                num_TR_dephasing=3, 
                                                                refocus_level=1.25, 
                                                                print_flag=False)
         
        # Done if no refocusing occurred in segment. Else increase lambda and try again
        if refocus_metric != 0:
            print(f"{refocus_metric}% of TRs contained refocusing for lambda {lamda}. Increasing lambda and re-trying..\n")
            
        elif refocus_metric == 0:
            print(f"Finished grid search! No instances of refocusing occured for lambda {lamda}")

            covg_metric = cov_uniformity_metric(coords_single_segment[:, -1, :], n=3000)
            print(f"Coverage uniformity metric was {covg_metric:.3f}\n ")

            save_Rs_txt(scheme, args.out_rotmat_txt_path)
            print(f"Saved rotation matrices for each spoke in segment at {args.out_rotmat_txt_path}")
            break
    
    # Grid search failed
    if refocus_metric != 0:
        raise ValueError('Refocusing occurred for all values of lambdas_for_grid_search. Increase values.')
    
    # Take calculated single segment and rotate by golden angle
    coords_full_traj = np.zeros((args.num_segs*args.spokes_per_seg, args.nReadout, 3))
    M_file = read_seg_rotmats_from_txt(args.seg_rot_txt_file, args.spokes_per_seg)

    for i in range(args.num_segs):

        if i == 0:
            coords_rot_seg = coords_single_segment # don't rotate first segment
        else:
            # apply rotation matrix
            coords_rot_seg = M_file[i-1].reshape(3,3) @ coords_single_segment.reshape(-1, 3).transpose(1,0) # [3, spokes*RO]
            coords_rot_seg = coords_rot_seg.transpose(1,0).reshape(args.spokes_per_seg, args.nReadout, 3) # reshape back

        coords_full_traj[i*args.spokes_per_seg : (i+1)*args.spokes_per_seg] = coords_rot_seg
    
    # Save coordinates to npy file if specified
    if args.save_coords:
        np.save(args.out_coords_npy_path, coords_full_traj)
        print(f"Saved trajectory coords for segment at {args.out_coords_npy_path}")

if __name__ == "__main__":
    main()