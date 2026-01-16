import argparse
import numpy as np

from arc_zte_sim.metrics import percentage_TRs_with_refocusing_metric, cov_uniformity_metric
from arc_zte_sim.theta_i_schemes import FurthestDist_CostFunction 
from arc_zte_sim.rotate_spokes import save_Rs_txt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate trajectory for Arc-ZTE segment using optimization scheme"
    )
    # Required arguments
    parser.add_argument(
        "--arc_angle", type=float, required=True, help="Arc angle (deg)"
    )
    parser.add_argument(
        "--nSpokes_seg", type=int, required=True, help="Number of spokes in segment"
    )
    
    # Optional arguments
    parser.add_argument(
        "--nReadout", type=int, default=256, required=False, help="Number of readout points per spoke"
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
        "--save_coords", type=bool, required=False, default=False, 
        help="Set true to also save traj coordinates"
    )
    parser.add_argument(
        "--out_coords_npy_path", type=str, required=False, default=None, 
        help="Path of output npy file to save coordinates"
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
        args.out_rotmat_txt_path = f"rotmats_1seg_{args.arc_angle}deg_{args.nSpokes_seg}spokes.txt"
    if args.out_coords_npy_path is None:
        args.out_coords_npy_path = f"coords_1seg_{args.arc_angle}deg_{args.nSpokes_seg}spokes.npy"

    print(f"Calculating trajectory for arc angle {args.arc_angle}")
    for lamda in args.lambdas_for_grid_search: 

        # Run optimization
        scheme = FurthestDist_CostFunction(lamda, args.arc_angle, args.nReadout, 
                                           args.nSpokes_seg, nTestAngles=args.nTestAngles)
        scheme.rotate()
        
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

            covg_metric = cov_uniformity_metric(scheme.spoke_arr.transpose(0,2,1)[:, -1, :], n=3000)
            print(f"Coverage uniformity metric was {covg_metric:.3f}\n ")

            save_Rs_txt(scheme, args.out_rotmat_txt_path)
            print(f"Saved rotation matrices for segment at {args.out_rotmat_txt_path}")

            if args.save_coords:
                np.save(args.out_coords_npy_path, scheme.spoke_arr)
                print(f"Saved trajectory coords for segment at {args.out_coords_npy_path}")
            break
    
    # Grid search failed
    if refocus_metric != 0:
        raise ValueError('Refocusing occurred for all values of lambdas_for_grid_search. Increase values.')
    

if __name__ == "__main__":
    main()