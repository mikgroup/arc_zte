import bart

def est_maps_3d(ksp_lowres, coord_lowres, img_shape, soft_sens=False, 
                crop_thres=0.001, grid_size=10, ker_size=4):
    """Estimates ESPIRiT maps from low-res non-cart data
    and corresponding coordinates. Grids the non-cart data
    to cartesian, zeropads gridded data in k-space, then
    runs ESPIRiT with passed params. 

    Runs ESPIRiT using BART. One map per coil. 

    Parameters
    ----------
    ksp_lowres : ndarray, (1, nSpokes, nReadout, nCoils)
        Lowres non-cartesian k-space in BART dims
    coord_lowres : ndarray, (3, nSpokes, nReadout)
        Lowres non-cartesian coords in BART dims
    img_shape : [nX, nY, nZ]
        Shape of output maps 
    soft_sens : bool, optional
        Soft cutoffs for ESPIRiT eval thresh, by default False
    crop_thres : float, optional
        Eval thresh for ESPIRiT mask, by default 0.001 (no mask)
    grid_size : int, optional
        Grid size for gridding non-cart data onto, by default 10
        (grid size 10 empirically works for gridding WASPI with scale 1/8)
    ker_size : int, optional
        ESPIRiT kernel size, by default 4
        (ker size 4 works for small grid size used with WASPI gridding)

    Returns
    -------
    ndarray (nX, nY, nZ, nCoils)
        ESPIRiT maps output from running BART
    """

    # Inverse gridding of lowres data
    inv_nufft_cmd = f"nufft -i -d {grid_size}:{grid_size}:{grid_size}"
    lowres_img = bart.bart(1, inv_nufft_cmd, coord_lowres, ksp_lowres)

    # Get lowres gridded k-space
    lowres_cart_ksp = bart.bart(1, 'fft 7', lowres_img)

    # Zeropad to size of desired maps
    zp_cmd = f"resize -c 0 {img_shape[0]} 1 {img_shape[1]} 2 {img_shape[2]}"
    ksp_zeropd = bart.bart(1, zp_cmd, lowres_cart_ksp)

    # Run espirit
    ecalib_cmd = f"ecalib -m 1 -k {ker_size} -c {crop_thres} -r {grid_size} "
    if soft_sens:
        ecalib_cmd = ecalib_cmd + ' -S'
    maps = bart.bart(1, ecalib_cmd, ksp_zeropd) 

    return maps