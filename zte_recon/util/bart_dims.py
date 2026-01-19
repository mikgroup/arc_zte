
def ksp_to_bart(ksp):
    # input sigpy dim [nCoils, nSpokes, nRO]
    # output dim [1, nSpokes, nRO, nCoils]
    return ksp.transpose(1,2,0)[None]


def coord_to_bart(coord):
    # input sigpy dim [nSpokes, nRO, 3]
    # output dim [3, nSpokes, nRO]
    return coord.transpose(2,0,1)

def temporal_ksp_to_bart(ksp):
    # puts temporal dimension into dim 10
    # input sigpy dim [frames, nCoils, nSpokes, nRO]
    # output dim [1, nSpokes, nRO, nCoils, 1,1,1,1,1, time]

    return ksp.transpose(2,3,1,0)[None, :, :, :, None, None, None, None, None, None, :]

def temporal_coord_to_bart(coord):
    # puts temporal dimension into dim 10
    # input sigpy dim [frames, nSpokes, nRO, 3]
    # output dim [3, nSpokes, nRO, 1, 1,1,1,1,1, time]

    return coord.transpose(3,1,2,0)[:, :, :, None,  None, None, None, None, None, None, :]

def ir_ksp_to_bart(ksp):
    # puts ir dimension into dim 5
    # input sigpy dim [ir_time, nCoils, num_segs, nRO]
    # output dim [1, num_segs, nRO, nCoils, 1, ir_time]
    return ksp.transpose(2,3,1,0)[None, :, :, :, None, :]

def ir_coord_to_bart(coord):
    # puts ir dimension into dim 5
    # input sigpy dim [ir_time, num_segs, nRO, 3]
    # output dim [3, nSpokes, nRO, 1, 1, ir_time]

    return coord.transpose(3,1,2,0)[:, :, :, None, None, :]