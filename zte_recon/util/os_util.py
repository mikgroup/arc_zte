import os

def set_gpu(device_num):
    '''Make only one device visible in notebook
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)


def set_num_cpus(num_cpus):
    '''Set max number of cpus
    '''
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)