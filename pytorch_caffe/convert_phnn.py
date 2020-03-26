from caffenet import *
import torch

def convert_caffe(protofile, weightfile, out_path):

    net = CaffeNet(protofile)
    net.load_weights(weightfile)
    import ipdb; ipdb.set_trace()

    net.eval()
    torch.save(net.state_dict(), out_path)

