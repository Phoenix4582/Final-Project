import torch
import torch.nn as nn
import torch.nn.functional as F

# from newNN import get_model_instance_segmentation
# from newNN import fusionModel

def main():
    x = torch.randn([4,3,320,540])
    y = torch.randn([4,3,320,540])

    z = torch.randn([1,320,320])

    zz = torch.cat((z,z,z), dim = 0)
    print(zz.shape)
    # model = fusionModel(num_classes = 2)
    # result = model(x, y)
    #
    # print(result)

if __name__ == '__main__':
    main()
