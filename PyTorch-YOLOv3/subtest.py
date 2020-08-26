import numpy as np
import torch

def main():
    label_path = 'data/custom/labels/train000001.txt'
    boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
    print(boxes)
    # Extract coordinates for unpadded + unscaled image
    # x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    # y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    # x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    # y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
    # # Adjust for added padding
    # x1 += pad[0]
    # y1 += pad[2]
    # x2 += pad[1]
    # y2 += pad[3]
    # # Returns (x, y, w, h)
    # boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    # boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    # boxes[:, 3] *= w_factor / padded_w
    # boxes[:, 4] *= h_factor / padded_h

    targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes
    print(targets)
    targets[:, 2] = 1 - targets[:, 2]
    print(targets)

if __name__ == '__main__':
    main()
