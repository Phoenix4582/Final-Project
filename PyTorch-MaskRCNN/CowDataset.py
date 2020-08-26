import torch
import os, math
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# from xml.etree import ElementTree as et
# import xml.dom.minidom as x
class NoModeException(Exception):
    pass

class CowDataset(object):
    def __init__(self, root, transforms, mode = "normal", hasMotion = "true"):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Frames"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))
        #self.xmls = list(sorted(os.listdir(os.path.join(root, "Xmls"))))
        if hasMotion:
            self.motion = list(sorted(os.listdir(os.path.join(root, "Motion"))))
            self.hasMotion = True
        else:
            self.hasMotion = False
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'postFusion':
            assert(self.hasMotion == True)
            img, motion = self.getZipImages(idx)
        else:
            img = self.getTrainImages(idx)

        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        #xml_path = os.path.join(self.root, "Xmls", self.xmls[idx])

        width, height = img.size
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:,None,None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # boxes = []
        # for children in root:
        #     #label = 0
        #     for g in children.iter('object'):
        #         for c in g:
        #             if c.tag == 'name':
        #                 if c.text == 'cow':
        #                     #label = 1
        #                     continue
        #                 else:
        #                     break
        #
        #             if c.tag == 'bndbox':
        #                 for last in c.iterfind('xmin'):
        #                     xmin = round(float(last.text))
        #                 for last in c.iterfind('xmax'):
        #                     xmax = round(float(last.text))
        #                 for last in c.iterfind('ymin'):
        #                     ymin = round(float(last.text))
        #                 for last in c.iterfind('ymax'):
        #                     ymax = round(float(last.text))
        #
        #                 if self.datatype == "COCO":
        #                     x = xmin
        #                     y = ymin
        #                     w = xmax - xmin
        #                     h = ymax - ymin
        #                     boxes.append([x,y,w,h])
        #                 elif self.datatype == "YOLO":
        #                     xl = xmin / width
        #                     yl = ymin / height
        #                     xh = xmax / width
        #                     yh = ymax / height
        #                     boxes.append([xl,yl,xh,yh])
        #
        #
        #             if c.tag == 'robndbox':
        #                 for last in c.iterfind('cx'):
        #                     cx = round(float(last.text))
        #                 for last in c.iterfind('cy'):
        #                     cy = round(float(last.text))
        #                 for last in c.iterfind('w'):
        #                     w =  round(float(last.text))
        #                 for last in c.iterfind('h'):
        #                     h =  round(float(last.text))
        #                 for last in c.iterfind('angle'):
        #                     angle = round(float(last.text),3)
        #
        #                 if angle > 3.142:
        #                     angle = angle - 3.142
        #                 if angle < -3.142:
        #                     angle = angle + 3.142
        #
        #                 w_final = w
        #                 h_final = h
        #                 a_final = angle
        #
        #                 bxmin = round(cx - w_final/2 * math.cos(a_final) - h_final/2 * math.sin(a_final))
        #                 bxmax = round(cx + w_final/2 * math.cos(a_final) + h_final/2 * math.sin(a_final))
        #                 bymin = round(cy - w_final/2 * math.sin(a_final) - h_final/2 * math.cos(a_final))
        #                 bymax = round(cy + w_final/2 * math.sin(a_final) + h_final/2 * math.cos(a_final))
        #
        #                 #width_real = round(w_final * abs(math.cos(a_final)) + h_final * abs(math.sin(a_final)))
        #                 #height_real = round(w_final * abs(math.sin(a_final)) + h_final * abs(math.cos(a_final)))
        #
        #                 if bxmin > bxmax:
        #                     tempX = bxmin
        #                     bxmin = bxmax
        #                     bxmax = tempX
        #
        #                 if bymin > bymax:
        #                     tempY = bymin
        #                     bymin = bymax
        #                     bymax = tempY
        #
        #                 bxmin = max(minwidth, bxmin)
        #                 bxmax = min(maxwidth, bxmax)
        #                 bymin = max(minheight, bymin)
        #                 bymax = min(maxheight, bymax)
        #
        #                 width_real = bxmax - bxmin
        #                 height_real = bymax - bymin
        #
        #                 if self.datatype == "COCO":
        #                     boxes.append([bxmin, bymin, width_real, height_real])
        #                 elif self.datatype == "YOLO":
        #                     boxes.append([bxmin/width, bymin/height, bxmax/width, bymax/height])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            if self.mode == 'postFusion':
                data, target = self.transforms([img, motion], target)
                img, motion = data
            else:
                img, target = self.transforms(img, target)

        if self.mode == 'postFusion':
            return img, motion, target
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

    def getTrainImages(self, idx):
        if self.mode == "motion":
            assert(self.hasMotion == True)
            img_path = os.path.join(self.root, "Motion", self.motion[idx])
            img = Image.open(img_path).convert("L")
            return img

        elif self.mode == "normal":
            img_path = os.path.join(self.root, "Frames", self.imgs[idx])
            img = Image.open(img_path).convert("RGB")
            return img

        elif self.mode == "preFusion":
            img_base_path = os.path.join(self.root, "Frames", self.imgs[idx])
            img_base = Image.open(img_base_path).convert("RGB")
            if self.hasMotion:
                img_sub_path = os.path.join(self.root, "Motion", self.motion[idx])
                img_sub = Image.open(img_sub_path).convert("L")
                tensorT = transforms.ToTensor()
                imageT = transforms.ToPILImage()
                stackedImages = []
                baseTensor, subTensor = tensorT(img_base), tensorT(img_sub)
                baseTensor, subTensor = baseTensor.mul(255).to(torch.int32), subTensor.mul(255).to(torch.int32)
                # print(baseTensor)
                # print(subTensor)
                for i in range(baseTensor.size()[0]):
                    layer = (baseTensor[i] | subTensor)
                    stackedImages.append(layer)
                resultTensor = torch.stack(tuple(stackedImages),dim = 0)
                resultTensor = resultTensor.squeeze(1)
                # print(resultTensor.size())
                #img = Image.fromarray(resultTensor.mul(255).byte().cpu().numpy())
                img = imageT(resultTensor.mul(1/255).to(torch.float))
            else:
                img = img_base
            return img
        else:
            NoModeException('No mode' + self.mode + 'available.')

    def getZipImages(self, idx):
        tensorT = transforms.ToTensor()
        imageT = transforms.ToPILImage()
        img_path = os.path.join(self.root, "Frames", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        motion_path = os.path.join(self.root, "Motion", self.motion[idx])
        motion = Image.open(motion_path).convert("L")
        motion_tensor = tensorT(motion)
        result_tensor = torch.stack((motion_tensor,motion_tensor,motion_tensor),dim = 0) # Convert 1 * H * W to 3 * H * W
        motion = imageT(result_tensor)
        return img, motion
