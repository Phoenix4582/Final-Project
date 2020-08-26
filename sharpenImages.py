import os, glob
import cv2
import numpy as np
import argparse

def initArgs():
    parser = argparse.ArgumentParser("Sharpen the raw images extracted from video")
    parser.add_argument('--src', default = '', type = str, help = 'secondary level parent directory of image files')

    args = parser.parse_args()
    return args

def test(kernel):
    img = "Screenshots/RGB1/F00000015.png"

    try:
        image = cv2.imread(img)
        output = cv2.filter2D(image, -1, enhanceKernel)

        # cv2.namedWindow("raw", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("sharpened", cv2.WINDOW_AUTOSIZE)
        #
        # cv2.imshow("raw", image)
        cv2.imwrite("sharpened", output)

        cv2.waitKey(0)

        cv2.destroyAllWindows()
    except:
        print("File "+ img + " does not exist")
        exit(1)

def main():
    sharpeningKernel = np.array(([-1,-1,-1],[-1,9,-1],[-1,-1,-1]), dtype='int')
    exsharpeningKernel = np.array(([1,1,1],[1,-7,1],[1,1,1]), dtype='int')
    enhanceKernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]]) / 8.0

    # img = "Sample/Origin.png"
    #
    # try:
    #     image = cv2.imread(img)
    #     output1 = cv2.filter2D(image, -1, sharpeningKernel)
    #     output2 = cv2.filter2D(image, -1, exsharpeningKernel)
    #     output3 = cv2.filter2D(image, -1, enhanceKernel)
    #     # cv2.namedWindow("raw", cv2.WINDOW_AUTOSIZE)
    #     # cv2.namedWindow("sharpened", cv2.WINDOW_AUTOSIZE)
    #     #
    #     # cv2.imshow("raw", image)
    #     cv2.imwrite("Sample/Sharpened.png", output1)
    #     cv2.imwrite("Sample/Ex_sharpened.png", output2)
    #     cv2.imwrite("Sample/Enhanced.png", output3)
    #
    #     cv2.waitKey(0)
    #
    #     cv2.destroyAllWindows()
    # except:
    #     print("File "+ img + " does not exist")
    #     exit(1)

    # test(sharpeningKernel)
    # test(exsharpeningKernel)
    # test(enhanceKernel)

    args = initArgs()
    if args.src:
        dirs = os.listdir(args.src)
        for d in dirs:
            subdir = os.path.join(args.src, d)
            frames = sorted(glob.glob(os.path.join(subdir, '*.png')))
            for f in frames:
                filename = f.strip(subdir + '\\')
                if filename[0] == 'F':
                    print("Reading file:" + f)
                    image = cv2.imread(f)
                    output = cv2.filter2D(image, -1, enhanceKernel)
                    cv2.imwrite(f, output)
                    print("File " + f + "has been sharpened.")

    else:
        print("Where is your parent directory?")
        exit(1)

if __name__ == '__main__':
    main()
