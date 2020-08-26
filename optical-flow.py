import numpy as np
import cv2
import argparse
import os
import glob

parser = argparse.ArgumentParser(description = 'Optical flow of video files. May be switched between normal and dense optical flow modes.')
parser.add_argument('--src', type = str, help = 'your video source directory')
parser.add_argument('--mode', default = 'dense', type = str, help = 'options for different optical flow modes')
parser.add_argument('--save', default = True, type = bool, help = 'choose to save the screenshots')
parser.add_argument('--format',default = 'avi', type = str, help = 'video format')
parser.add_argument('--thr', default = 50, type = int, help = 'interval for image saving')
args = parser.parse_args()


def writeImageFolders(path):
    try:
        os.mkdir(path)
        print("Created new directory: " + path + "/")
    except OSError as error:
        print("Created new directory: " + path + "/")

def normalOFlow(cap, path, thr):
    cnt = 0
    imgsuffix = 0
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # if args.save == True:
    #     subpath = os.path.join(path,"NORMAL")
    #     try:
    #         os.mkdir(subpath)
    #         print("Created new subdirectory: "+ subpath + "/")
    #     except OSError as error:
    #         print("Directory: " + subpath + "/ already exists")

    while(1):
        cnt += 1
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        if(args.save == True):
            if cnt % thr == 0:
                sfx = "%08d" % imgsuffix
                cv2.imwrite(path+"/"+sfx+".png", img)
                imgsuffix += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

def denseOFlow(cap, path, thr):
    cnt = 0
    imgsuffix = 0
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    # if args.save == True:
    #     subpath = os.path.join(path,"DENSE")
    #     try:
    #         os.mkdir(subpath)
    #         print("Created new subdirectory: " + subpath + "/")
    #     except OSError as error:
    #         print("Directory: " + subpath + "/ already exists")

    while(1):
        cnt += 1
        ret, frame2 = cap.read()
        try:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('frame2',bgr)
            if(args.save == True):
                if cnt % thr == 0:
                    sfx = "%08d" % imgsuffix
                    cv2.imwrite(path+"/F"+sfx+".png", frame2)
                    cv2.imwrite(path+"/M"+sfx+".png", bgr)
                    imgsuffix += 1

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png',frame2)
            #     cv2.imwrite('opticalhsv.png',rgb)
            prvs = next
        except:
            print("END OF VIDEO.")
            cap.release()
            cv2.destroyAllWindows()
            break

if args.src:
    format = '.'+args.format
    vidsrc = sorted(glob.glob(os.path.join(args.src, '*'+format)))
    for vs in vidsrc:
        cap = cv2.VideoCapture(vs)
        directory = vs.strip(args.src)[:-3]
        #directory = directory[1:]
        path = os.path.join(os.path.join(os.path.dirname(__file__),'Sample'),directory)
        print('Images generated on directory:' + path)
        if args.save == True:
            writeImageFolders(path)

        if args.mode == 'normal':
            # Do normal optical flow
            # params for ShiTomasi corner detection
            normalOFlow(cap, path, args.thr)
        elif args.mode == 'dense':
            denseOFlow(cap, path, args.thr)
            # break
        else:
            print('I do not understand your mode selection?')
            exit(1)

else:
    print('Please input your image directory')
    exit(1)
