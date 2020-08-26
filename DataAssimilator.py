import os, glob
import argparse
import shutil

parser = argparse.ArgumentParser(description = "Gather frames, masks and motion data into one folder.")

parser.add_argument("--src", default = '', type = str, help = "parent directory of the folder(s)")
parser.add_argument("--dest",default = 'Test2', type = str, help = "parent directory of the destination")
parser.add_argument("--mode", default = 'train', type = str, help = "define whether training or testing data is assimilated.")

args = parser.parse_args()

def initFolders(dir):
    #dest = os.path.join(args.dest, dir)
    try:
        os.mkdir(dir)
        print("Created subfolder: "+ dir)
    except:
        print("Directory: "+ dir + " already exists.")

def main():
    cnt = 0
    frame_name = 'img.png'
    mask_name = 'label.png'
    if args.mode == 'train':
        frame_dir = 'Frames'
        masks_dir = 'Masks'
        motion_dir = 'Motion'
        d_dirs = [frame_dir, masks_dir, motion_dir]
        dest_dirs = [os.path.join(args.dest, d) for d in d_dirs]
        for d in dest_dirs:
            print(d)
            initFolders(d)

        f_dir, m_dir, o_dir = dest_dirs

        if args.src:
            folders = os.listdir(args.src)
            for f in folders:
                dir = os.path.join(args.src, f)
                jsons = sorted(glob.glob(os.path.join(dir, '*.json')))
                for js in jsons:
                    js_src = js[-14:]
                    subfolder_src = js_src.replace(".","_")
                    data_id = js_src[:-5].strip('F')

                    c = "%07d" % cnt
                    subdir = os.path.join(dir, subfolder_src)
                    frame_png = os.path.join(subdir, frame_name)
                    mask_png = os.path.join(subdir, mask_name)

                    motion_png = os.path.join(dir, "M"+data_id+".png")
                    shutil.copy(frame_png, os.path.join(f_dir, "FRAME"+c+".png"))
                    print("Copied " + frame_png + "to " + os.path.join(f_dir, "FRAME"+c+".png"))
                    shutil.copy(mask_png, os.path.join(m_dir, "MASK"+c+".png"))
                    print("Copied " + mask_png + "to " + os.path.join(m_dir, "MASK"+c+".png"))
                    shutil.copy(motion_png, os.path.join(o_dir, "MOTION"+c+".png"))
                    print("Copied " + motion_png + "to " + os.path.join(o_dir, "MOTION"+c+".png"))
                    cnt += 1

        print("-----------------------------------------------")
        print("Done! Training data size : {}".format(cnt))
        print("-----------------------------------------------")

    elif args.mode == 'test':
        frame_dir = 'Frames'
        masks_dir = 'Masks'
        d_dirs = [frame_dir, masks_dir]
        dest_dirs = [os.path.join(args.dest, d) for d in d_dirs]
        for d in dest_dirs:
            print(d)
            initFolders(d)

        f_dir, m_dir = dest_dirs

        if args.src:
            folders = os.listdir(args.src)
            for f in folders:
                dir = os.path.join(args.src, f)
                jsons = sorted(glob.glob(os.path.join(dir, '*.json')))
                for js in jsons:
                    js_src = js.strip(dir + '\\')
                    #print(js_src)
                    subfolder_src = js_src.replace(".","_")
                    #print(subfolder_src)

                    c = "%07d" % cnt
                    subdir = os.path.join(dir, subfolder_src)
                    frame_png = os.path.join(subdir, frame_name)
                    mask_png = os.path.join(subdir, mask_name)

                    shutil.copy(frame_png, os.path.join(f_dir, "FRAME"+c+".png"))
                    print("Copied " + frame_png + "to " + os.path.join(f_dir, "FRAME"+c+".png"))
                    shutil.copy(mask_png, os.path.join(m_dir, "MASK"+c+".png"))
                    print("Copied " + mask_png + "to " + os.path.join(m_dir, "MASK"+c+".png"))
                    cnt += 1

        print("-----------------------------------------------")
        print("Done! Training data size : {}".format(cnt))
        print("-----------------------------------------------")

if __name__ == '__main__':
    main()
