import cv2
import os
import sys
import argparse

join=os.path.join

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-inp_dir', '--inp_dir', type=str, required=True,
                    help='Path to input directory')
parser.add_argument('-out_dir', '--out_dir', type=str, required=True,
                    help='Path to output directory')
args = parser.parse_args()


def getFrame(sec, fname):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    
    if hasFrames:
        print(fname)
        h,w = image.shape[0], image.shape[1]
        max_dim = 512
        if h > w:
            height = max_dim
            width = float(w)*max_dim/float(h)
        else:
            width = max_dim
            height = float(h)*max_dim/float(w)

        if width%2 != 0:
            width += 1
        if height%2 != 0:
            height += 1
            
        cv2.imwrite(fname, cv2.resize(image, (int(width), int(height))))
    return hasFrames

inp_dir = args.inp_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

l = len(os.listdir(inp_dir))
print(l)
fnames = [join(inp_dir, fname) for fname in os.listdir(inp_dir) if not ".DS" in fname]

for i, fname in enumerate(fnames):
    vidcap = cv2.VideoCapture(fname)
    out_dir = join(output_dir, str(fname.split("/")[-1].split(".")[0]))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sec = 0
    frameRate = 1.0/30.0 #//fps = 30
    count=1
    idx = "000000" + str(count)
    idx = idx[-6:]
    fname = join(out_dir, "image"+str(idx)+".jpg")
    success = getFrame(sec, fname)
    while success:
        count = count + 1
        sec = sec + frameRate
        idx = "000000" + str(count)
        idx = idx[-6:]
        fname = join(out_dir, "image"+str(idx)+".jpg")
        success = getFrame(sec, fname)
        print(count)
        if count == 360:
            break




