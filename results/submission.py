
import os

if __name__ == '__main__':
    folder = 'submission'
    out_file = 'submission.txt'
    cam_file = os.listdir(folder)

    with open(out_file, 'w') as fo:
        for cam in cam_file:
            if 'debug' in cam:
                continue
            cam_path = os.path.join(folder, cam)
            with open(cam_path, 'r') as fi:
                data = fi.read()
                lines = data.splitlines()
                print("{}: {}".format(cam, len(lines)))
                for line in lines:
                    cam, frame, path, label = line.split()
                    if frame == '13499':
                        continue
                    fo.write("{} {} {} {}\n".format(cam, frame, path, label))
    print("Done! Good luck")