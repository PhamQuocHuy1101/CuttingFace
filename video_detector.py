import cv2
import argparse
import os

import time
import insightface
from insightface.utils import face_align
from imutils.video import FileVideoStream
from imutils.video import FPS

# ignore warning of onnxruntime
import onnxruntime as ort
ort.set_default_logger_severity(3)

detector_path = './antelope/scrfd_10g_bnkps.onnx' # Path to antelope file - insightface 
detector_device = -1 # -1 cpu: 0, 1 gpu 

class App():
    def __init__(self, video_path, store_dir, prefix):
        self.face_detector = insightface.model_zoo.get_model(name=detector_path)
        self.face_detector.prepare(ctx_id=detector_device, input_size=(640, 640))

        self.fvs = FileVideoStream(video_path)
        self.origin_fps = self.fvs.stream.get(cv2.CAP_PROP_FPS)
        self.store_dir = store_dir
        self.prefix = prefix

    def __del__(self):
        print("delete")
    
    def run(self):
        '''
            Will clear the previous records
        '''
        fvs = self.fvs.start()
        time.sleep(1.0)

        fps = FPS().start()
        count = 0
        check_frame = True
        while fvs.more() and check_frame:
            i = 0
            frame = fvs.read()
            while i < self.origin_fps / 2 and frame is not None: # fast skip frame 
                frame = fvs.read()
                i += 1
            if frame is None :
                    break
            cv2.imshow("Face", frame)
            # self.out_video(frame)
            fps.update()
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
            if cv2.waitKey(0) & 0xFF == ord("c"):
                print('start cut')
                _, pts5s = self.face_detector.detect(frame)
                if pts5s is not None:
                    for pts5 in pts5s:
                        count += 1
                        face_img = face_align.norm_crop(frame, pts5)
                        cv2.imwrite(os.path.join(self.store_dir, f'{self.prefix}_{count}.jpg'), face_img)
                    print("Cut ", len(pts5s))

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        cv2.destroyAllWindows()
        fvs.stop()

parser = argparse.ArgumentParser(description="Record child from video")
parser.add_argument('-v', '--video-path', required=True, help='Video path')
parser.add_argument('-s', '--store-dir', required=True, help='store dir path')
parser.add_argument('-p', '--prefix', required=True, help='prefix')


args = parser.parse_args()
os.makedirs(args.store_dir, exist_ok=True)
app = App(args.video_path, args.store_dir, args.prefix)
app.run()