import sys
from time import time

import cv2
import numpy as np
import torch


class ObjectDetection:

    def __init__(self, input_file, out_file="Labeled_Video.avi"):

        self.input_file = input_file
        self.model = self.load_model()
        self.model.conf = 0.4  # set inference threshold at 0.3
        self.model.iou = 0.3  # set inference IOU threshold at 0.3
        self.model.classes = [0]  # set model to only detect "Person" class
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_file(self):

        cap = cv2.VideoCapture(self.input_file)
        assert cap is not None
        return cap

    def load_model(self):

        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1].to(
            'cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        return labels, cord

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(
                row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
            label = f"{int(row[4]*100)}"
            cv2.putText(frame, label, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(
                frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def __call__(self):
        player = self.get_video_from_file()  # create streaming service for application
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        fc = 0
        fps = 0
        tfc = int(player.get(cv2.CAP_PROP_FRAME_COUNT))
        tfcc = 0
        while True:
            fc += 1
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps += 1/np.round(end_time - start_time, 3)
            if fc == 10:
                fps = int(fps / 10)
                tfcc += fc
                fc = 0
                per_com = int(tfcc / tfc * 100)
                print(
                    f"Frames Per Second : {fps} || Percentage Parsed : {per_com}")
            out.write(frame)
        player.release()


link = sys.argv[1]
output_file = sys.argv[2]
a = ObjectDetection(link, output_file)
a()
