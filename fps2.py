import argparse
import time
from data import read_image
import cv2
import torch
from numpy import random
import pyrealsense2 as rs
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from decimal import Decimal

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz,save_vidio = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size,opt.save_vidio
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP33 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    pipe = rs.pipeline()
    cfg = rs.config()
    # cfg.enable_device_from_file(file_name=source)
    cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640,480, rs.format.rgb8, 30)
    pipe.start(cfg)
    a=0
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    start = time.time()
    # save_path += '.mp4'
    colorizer = rs.colorizer()
    hole_filling = rs.hole_filling_filter()
    try:
        while True:
            # t0=time.time()
            # Wait for a coherent pair of frames: depth and color
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            im0s = np.asanyarray(color_frame.get_data())
            # colorizer = rs.colorizer()
            align = rs.align(rs.stream.color)
            frames= align.process(frames)
            # 获取对齐更新后的深度图
            aligned_depth_frame = frames.get_depth_frame()
            #滤镜
            # hole_filling = rs.hole_filling_filter()
            filled_depth = hole_filling.process(aligned_depth_frame)
            colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            depth = np.asanyarray(filled_depth.get_data())
            # depth = np.asanyarray(aligned_depth_frame.get_data())
            # t1=time.time()
            # print("forward time {:4f} ".format(t1-t0))
            img =read_image(im0s)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                # Inference
                # t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # t2 = time_synchronized()
                # t2=time.time()
                # #
                # print("inference time :"+str(t2-t1))
            t3 = time.time()
            for _,det in enumerate(pred):
                if len(det):
                        # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                        depth_temp = depth[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])].astype(float)
                        depth_temp = depth_temp * 0.001
                        dist, _, _, _ = cv2.mean(depth_temp)
                        dist_temp = Decimal(dist).quantize(Decimal('0.000'))
                        y_mid=(int(xyxy[1])+int(xyxy[3]))/2
                        dis_y = int(xyxy[3])-int(xyxy[1])
                        x_mid = (xyxy[0] + xyxy[2]) / 2
                        dis_x = int(xyxy[2]- xyxy[0])
                        len_x = 0
                        for i in range(dis_x):
                            u = depth[int(y_mid), int(i + xyxy[0])]
                            len_i = 0.00165 * u
                            len_x += len_i
                        len_y = 0
                        for i in range(dis_y):
                            u =depth[int(i + xyxy[1] - 7), int(x_mid)]
                            len_i = 0.00165 * u
                            len_y += len_i
                        label = f'{names[int(cls)]} {conf:.2f}'
                        label1 = str(dist_temp) + "m" + str(int(len_x)) + "mm"+',' +str(int(len_y))+ "mm"

                        plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)
                        plot_one_box(xyxy, colorized_depth, label=label1, color=colors[int(cls)],
                                       line_thickness=2)
            # t4 = time.time()
            # print("deep  time :" + str(t4 - t3))
            images = np.hstack((cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR), cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR)))
            a+=1
            if a %10 ==0:
                print("\r A FPS:{:.2f}".format(a/(time.time()-start)))
            if view_img:
                cv2.imshow("RealSense", images)
            # t6 = time.time()
            # print("cv2imshow()  time :" + str(t6 - t4))
                # 按下“q”键停止q
                if cv2.waitKey(1) & 0xFF == ord('q'):  # cv2.waitKey(1) 1毫秒读一次
                     # cv2.destroyAllWindows()
                     break
            if  save_vidio : #and a!=100:  #!=len(images):
                fps, w, h = 30, images.shape[1], images.shape[0]
                # save_path += '.mp4'
                if a==1:
                    vid_writer = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(images)
                # vid_writer.release()
                # print("video [{}/{}] save path{} ".format(a,len(im0s),"./test.mp4"))
            else:
                print("Done,time{:.2f}".format(time.time()-start))
                vid_writer.release()
                cv2.destroyAllWindows()
                break
            # t7 =time.time()
            # print('cv2.waitKey()' + str(t7 - t6))

    finally:
        pipe.stop()
        pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source',default='runs/20201118_145711.bag', type=str, help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_vidio', default=True, help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
