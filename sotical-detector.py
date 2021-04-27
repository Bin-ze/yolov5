import argparse
import time
from data import read_image
import cv2
import torch
from numpy import random
import pyrealsense2 as rs
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import numpy as np
from utils.utils import plot_dots_on_people,distancing_all
from decimal import Decimal
def detects_img(pipe,hole_filling,colorizer):
    for x in range(100):
        frames =pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    # 停止管道传输
    pipe.stop()
    # if not color_frame:
    #     continue
    im0s = np.asanyarray(color_frame.get_data())
    # colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    # 获取对齐更新后的深度图
    aligned_depth_frame = frames.get_depth_frame()
    filled_depth = hole_filling.process(aligned_depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    depth = np.asanyarray(filled_depth.get_data())
    return im0s, depth, colorized_depth
def detects_vidio(pipe,hole_filling,colorizer):
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    # if not color_frame:
    #     continue

    im0s = np.asanyarray(color_frame.get_data())
    # colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    # 获取对齐更新后的深度图
    aligned_depth_frame = frames.get_depth_frame()
    filled_depth = hole_filling.process(aligned_depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    # depth = np.asanyarray(aligned_depth_frame.get_data())
    depth = np.asanyarray(filled_depth.get_data())
    return im0s,depth,colorized_depth

def detect():
    source, weights, view_img, imgsz,save_vidio,detect_image,detect_vidio,webcam,dist_thres_lim = opt.source, opt.weights, opt.view_img,\
                                                                          opt.img_size,opt.save_vidio,opt.detect_img,opt.detect_vidio,\
                                                                                                  opt.detect_webcam,opt.dist_thres_lim
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
    if not webcam:
        cfg.enable_device_from_file(file_name=source)
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
    colorizer = rs.colorizer()
    hole_filling = rs.hole_filling_filter()
    try:
        while True:
            if detect_image:
                im0s, depth, colorized_depth=detects_img(pipe,hole_filling,colorizer)
            if detect_vidio:
                im0s, depth, colorized_depth=detects_vidio(pipe,hole_filling,colorizer)
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

                all_coords = []
            # t3 = time.time()
            for _,det in enumerate(pred):
                if len(det):
                    det=det.cpu()
                        # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    depth_temp = depth[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])].astype(float)
                    depth_temp = depth_temp * 0.001
                    dist, _, _, _ = cv2.mean(depth_temp)
                    dist_temp = Decimal(dist).quantize(Decimal('0.000'))
                    y_mid = (int(xyxy[1]) + int(xyxy[3])) / 2
                    x_mid = (xyxy[0] + xyxy[2]) / 2
                    len_x = np.cumsum(depth[int(y_mid), int(xyxy[0]):int(xyxy[2])])[-1] * 0.00165
                    len_y = np.cumsum(depth[int(xyxy[1]):int(xyxy[3]), int(x_mid)])[-1] * 0.00165
                    label1 = str(dist_temp) + "m" + str(int(len_x)) + "mm" + ',' + str(int(len_y)) + "mm"
                    all_coords.append(xyxy)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)
                    plot_dots_on_people(xyxy, im0s)
                    plot_one_box(xyxy, colorized_depth, label=label1, color=colors[int(cls)],line_thickness=2)
                    distancing_all(all_coords, im0s,depth=depth,dist_thres_lim=dist_thres_lim)

            images = np.hstack((cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR), cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR)))
            a+=1
            # if a %1 ==0:
            print("\r>>>FPS:{:.2f}<<<     ".format(a/(time.time()-start)),end="")
            if view_img:
                cv2.imshow("RealSense", images)
                # t6 = time.time()
                # print("cv2imshow()  time :" + str(t6 - t4))
                # 按下“q”键停止q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     # cv2.destroyAllWindows()
                     break
            if detect_image:
                cv2.imwrite("out_img.png",images)
                break
            if  save_vidio is not None and a!=10000:
                fps, w, h = 30, images.shape[1], images.shape[0]
                # save_path += '.mp4'
                if a==1:
                    vid_writer = cv2.VideoWriter(save_vidio+"test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(images)
                # print("\r video [{}/{}] save path{} ".format(a,300,save_vidio),end="")
            else:
                print(" Done,vidio save to{}, time{:.2f}".format(save_vidio,time.time()-start))
                # vid_writer.release()
                cv2.destroyAllWindows()
                break
            # t7 =time.time()
            # print('cv2.waitKey()' + str(t7 - t6))

    finally:
        # pipe.stop()
        pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source',default='20210422_134345.bag', type=str, help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)') #img size is 320 *n
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',default=True, action='store_true', help='display results')
    parser.add_argument('--classes',nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--save_vidio', default="./vidio/", help='if default ="",not save vidio')
    parser.add_argument('--detect_img', default=False, help='detect image')
    parser.add_argument('--detect_vidio', default=True, help="detect vidio")
    parser.add_argument('--detect_webcam', default=True, help="detect webcam")
    parser.add_argument('--dist_thres_lim', default=(2000,4000), help="safe distance")

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
