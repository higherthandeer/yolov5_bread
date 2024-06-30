# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s_m_train_coco.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s_m_train_coco.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse  # 解析命令行参数的库
import os  # 与操作系统进行交互的文件库，包含文件路径操作与解析
import platform  # 用于获取操作系统相关信息，根据不同的操作系统执行不同的操作
import sys  # sys模块包含了与python解释器和他的环境有关的函数
from pathlib import Path  # Path能够更加方便地对字符串路径进行处理

import torch

FILE = Path(__file__).resolve()  # 获取当前文件的绝对路径 D://yolov5-master/train.py
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 权重文件  best.pt为训练好的权重文件
        source=ROOT / 'data/images',  # 测试图片所在的文件夹或文件路径
        data=ROOT / 'bread/bread_parameter.yaml',  # 含有数据集相关信息的dataset.yaml所在的路径
        imgsz=(640, 640),  # 预测推理时图片的大小(height, width)
        conf_thres=0.25,  # 置信度值
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 每张图最大检测数量，默认是最多检测1000个目标
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # # 检测的时候是否实时的把检测结果显示出来，默认False。
        save_txt=False,  # 是否把检测结果保存成一个.txt的YOLO格式，默认False。
        save_conf=False,  #  上面保存的txt中是否包含置信度，默认False。
        save_crop=False,  #  是否把模型检测的物体裁剪下来，默认False。
        nosave=False,  #  不保存预测的结果，默认False。
        classes=None,  # 指定检测某几种类别。比如coco128.yaml中person是第一个类别，classes指定“0”，则表示只检测图片中的person。
        agnostic_nms=False,  # 跨类别nms。比如待检测图像中有一个长得很像排球的足球，pt文件的分类中有足球和排球两种，那在识别时这个足球可能会被同时框上2个框：一个是足球，一个是排球。开启agnostic-nms后，那只会框出一个框，默认False。
        augment=False,  #  数据增强，默认False。
        visualize=False,  #  是否可视化特征图。如果开启了这和参数可以看到exp文件夹下又多了一些文件，这里.npy格式的文件就是保存的模型文件，可以使用numpy读写。还有一些png文件，默认False。
        update=False,  # update：如果指定这个参数，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认False。
        project=ROOT / 'runs/detect',  # # 预测结果保存的路径
        name='exp',  # # 预测结果保存文件夹名
        exist_ok=False,  # 每次预测模型的结果是否保存在原来的文件夹。如果指定了这个参数的话，那么本次预测的结果还是保存在上一次保存的文件夹里；如果不指定就是每次预测结果保存一个新的文件夹下，默认False。
        line_thickness=3,  # 调节预测框线条粗细的，default=3。
        hide_labels=False,  # 隐藏预测图片上的标签（只有预测框），默认False。
        hide_conf=False,  #  隐藏置信度（还有预测框和类别信息，但是没有置信度），默认False。
        half=False,  #  是否使用 FP16 半精度推理，默认False。在training阶段，梯度的更新往往是很微小的，需要相对较高的精度，一般要用到FP32以上。在inference的时候，精度要求没有那么高，一般F16（半精度）就可以，甚至可以用INT8（8位整型），精度影响不会很大。同时低精度的模型占用空间更小了，有利于部署在嵌入式模型里面。
        dnn=False,  # 是否使用 OpenCV DNN 进行 ONNX 推理，默认False
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)  # 测试图片所在的文件夹或文件路径
    save_img = not nosave and not source.endswith('.txt')  # 是否保存图片和txt文件，如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    # 判断source是不是视频/图像文件路径
    # Path()提取文件名。suffix：最后一个组件的文件扩展名。若source是"D://YOLOv5/data/1.jpg",则Path(source).suffix是".jpg",Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 是否为图片文件或视频文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 是否为网络地址即是否为链接
    # isnumeric函数判断是否是一个数字，是一个数字即表明传入的是电脑上的摄像头路径例0：表明打开电脑上第一个摄像头
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  # 是否为网络摄像头
    screenshot = source.lower().startswith('screen')  # 是否为屏幕截图
    if is_url and is_file:  # 如果是网络流则开始下载文件
        source = check_file(source)  # 下载url文件

    # 保存目录路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 三元表达式形式
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 读取模型
    device = select_device(device)
    # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，
    # weights: 模型的权重路径（比如yolov5s.pt） device: 设备 dnn: 是否使用OpenCV data:数据模型如data.yaml fp16: 是否使用半精度浮点数进行推理
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    '''
          stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
          names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...] 
          pt: 加载的是否是pytorch模型（也就是pt格式的文件）
    '''
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader数据读取器
    bs = 1  # batch_size
    if webcam:  # 网络摄像头
        view_img = check_imshow(warn=True)  # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载输入流
        bs = len(dataset)  # batch_size大小
    elif screenshot:  # 屏幕截图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:  # 除了网络摄像头、屏幕截图，其他执行以下语句
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 保存视频的路径，初始化为长度为batch_size的空列表
    vid_path, vid_writer = [None] * bs, [None] * bs  # 前者是视频路径,后者是一个cv2.VideoWriter对象
    # print('vid_path, vid_writer',vid_path, vid_writer)

    # Run inference 预测推理 热身部分
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup，模型预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # dt: 存储每一步骤的耗时 seen: 计数功能，已经处理完了多少帧图片
    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集
        '''
        在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
        path：文件路径（即source）
        im: resize后的图片（经过了放缩操作）
        im0s: 原始图片
        vid_cap=none
        s： 图片的基本信息，比如路径，大小
        '''
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 将numpy变为torch张量，将图片放到指定设备(如GPU)上识别。#torch.size=[3,640,480]480？？？
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 把输入从整型转化为半精度/全精度浮点数
            im /= 255  # 0 - 255 to 0.0 - 1.0 归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 扩展维度[channel,width,height]->[None,channel,width,height]缺少batch这个尺寸

        # Inference推理
        with dt[1]:
            # 可视化文件路径。如果为True则保留推理过程中的特征图，保存在runs文件夹中
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 推理结果，pred保存的是所有的bound_box的信息，
            pred = model(im, augment=augment, visualize=visualize)  # 直接调用了forward()

        # NMS非极大抑制，返回值为过滤后的预测框，除去多余的框
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        '''
        pred: 网络的输出结果
        conf_thres： 置信度阈值
        iou_thres： iou阈值
        classes: 是否只保留特定的类别 默认为None
        agnostic_nms： 进行nms是否也去除不同类别之间的框
        max_det: 检测框结果的最大数量 默认1000
        '''
        # print('pred',pred)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions 预测过程
        # 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image i:每个batch的信息，det: 表示5个检测框的信息
            seen += 1
            if webcam:  # batch_size >= 1
                # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '  # s后面拼接一个字符串i
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                '''
                大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                s: 输出信息 初始为 ''
                im0: 原始图片 letterbox + pad 之前的图片
                frame: 视频流,此次取的是第几张图片
                '''
            # print('p, im0, frame',p, im0, frame)

            p = Path(p)
            # print('p',p)
            save_path = str(save_dir / p.name)  # im.jpg # to Path # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\fire.jpg
            # print('save_path',save_path)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            # print('txt_path',txt_path)
            s += '%gx%g ' % im.shape[2:]  # print string # 设置输出图片信息。图片shape (w, h)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh # 得到原图的宽和高

            # 保存截图。如果save_crop的值为true，则将检测到的bounding_box单独保存成一张图片。
            imc = im0.copy() if save_crop else im0  # for save_crop

            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # print('annotator',annotator)

            # 判断有没有框
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）此时坐标格式为xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results打印结果
                # Print results
                # 打印检测到的类别数量
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results写入结果
                # 保存预测结果：txt/图片画框/crop-image
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line的形式是： ”类别 x y w h“，若save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下，在原图像画图或者保存结果
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class # 类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 类别名
                        annotator.box_label(xyxy, label, color=colors(c, True))  #绘制边框
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # 如果设置展示，则show图片 / 视频
            im0 = annotator.result()  # im0是绘制好的图片
            # 显示图片
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond # 暂停 1 millisecond

            # Save results (image with detections)
            # 设置保存图片/视频
            if save_img:  # 如果save_img为true,则保存绘制完的图片
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video vid_path[i] != save_path,说明这张图片属于一段新的视频,需要重新创建视频文件
                        vid_path[i] = save_path
                        # 以下的部分是保存视频文件
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 视频帧速率 FPS
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧高度
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')   # 指定预训练权重路径；如果这里设置为空的话，就是自己从头开始进行训练
    parser.add_argument('--source', type=str, default=ROOT /'datasets/video', help='file/dir/URL/glob/screen/0(webcam)')  # 测试集文件/文件夹：1.图片/视频路径，2.'0'（电脑自带摄像头），3.rtsp等视频流，默认data/images
    parser.add_argument('--data', type=str, default=ROOT / 'data/data.yaml', help='(optional) dataset.yaml path')  # 数据集对应的yaml参数文件；里面主要存放数据集的类别和路径信息
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')  # 输入网络的图片大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')  # 置信度的阈值。超过这个阈值的预测框就会被预测出来。比如conf-thres参数依次设置成“0”, “0.25”，“0.8”
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')  # iou阈值，NMS时用到
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')  # 每张图最大检测数量，默认是最多检测1000个目标
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 推理使用的硬件设备
    parser.add_argument('--view-img', action='store_true', help='show results')  # 检测的时候是否实时的把检测结果显示出来
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')  # 是否把检测结果保存成一个.txt的格式。txt默认保存物体的类别索引和预测框坐标（YOLO格式），每张图一个txt，txt中每行表示一个物体
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')  # 上面保存的txt中是否包含置信度
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')  # 是否把模型检测的物体裁剪下来，开启了这个参数会在crops文件夹下看到几个以类别命名的文件夹，里面保存的都是裁剪下来的图片。
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')  # 不保存预测的结果。但是还会生成exp文件夹，只不过是一个空的exp。这个参数应该是和“–view-img”配合使用的
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')  # 指定检测某几种类别。比如coco128.yaml中person是第一个类别，classes指定“0”，则表示只检测图片中的person。
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # 跨类别nms。比如待检测图像中有一个长得很像排球的足球，pt文件的分类中有足球和排球两种，那在识别时这个足球可能会被同时框上2个框：一个是足球，一个是排球。开启agnostic-nms后，那只会框出一个框
    parser.add_argument('--augment', action='store_true', help='augmented inference')  # 数据增强。
    parser.add_argument('--visualize', action='store_true', help='visualize features')  # 是否可视化特征图。如果开启了这和参数可以看到exp文件夹下又多了一些文件，这里.npy格式的文件就是保存的模型文件，可以使用numpy读写。还有一些png文件。
    parser.add_argument('--update', action='store_true', help='update all models')  # update：如果指定这个参数，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')  # 预测结果保存的路径
    parser.add_argument('--name', default='exp', help='save results to project/name')  # 预测结果保存文件夹名
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')  # 每次预测模型的结果是否保存在原来的文件夹。如果指定了这个参数的话，那么本次预测的结果还是保存在上一次保存的文件夹里；如果不指定就是每次预测结果保存一个新的文件夹下
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')  # ine-thickness：调节预测框线条粗细的，default=3。因为有的时候目标重叠太多会产生遮挡
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # 隐藏预测图片上的标签（只有预测框）
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # 隐藏置信度（还有预测框和类别信息，但是没有置信度）
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')  # 是否使用 FP16 半精度推理。在training阶段，梯度的更新往往是很微小的，需要相对较高的精度，一般要用到FP32以上。在inference的时候，精度要求没有那么高，一般F16（半精度）就可以，甚至可以用INT8（8位整型），精度影响不会很大。同时低精度的模型占用空间更小了，有利于部署在嵌入式模型里面。
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')  # 是否使用 OpenCV DNN 进行 ONNX 推理。
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride') # 
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # 检查环境/打印参数,主要是requrement.txt的包是否安装，用彩色显示设置的参数
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    """
    # 命令使用
    # python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source  data/images/fishman.jpg # webcam(摄像头)
    """
    opt = parse_opt()
    main(opt)
