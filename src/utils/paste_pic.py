import cv2, os
import numpy as np
from tqdm import tqdm
import uuid
from concurrent.futures import ThreadPoolExecutor
from src.utils.videoio import save_video_with_watermark


def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False, max_threads=5):
    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        # loader for first frame
        full_img = cv2.imread(pic_path)
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            break
        full_img = frame
    frame_h = full_img.shape[0]
    frame_w = full_img.shape[1]

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)

    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx

    tmp_path = str(uuid.uuid4()) + '.mp4'

    # 原始代码 >>>>
    # out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))
    # for crop_frame in tqdm(crop_frames, 'seamlessClone:'):
    #     p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1))
    #
    #     mask = 255*np.ones(p.shape, p.dtype)
    #     location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
    #     gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
    #     out_tmp.write(gen_img)
    # <<<< 原始代码

    # 参考教程 https://github.com/OpenTalker/SadTalker/issues/520#issuecomment-1646495988
    # !!! 自定义开始 >>>
    # 可以通过增加线程池的办法并行处理加快
    def process_image(crop_frame):
        p = cv2.resize(crop_frame.astype(np.uint8), (ox2 - ox1, oy2 - oy1))

        mask = 255 * np.ones(p.shape, p.dtype)
        location = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)

        return gen_img

    tmp_path = str(uuid.uuid4()) + '.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))

    processed_frames = []  # 存储处理后的图像

    # 创建线程池
    # 指定线程池的最大线程数
    # max_threads = max_threads

    # 限定理论最大值
    limit_max_threads = os.cpu_count() + 4
    max_threads = max_threads if max_threads <= os.cpu_count() + 4 else limit_max_threads

    print(f"Seamless Clone 当前加速进程数：“{max_threads}”")
    print(f"程序感知到cpu数量为 {os.cpu_count()} 个")
    print(f"理论最大工作线程总数 = （系统中的 CPU） + 4 ，即 {limit_max_threads} 个")

    # 创建线程池并设置max_workers参数
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交任务并获取处理结果
        processed_frames = []
        for gen_img in tqdm(executor.map(process_image, crop_frames), total=len(crop_frames), desc='seamlessClone:'):
            processed_frames.append(gen_img)

    # 一次将所有处理后的图像写入视频文件
    for frame in processed_frames:
        out_tmp.write(frame)
    # <<<<<< 自定义修改结束 !!!

    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)
