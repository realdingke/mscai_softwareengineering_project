import os.path as osp
import os
import json
import requests
import time
import cv2
import moviepy.editor as mpy  # pip install moviepy
import numpy as np

from src.cord_loader import mkdirs


class SeqReadError(Exception):
    pass


def download_mp4(path='/content/drive/MyDrive/car_data_MCMOT/', seqs=None):
    if seqs is None:
        raise SeqReadError('Error occured during reading seq_name from server')
    for s in seqs:
        with open(path + '/' + s + '/objects.json', "r") as f:
            data = json.load(f)
        mkdirs(path + "/images/train/" + s)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' + \
            ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
        }
        url = list(data['data_units'].values())[0]['data_link']
        movie_url = url
        movie_name = data['data_title']
        downsize = 0
        print('Starting download')
        startTime = time.time()
        req = requests.get(movie_url, headers=headers, stream=True, verify=False)
        with(open(path + "/images/train/" + s + '/' + movie_name, 'wb')) as f:
            for chunk in req.iter_content(chunk_size=10000):
                if chunk:
                    f.write(chunk)
                    downsize += len(chunk)
                    line = 'downloading %d KB/s - %.2f MB， in total %.2f MB'
                    line = line % (
                        downsize / 1024 / (time.time() - startTime), downsize / 1024 / 1024, downsize / 1024 / 1024)
                    print(line)


def download_pics(path='/content/drive/MyDrive/food_data/'):
    for s in os.listdir(path):
        with open(path + s + '/objects.json', "r") as f:
            data = json.load(f)
        mkdirs(path + "/images/train/" + s)
        for i, link in enumerate(data['data_units'].values()):
            url = link['data_link']
            r = requests.get(url, allow_redirects=True)
            with open(path + "/images/train/" + s + f"/frame{i}.jpg", 'wb') as f:
                f.write(r.content)


class LoadVideo:  # for inference
    def __init__(self,
                 save_root='/content/drive/MyDrive/car_data_MCMOT/images/train',
                 seq_name='Heavy_traffic.mp4'):
        video_name = seq_name.replace('_', ' ')
        self.path = osp.join(save_root, seq_name, video_name)
        self.save_path = osp.join(save_root, f"{seq_name}/img1")
        print(self.path)
        self.cap = cv2.VideoCapture(self.path)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        self.get_image_size()
        self.count = 0

        # self.w, self.h = 1920, 1080
        # print('Lenth of the video: {:d} frames'.format(self.vn))

    def __len__(self):
        return self.vn  # number of files

    def save_imgs(self):
        # ret, frame1 = cap.read()
        # ret, frame2 = cap.read()

        frame_idx = 0

        while frame_idx < self.vn:
            ret, frame1 = self.cap.read()

            mkdirs(self.save_path)

            if ret is True:
                cv2.imwrite(
                    (self.save_path + "/{:06d}.jpg".format(frame_idx + 1)),
                    frame1,
                )

            frame_idx += 1
        return None

    def get_image_size(self):
        read_success_flag = False

        while not read_success_flag:
            ret, frame1 = self.cap.read()
            if ret is True:
                self.height, self.width = frame1.shape[:2]
                read_success_flag = True

        return None

    def __iter__(self):
        self.count = -1
        return self


def _generate_seqinfo(video, seq, path):
    """generate seqinfo.ini file for current seq dataset"""
    width = video.width
    height = video.height
    length = video.vn
    frame_rate = video.frame_rate
    info = '[Sequence]\nname=' + seq + '\nimDir=img1\n' + \
           'frameRate={:.6f}\nseqLength={:d}\nimWidth={:d}\nimHeight={:d}\n'.format(
               frame_rate,
               length,
               width,
               height,
           ) + 'imExt=.jpg\n'
    with open(osp.join(path, 'images/train', seq, 'seqinfo.ini'), 'w') as f:
        f.write(info)


def _extract_images(path, save_dir):
    myclip = mpy.VideoFileClip(path)
    times = np.linspace(0, myclip.duration, num=round(myclip.duration * myclip.fps))
    frame_count = 1
    mkdirs(save_dir)
    for time in times:
        imgpath = osp.join(save_dir, "{:0>6d}.jpg".format(frame_count))
        myclip.save_frame(imgpath, time)
        if frame_count % 100 == 0:
            print(f"{frame_count} images generated")
        frame_count += 1
    print(f"{frame_count} images in total")


def save_mp4_frame_gen_seqini(seqs=None,
                              path='/content/drive/MyDrive/car_data_MCMOT/'):
    if seqs is None:
        seqs = [
            'Heavy_traffic.mp4',
            'Highway_traffic_2.mp4',
            'Highway_traffic.mp4',
            'Light_traffic.mp4',
        ]
    save_root = path + '/images/train'
    for seq in seqs:
        v1 = LoadVideo(save_root=save_root, seq_name=seq)
        # v1.save_imgs()    #save imgs func now being superceded by the moviepy method
        _extract_images(v1.path, v1.save_path)
        _generate_seqinfo(v1, seq, path)

# if __name__ == '__main__':
#    seqs = [
#        'Heavy_traffic.mp4', 
#        'Highway_traffic_2.mp4', 
#        'Highway_traffic.mp4', 
#        'Light_traffic.mp4',
#    ]
#     seqs = gen_seq_name_list(load_cord_data())
#     path = '/content/drive/MyDrive/car_data_MCMOT/'
#     # save_root = '/content/drive/MyDrive/car_data_MCMOT/images/train'
#     download_mp4(path, seqs)
#     # download_pics(path='/content/drive/MyDrive/food_data/') # for food
#     save_mp4_frame_gen_seqini(seqs, path)
