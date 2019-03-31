#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 14:47
# @Author  : tomoncle
# @Site    : https://github.com/tomoncle/face-detection-induction-course
# @File    : input_static_pic_to_gif2.py

# 参考信息：https://www.makeartwithpython.com/blog/deal-with-it-generator-face-recognition/
# 描述：
#       程序从命令行参数获取图片信息，然后，它将使用Dlib中的人脸检测算法来查看是否有人脸存在。
#       如果有，它将为每个人脸创建一个结束位置，眼镜和烟卷会移动到那里结束。
#
#       然后我们需要缩放和旋转我们的眼镜以适合每个人的脸。
#       我们将使用从Dlib的68点模型返回的点集来找到眼睛的中心，并为它们之间的空间旋转。
#
#       在我们找到眼镜的最终位置和旋转后，我们可以为gif制作动画，眼镜从屏幕顶部进入。
#       我们将使用MoviePy和一个自定义的FaceDetect工具类绘制它。
#
#       同理烟卷也是这样。
#
#       应用程序的体系结构非常简单。我们首先接收图片，然后将其转换为灰度NumPy数组。
#       假如没有人脸，程序会自己退出，如果存在，我们就可以将检测到的人脸信息传递到人脸方向预测模型中。
#
#       通过返回的脸部方向，我们可以选择眼睛，缩放和旋转我们的眼镜框架以适合人的面部大小。
#
#       当然这个程序不仅仅只针对于一张人脸，可以检测多个人脸信息。
#
#       最后，通过获取的人脸列表，我们可以使用MoviePy创建一个绘图，然后生成我们的动画gif。


import moviepy.editor as mpy
import numpy as np
from PIL import Image
from imutils import face_utils

try:
    from dlib import get_frontal_face_detector, shape_predictor
except ImportError:
    raise


class FaceDetect(object):
    def __init__(self, img_src, gif_path=None):
        self.gif_max_width = 500
        self.duration = 4
        self.image = self.load(img_src).convert('RGBA')
        self.img_gray = None
        self.rects = None
        self.deal = None
        self.text = None
        self.cigarette = None
        if not self.validate:
            print("没有检测到人脸，程序退出.")
            exit(1)
        self.init_mask()
        self.make_gif(gif_path=gif_path)

    @property
    def validate(self):
        """
        验证是否存在人脸，如果不存在返回False
        :return:
        """
        if self.image.size[0] > self.gif_max_width:
            scaled_height = int(self.gif_max_width * self.image.size[1] / self.image.size[0])
            self.image.thumbnail((self.gif_max_width, scaled_height))
        self.img_gray = np.array(self.image.convert('L'))
        self.rects = self.detector(self.img_gray, 0)
        return len(self.rects) > 0

    @classmethod
    def load(cls, img_src):
        """
        加载图片转为Image对象
        :param img_src:
        :return:
        """
        return Image.open(img_src)

    @property
    def detector(self):
        """
        检测是否有人脸
        :return:
        """
        return get_frontal_face_detector()

    @property
    def predictor(self):
        """
        预测我们的面部方向
        :return:
        """
        return shape_predictor('shape_predictor_68_face_landmarks.dat')

    def init_mask(self):
        """
        加载面具
        :return:
        """
        self.deal, self.text, self.cigarette = (
            self.load(x) for x in ["../images/deals.png", "../images/text.png", "../images/cigarette.png"]
        )

    def get_glasses_info(self, face_shape, face_width):
        """
        获取当前面部的眼镜信息
        :param face_shape:
        :param face_width:
        :return:
        """
        left_eye = face_shape[36:42]
        right_eye = face_shape[42:48]

        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

        y = left_eye_center[1] - right_eye_center[1]
        x = left_eye_center[0] - right_eye_center[0]
        eye_angle = np.rad2deg(np.arctan2(y, x))

        deal = self.deal.resize(
            (face_width, int(face_width * self.deal.size[1] / self.deal.size[0])),
            resample=Image.LANCZOS)

        deal = deal.rotate(eye_angle, expand=True)
        deal = deal.transpose(Image.FLIP_TOP_BOTTOM)

        left_eye_x = left_eye[0, 0] - face_width // 4
        left_eye_y = left_eye[0, 1] - face_width // 6

        return {"image": deal, "pos": (left_eye_x, left_eye_y)}

    def get_cigarette_info(self, face_shape, face_width):
        """
        获取当前面部的烟卷信息
        :param face_shape:
        :param face_width:
        :return:
        """
        mouth = face_shape[49:68]
        mouth_center = mouth.mean(axis=0).astype("int")

        cigarette = self.cigarette.resize(
            (face_width, int(face_width * self.cigarette.size[1] / self.cigarette.size[0])),
            resample=Image.LANCZOS)

        x = mouth[0, 0] - face_width + int(16 * face_width / self.cigarette.size[0])
        y = mouth_center[1]
        return {"image": cigarette, "pos": (x, y)}

    def orientation(self):
        """
        人脸定位
        :return:
        """
        faces = []
        for rect in self.rects:
            face = {}
            face_shades_width = rect.right() - rect.left()
            predictor_shape = self.predictor(self.img_gray, rect)
            face_shape = face_utils.shape_to_np(predictor_shape)

            face['cigarette'] = self.get_cigarette_info(face_shape, face_shades_width)
            face['glasses'] = self.get_glasses_info(face_shape, face_shades_width)

            faces.append(face)

        return faces

    def drawing(self, t):
        """
        动态画图
        :param t:
        :return:
        """
        draw_img = self.image.convert('RGBA')
        if t == 0:
            return np.asarray(draw_img)

        for face in self.orientation():
            if t <= self.duration - 2:
                current_x = int(face["glasses"]["pos"][0])
                current_y = int(face["glasses"]["pos"][1] * t / (self.duration - 2))
                draw_img.paste(face["glasses"]["image"], (current_x, current_y), face["glasses"]["image"])

                cigarette_x = int(face["cigarette"]["pos"][0])
                cigarette_y = int(face["cigarette"]["pos"][1] * t / (self.duration - 2))
                draw_img.paste(face["cigarette"]["image"], (cigarette_x, cigarette_y), face["cigarette"]["image"])
            else:
                draw_img.paste(face["glasses"]["image"], face["glasses"]["pos"], face["glasses"]["image"])
                draw_img.paste(face["cigarette"]["image"], face["cigarette"]["pos"], face["cigarette"]["image"])
                draw_img.paste(self.text, (75, draw_img.height // 2 + 128), self.text)

        return np.asarray(draw_img)

    def make_gif(self, gif_path=None):
        """
        :param gif_path: 保存路径
        :return:
        """
        gif_path = gif_path or "deal.gif"
        animation = mpy.VideoClip(self.drawing, duration=self.duration)
        animation.write_gif(gif_path, fps=self.duration)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-image", required=True, help="path to input image")
    parser.add_argument("-save", required=False, default="deal.gif", help="path to output image")
    args = parser.parse_args()
    FaceDetect(args.image, args.save)
