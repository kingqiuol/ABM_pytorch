# -*- coding: utf-8 -*-
import cv2
import abc
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw


def cv2pil(image):
	"""
	将bgr格式的numpy的图像转换为pil
	:param image:   图像数组
	:return:    Image对象
	"""
	assert isinstance(image, np.ndarray), 'input image type is not cv2'
	if len(image.shape) == 2:
		return Image.fromarray(image)
	elif len(image.shape) == 3:
		return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_pil_image(image):
	"""
	将图像统一转换为PIL格式
	:param image:   图像
	:return:    Image格式的图像
	"""
	if isinstance(image, Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
		return image
	elif isinstance(image, np.ndarray):
		return cv2pil(image)


def get_cv_image(image):
	"""
	将图像转换为numpy格式的数据
	:param image:   图像
	:return:    ndarray格式的图像数据
	"""
	if isinstance(image, np.ndarray):
		return image
	elif isinstance(image, Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
		return pil2cv(image)


def pil2cv(image):
	"""
	将Image对象转换为ndarray格式图像
	:param image:   图像对象
	:return:    ndarray图像数组
	"""
	if len(image.split()) == 1:
		return np.asarray(image)
	elif len(image.split()) == 3:
		return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
	elif len(image.split()) == 4:
		return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)


class TransBase(object):
	"""
	数据增广的基类
	"""

	def __init__(self, probability=1.):
		"""
		初始化对象
		:param probability:     执行概率
		"""
		super(TransBase, self).__init__()
		self.probability = probability

	@abc.abstractmethod
	def trans_function(self, _image):
		"""
		初始化执行函数，需要进行重载
		:param _image:  待处理图像
		:return:    执行后的Image对象
		"""
		pass

	# @utils.zlog
	def process(self, _image):
		"""
		调用执行函数
		:param _image:  待处理图像
		:return:    执行后的Image对象
		"""
		if np.random.random() < self.probability:
			return self.trans_function(_image)
		else:
			return _image

	def __call__(self, _image):
		"""
		重载()，方便直接进行调用
		:param _image:  待处理图像
		:return:    执行后的Image
		"""
		return self.process(_image)


class RandomContrast(TransBase):
	"""
	随机对比度
	"""

	def setparam(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, _image):
		_image = get_pil_image(_image)
		contrast_enhance = ImageEnhance.Contrast(_image)
		return contrast_enhance.enhance(random.uniform(self.lower, self.upper))


class RandomLine(TransBase):
	"""
	在图像增加一条简单的随机线
	"""

	def trans_function(self, image):
		image = get_pil_image(image)
		draw = ImageDraw.Draw(image)
		h = image.height
		w = image.width
		y0 = random.randint(h // 4, h * 3 // 4)
		y1 = np.clip(random.randint(-3, 3) + y0, 0, h - 1)
		color = random.randint(0, 30)
		draw.line(((0, y0), (w - 1, y1)), fill=(color, color, color), width=2)
		return image


class RandomBrightness(TransBase):
	"""
	随机对比度
	"""

	def setparam(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, image):
		image = get_pil_image(image)
		bri = ImageEnhance.Brightness(image)
		return bri.enhance(random.uniform(self.lower, self.upper))


class RandomColor(TransBase):
	"""
	随机色彩平衡
	"""

	def setparam(self, lower=0.5, upper=1.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, image):
		image = get_pil_image(image)
		col = ImageEnhance.Color(image)
		return col.enhance(random.uniform(self.lower, self.upper))


class RandomSharpness(TransBase):
	"""
	随机锐度
	"""

	def setparam(self, lower=0.1, upper=2.5):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, image):
		image = get_pil_image(image)
		sha = ImageEnhance.Sharpness(image)
		return sha.enhance(random.uniform(self.lower, self.upper))


class Compress(TransBase):
	"""
	随机压缩率，利用jpeg的有损压缩来增广
	"""

	def setparam(self, lower=5, upper=85):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, image):
		img = get_cv_image(image)
		param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.lower, self.upper)]
		img_encode = cv2.imencode('.jpeg', img, param)
		img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
		pil_img = cv2pil(img_decode)
		if len(image.split()) == 1:
			pil_img = pil_img.convert('L')
		return pil_img


class Exposure(TransBase):
	"""
	随机区域曝光
	"""

	def setparam(self, lower=1, upper=20):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, image):
		image = get_cv_image(image)
		h, w = image.shape[:2]
		x0 = random.randint(0, w)
		y0 = random.randint(0, h)
		x1 = random.randint(x0, w)
		y1 = random.randint(y0, h)
		transparent_area = (x0, y0, x1, y1)
		mask = Image.new('L', (w, h), color=255)
		draw = ImageDraw.Draw(mask)
		mask = np.array(mask)
		if len(image.shape) == 3:
			mask = mask[:, :, np.newaxis]
			mask = np.concatenate([mask, mask, mask], axis=2)
		draw.rectangle(transparent_area, fill=random.randint(200, 255))
		reflection_result = image + (255 - mask)
		reflection_result = np.clip(reflection_result, 0, 255)
		return cv2pil(reflection_result)


class Rotate(TransBase):
	"""
	随机旋转
	"""

	def setparam(self, lower=-5, upper=5, fillcolor=None):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."

		self.fillcolor = fillcolor

	def trans_function(self, image):
		image = get_pil_image(image)
		rot = random.uniform(self.lower, self.upper)
		trans_img = image.rotate(rot, expand=True, fillcolor=self.fillcolor)
		return trans_img


class Blur(TransBase):
	"""
	随机高斯模糊
	"""

	def setparam(self, lower=0, upper=1):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "upper must be >= lower."
		assert self.lower >= 0, "lower must be non-negative."

	def trans_function(self, image):
		image = get_pil_image(image)
		radius = random.randint(self.lower, self.upper)
		image = image.filter(ImageFilter.GaussianBlur(radius=radius))
		return image


class MotionBlur(TransBase):
	"""
	随机运动模糊
	"""

	def setparam(self, degree=5, angle=180):
		self.degree = degree
		self.angle = angle

	def trans_function(self, image):
		image = get_pil_image(image)
		angle = random.randint(0, self.angle)
		M = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), angle, 1)
		motion_blur_kernel = np.diag(np.ones(self.degree))
		motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
		motion_blur_kernel = motion_blur_kernel / self.degree
		image = image.filter(ImageFilter.Kernel(size=(self.degree, self.degree), kernel=motion_blur_kernel.reshape(-1)))
		return image


class Salt(TransBase):
	"""
	随机椒盐噪音
	"""

	def setparam(self, rate=0.01):
		self.rate = rate

	def trans_function(self, image):
		image = get_pil_image(image)
		num_noise = int(image.size[1] * image.size[0] * self.rate)
		# assert len(image.split()) == 1
		for k in range(num_noise):
			i = int(np.random.random() * image.size[1])
			j = int(np.random.random() * image.size[0])
			value = int(np.random.random() * 255)
			image.putpixel((j, i), (value, value, value))
		return image


class AdjustResolution(TransBase):
	"""
	随机分辨率
	"""

	def setparam(self, max_rate=0.99, min_rate=0.5):
		self.max_rate = max_rate
		self.min_rate = min_rate

	def trans_function(self, image):
		image = get_pil_image(image)
		w, h = image.size
		rate = np.random.random() * (self.max_rate - self.min_rate) + self.min_rate
		w2 = int(w * rate)
		h2 = int(h * rate)
		image = image.resize((w2, h2))
		image = image.resize((w, h))
		return image


class Crop(TransBase):
	"""
	抠随机图，并且抠图区域透视变换为原图大小
	"""

	def setparam(self, maxv=10):
		self.maxv = maxv

	def trans_function(self, image):
		img = get_cv_image(image)
		h, w = img.shape[:2]
		org = np.array([[0, np.random.randint(0, self.maxv)],
						[w, np.random.randint(0, self.maxv)],
						[0, h - np.random.randint(0, self.maxv)],
						[w, h - np.random.randint(0, self.maxv)]], np.float32)
		dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
		M = cv2.getPerspectiveTransform(org, dst)
		res = cv2.warpPerspective(img, M, (w, h))
		return get_pil_image(res)


class Crop2(TransBase):
	"""
	随机抠图，并且抠图区域透视变换为原图大小
	"""

	def setparam(self, maxv_h=4, maxv_w=4):
		self.maxv_h = maxv_h
		self.maxv_w = maxv_w

	def trans_function(self, image_and_loc):
		image, left, top, right, bottom = image_and_loc
		w, h = image.size
		left = np.clip(left, 0, w - 1)
		right = np.clip(right, 0, w - 1)
		top = np.clip(top, 0, h - 1)
		bottom = np.clip(bottom, 0, h - 1)
		img = get_cv_image(image)
		try:
			res = get_pil_image(img[top:bottom, left:right])
			return res
		except AttributeError as e:
			print('error')
			image.save('test_imgs/t.png')
			print(left, top, right, bottom)

		h = bottom - top
		w = right - left
		org = np.array(
			[[left - np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h // 2)],
			 [right + np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h // 2)],
			 [left - np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h // 2)],
			 [right + np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h // 2)]],
			np.float32)
		dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
		M = cv2.getPerspectiveTransform(org, dst)
		res = cv2.warpPerspective(img, M, (w, h))
		return get_pil_image(res)


class Stretch(TransBase):
	"""
	随机图像横向拉伸
	"""

	def setparam(self, max_rate=1.5, min_rate=0.1):
		self.max_rate = max_rate
		self.min_rate = min_rate

	def trans_function(self, image):
		image = get_pil_image(image)
		w, h = image.size
		rate = np.random.random() * (self.max_rate - self.min_rate) + self.min_rate
		w2 = int(w * rate)
		image = image.resize((w2, h))
		return image


class TIADistort(TransBase):
	'''
	水平方向TIA图像变形增强
	'''

	def setparam(self):
		self.segment = random.randint(3, 6)

	def trans_function(self, _image):
		image = get_cv_image(_image)

		img_h, img_w = image.shape[:2]
		if img_h >= 20 and img_w >= 20:
			cut = img_w // self.segment
			thresh = cut // 3

			src_pts = list()
			dst_pts = list()

			src_pts.append([0, 0])
			src_pts.append([img_w, 0])
			src_pts.append([img_w, img_h])
			src_pts.append([0, img_h])

			dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
			dst_pts.append(
				[img_w - np.random.randint(thresh), np.random.randint(thresh)])
			dst_pts.append(
				[img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
			dst_pts.append(
				[np.random.randint(thresh), img_h - np.random.randint(thresh)])

			half_thresh = thresh * 0.5

			for cut_idx in np.arange(1, self.segment, 1):
				src_pts.append([cut * cut_idx, 0])
				src_pts.append([cut * cut_idx, img_h])
				dst_pts.append([
					cut * cut_idx + np.random.randint(thresh) - half_thresh,
					np.random.randint(thresh) - half_thresh
				])
				dst_pts.append([
					cut * cut_idx + np.random.randint(thresh) - half_thresh,
					img_h + np.random.randint(thresh) - half_thresh
				])

			trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
			dst = trans.generate()

			pil_img = cv2pil(dst)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img
		else:
			pil_img = cv2pil(image)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img


class TIAStretch(TransBase):
	'''
	竖直方向TIA图像变形增强
	'''

	def setparam(self):
		self.segment = random.randint(3, 6)

	def trans_function(self, _image):
		image = get_cv_image(_image)

		img_h, img_w = image.shape[:2]
		if img_h >= 20 and img_w >= 20:

			cut = img_w // self.segment
			thresh = cut * 4 // 5

			src_pts = list()
			dst_pts = list()

			src_pts.append([0, 0])
			src_pts.append([img_w, 0])
			src_pts.append([img_w, img_h])
			src_pts.append([0, img_h])

			dst_pts.append([0, 0])
			dst_pts.append([img_w, 0])
			dst_pts.append([img_w, img_h])
			dst_pts.append([0, img_h])

			half_thresh = thresh * 0.5

			for cut_idx in np.arange(1, self.segment, 1):
				move = np.random.randint(thresh) - half_thresh
				src_pts.append([cut * cut_idx, 0])
				src_pts.append([cut * cut_idx, img_h])
				dst_pts.append([cut * cut_idx + move, 0])
				dst_pts.append([cut * cut_idx + move, img_h])

			trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
			dst = trans.generate()

			pil_img = cv2pil(dst)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img
		else:
			pil_img = cv2pil(image)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img


class TIAPerspective(TransBase):
	'''
	多段多方向TIA图像变形增强
	'''

	def setparam(self):
		self.segment = random.randint(3, 6)

	def trans_function(self, _image):
		image = get_cv_image(_image)

		img_h, img_w = image.shape[:2]
		if img_h >= 20 and img_w >= 20:

			thresh = img_h // 2

			src_pts = list()
			dst_pts = list()

			src_pts.append([0, 0])
			src_pts.append([img_w, 0])
			src_pts.append([img_w, img_h])
			src_pts.append([0, img_h])

			dst_pts.append([0, np.random.randint(thresh)])
			dst_pts.append([img_w, np.random.randint(thresh)])
			dst_pts.append([img_w, img_h - np.random.randint(thresh)])
			dst_pts.append([0, img_h - np.random.randint(thresh)])

			trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
			dst = trans.generate()

			pil_img = cv2pil(dst)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img
		else:
			pil_img = cv2pil(image)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img


class WarpMLS:
	'''wrap仿射变换'''

	def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
		self.src = src
		self.src_pts = src_pts
		self.dst_pts = dst_pts
		self.pt_count = len(self.dst_pts)
		self.dst_w = dst_w
		self.dst_h = dst_h
		self.trans_ratio = trans_ratio
		self.grid_size = 100
		self.rdx = np.zeros((self.dst_h, self.dst_w))
		self.rdy = np.zeros((self.dst_h, self.dst_w))

	@staticmethod
	def __bilinear_interp(x, y, v11, v12, v21, v22):
		return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
													  (1 - y) + v22 * y) * x

	def generate(self):
		self.calc_delta()
		return self.gen_img()

	def calc_delta(self):
		w = np.zeros(self.pt_count, dtype=np.float32)

		if self.pt_count < 2:
			return

		i = 0
		while 1:
			if self.dst_w <= i < self.dst_w + self.grid_size - 1:
				i = self.dst_w - 1
			elif i >= self.dst_w:
				break

			j = 0
			while 1:
				if self.dst_h <= j < self.dst_h + self.grid_size - 1:
					j = self.dst_h - 1
				elif j >= self.dst_h:
					break

				sw = 0
				swp = np.zeros(2, dtype=np.float32)
				swq = np.zeros(2, dtype=np.float32)
				new_pt = np.zeros(2, dtype=np.float32)
				cur_pt = np.array([i, j], dtype=np.float32)

				k = 0
				for k in range(self.pt_count):
					if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
						break

					w[k] = 1. / (
							(i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
							(j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

					sw += w[k]
					swp = swp + w[k] * np.array(self.dst_pts[k])
					swq = swq + w[k] * np.array(self.src_pts[k])

				if k == self.pt_count - 1:
					pstar = 1 / sw * swp
					qstar = 1 / sw * swq

					miu_s = 0
					for k in range(self.pt_count):
						if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
							continue
						pt_i = self.dst_pts[k] - pstar
						miu_s += w[k] * np.sum(pt_i * pt_i)

					cur_pt -= pstar
					cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

					for k in range(self.pt_count):
						if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
							continue

						pt_i = self.dst_pts[k] - pstar
						pt_j = np.array([-pt_i[1], pt_i[0]])

						tmp_pt = np.zeros(2, dtype=np.float32)
						tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
									np.sum(pt_j * cur_pt) * self.src_pts[k][1]
						tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
									np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
						tmp_pt *= (w[k] / miu_s)
						new_pt += tmp_pt

					new_pt += qstar
				else:
					new_pt = self.src_pts[k]

				self.rdx[j, i] = new_pt[0] - i
				self.rdy[j, i] = new_pt[1] - j

				j += self.grid_size
			i += self.grid_size

	def gen_img(self):
		src_h, src_w = self.src.shape[:2]
		dst = np.zeros_like(self.src, dtype=np.float32)

		for i in np.arange(0, self.dst_h, self.grid_size):
			for j in np.arange(0, self.dst_w, self.grid_size):
				ni = i + self.grid_size
				nj = j + self.grid_size
				w = h = self.grid_size
				if ni >= self.dst_h:
					ni = self.dst_h - 1
					h = ni - i + 1
				if nj >= self.dst_w:
					nj = self.dst_w - 1
					w = nj - j + 1

				di = np.reshape(np.arange(h), (-1, 1))
				dj = np.reshape(np.arange(w), (1, -1))
				delta_x = self.__bilinear_interp(
					di / h, dj / w, self.rdx[i, j], self.rdx[i, nj],
					self.rdx[ni, j], self.rdx[ni, nj])
				delta_y = self.__bilinear_interp(
					di / h, dj / w, self.rdy[i, j], self.rdy[i, nj],
					self.rdy[ni, j], self.rdy[ni, nj])
				nx = j + dj + delta_x * self.trans_ratio
				ny = i + di + delta_y * self.trans_ratio
				nx = np.clip(nx, 0, src_w - 1)
				ny = np.clip(ny, 0, src_h - 1)
				nxi = np.array(np.floor(nx), dtype=np.int32)
				nyi = np.array(np.floor(ny), dtype=np.int32)
				nxi1 = np.array(np.ceil(nx), dtype=np.int32)
				nyi1 = np.array(np.ceil(ny), dtype=np.int32)

				if len(self.src.shape) == 3:
					x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
					y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
				else:
					x = ny - nyi
					y = nx - nxi
				dst[i:i + h, j:j + w] = self.__bilinear_interp(
					x, y, self.src[nyi, nxi], self.src[nyi, nxi1],
					self.src[nyi1, nxi], self.src[nyi1, nxi1])

		dst = np.clip(dst, 0, 255)
		dst = np.array(dst, dtype=np.uint8)

		return dst


class ColorDisturbance(TransBase):
	'''
	随机颜色扰动
	'''

	def setparam(self, delta=0.001):
		self.delta = delta

	def trans_function(self, _image):
		image = get_cv_image(_image)

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		delta = 0.001 * random.random()
		hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
		new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		pil_img = cv2pil(new_image)
		if len(_image.split()) == 1:
			pil_img = pil_img.convert('L')
		return pil_img


class RandomJitter(TransBase):
	'''
	随机抖动
	'''

	def setparam(self, delta=0.1):
		self.delta = delta

	def trans_function(self, _image):
		image = get_cv_image(_image)

		w, h, _ = image.shape
		if h > 10 and w > 10:
			thres = min(w, h)
			s = int(random.random() * thres * self.delta)
			src_img = image.copy()
			for i in range(s):
				image[i:, i:, :] = src_img[:w - i, :h - i, :]

			pil_img = cv2pil(image)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img
		else:
			pil_img = cv2pil(image)
			if len(_image.split()) == 1:
				pil_img = pil_img.convert('L')
			return pil_img


class DataAug:
	def __init__(self):
		# 随机剪切
		self.crop = Crop(probability=0.1)
		self.crop2 = Crop2(probability=1.1)
		# 随机对比度增强
		self.random_contrast = RandomContrast(probability=0.2)
		# 随机亮度增强
		self.random_brightness = RandomBrightness(probability=0.2)
		# 随机颜色增强
		self.random_color = RandomColor(probability=0.2)
		# 随机锐化
		self.random_sharpness = RandomSharpness(probability=0.1)
		# 随机压缩率，利用jpeg的有损压缩来增广
		self.compress = Compress(probability=0.3)
		# 随机区域曝光
		self.exposure = Exposure(probability=0.2)
		# 随机旋转
		self.rotate = Rotate(probability=0.1)
		# 随机模糊
		self.blur = Blur(probability=0.2)
		# 随机动态模糊
		self.motion_blur = MotionBlur(probability=0.2)
		# 随机椒盐噪声
		self.salt = Salt(probability=0.1)
		# 随机分辨率
		self.adjust_resolution = AdjustResolution(probability=1)
		# 随机宽度拉伸
		self.stretch = Stretch(probability=0.1)

		# 随机线条
		self.random_line = RandomLine(probability=0.01)

		# 随机TIA变形
		self.tia_distort = TIADistort(probability=0.2)  # 宽度方向
		self.tia_stretch = TIAStretch(probability=0.2)  # 长度方向
		self.tia_perspective = TIAPerspective(probability=0.2)  # 多方向

		# 随机颜色扰动
		self.color_disturbance = ColorDisturbance(probability=0.2)

		# 随机抖动
		self.random_jitter = RandomJitter(probability=0.2)

		self.crop.setparam()
		self.crop2.setparam()
		self.random_contrast.setparam()
		self.random_brightness.setparam()
		self.random_color.setparam()
		self.random_sharpness.setparam()
		self.compress.setparam()
		self.exposure.setparam()
		self.rotate.setparam()
		self.blur.setparam()
		self.motion_blur.setparam()
		self.salt.setparam()
		self.adjust_resolution.setparam()
		self.stretch.setparam()

		self.tia_distort.setparam()
		self.tia_stretch.setparam()
		self.tia_perspective.setparam()

		self.color_disturbance.setparam()

		self.random_jitter.setparam()

	def aug_img(self, img):
		img = cv2pil(img)
		# img = self.crop.process(img)
		img = self.random_contrast.process(img)
		img = self.random_brightness.process(img)
		# img = self.random_color.process(img)
		img = self.random_sharpness.process(img)
		# img = self.random_line.process(img)
		if img.size[1] >= 32:
			img = self.compress.process(img)
			img = self.adjust_resolution.process(img)
			# img = self.motion_blur.process(img)
			img = self.blur.process(img)
		img = self.exposure.process(img)
		# img = self.rotate.process(img)
		img = self.salt.process(img)
		# img = self.inverse_color(img)
		# img = self.stretch.process(img)

		# img=self.tia_distort(img)
		# img=self.tia_stretch(img)
		img=self.tia_perspective(img)

		img=self.color_disturbance(img)

		img=self.random_jitter(img)


		img = pil2cv(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img

	def inverse_color(self, image):
		if np.random.random() < 0.4:
			image = ImageOps.invert(image)
		return image


if __name__ == '__main__':
	data_augment = DataAug()
	img = cv2.imread("vlcsnap-2021-08-31-09h24m55s817_1.jpg")
	imgsave = data_augment.aug_img(img)
	cv2.imwrite("result.jpg", imgsave)
