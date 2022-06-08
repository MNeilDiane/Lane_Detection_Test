import numpy as np
import cv2

def roi_mask(img, vertices):#img 输入图像，vertices是感兴趣区域的四个点的坐标
    mask = np.zeros_like(img)
    print("mask_shape: ", mask.shape)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)#使用白色填充含兴趣区域，形成蒙版
    masked_img = cv2.bitwise_and(img, mask)# 只留下感兴趣区域的图像
    return masked_img

def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)
    #polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
    # img（array）：为ndarray类型（可以为cv.imread）直接读取的数据
    # pts（array）：为所画多边形的顶点坐标，举个简单的例子：当一张图片需要有多个四边形时，该数组ndarray的shape应该为（N，4，2）
    # isClosed（bool）：所画四边形是否闭合，通常为True
    # color（tuple）：RGB三个通道的值
    # thickness（int）：画线的粗细
    # shift：顶点坐标中小数的位数

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

#拟合点集,生成直线表达式，并计算直线在图像中的两个端点坐标
def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]#提取X
    y = [p[1] for p in point_list]#提取y
    fit = np.polyfit(y, x, 1)#用一次多项式x=ay+b拟合这些点，fit是（a，b）
    fit_fn = np.poly1d(fit)#生成多项式对象a*y+b

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]

#画出所需的道路线
def draw_lanes(img, lines, color=[255, 0, 0], thickness=6):
    left_lines, right_lines = [], []#存储左边与右边的直线
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2-y1)/(x2-x1)
            if k<0:#斜率小于0
                left_lines.append(line)
            else:
                right_lines.append(line)
    if(len(left_lines)<=0 or len(right_lines)<=0):
        return img

    clean_lines(left_lines, 0.1)#弹出不满足要求的直线
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]#提取左侧直线簇中所有第一个点
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]#提取左侧直线簇中所有第二个点
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, 148, img.shape[0])#拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
    right_vtx = calc_lane_vertices(right_points, 135, img.shape[0])#拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标

    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)

def clean_lines(lines, threshold):
    slope = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2-y2)/(x2-x1)
            slope.append(k)
    while len(lines)>0:
        mean = np.mean(slope)
        diff = [abs(s-mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx]>threshold:
            slope.pop(idx)#弹出斜率
            lines.pop(idx)#弹出直线
        else:
            break

def hough_lines(img, rho,theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    print(lines.shape)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)#生成绘制直线的绘图板，黑底
    draw_lines(line_img, lines)
    return line_img


img = cv2.imread('lane.jpg')
roi_vtx = np.array([[(0, img.shape[0]), (img.shape[1] / 2 - 20, img.shape[0] / 2 + 50),
                     (img.shape[1] / 2 + 20, img.shape[0] / 2 + 20), (img.shape[1], img.shape[0]), (0, 500),
                     (960, 500)]],
                   dtype=np.int32)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#转化3通道灰度图像
blur_gray = cv2.GaussianBlur(img, (5, 5), 0)#GaussianBlur(InputArray src, OutputArray dst,
# Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT);
edges = cv2.Canny(blur_gray, 10, 120)
#edges = cv.Canny( image, threshold1, threshold2[, apertureSize[, L2gradient]])
roi_edges = roi_mask(edges, roi_vtx)
line_image = hough_lines(roi_edges, 1, np.pi / 180, 5, 5, 80)
final = cv2.addWeighted(img, 0.8, line_image, 1, 0)
#cvAddWeighted( const CvArr* src1, double alpha,const CvArr* src2, double beta,double gamma, CvArr* dst );
#参数1：src1，第一个原数组.
#参数2：alpha，第一个数组元素权重
#参数3：src2第二个原数组
#参数4：beta，第二个数组元素权重
#参数5：gamma，图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了。
#参数6：dst，输出图片
cv2.imshow('1', gray)
cv2.imshow('2', blur_gray)
cv2.imshow('3', edges)
cv2.imshow('4', roi_edges)
cv2.imshow('5', line_image)
cv2.imshow('6', final)
cv2.waitKey(0)
cv2.destroyAllWindows()













