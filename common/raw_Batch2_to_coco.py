# -*- coding:utf-8 -*-
# 把龙猫给我们的xml文件，结合我们自己的图片，转化为Pytorch框架下的 AIPrimeDataset类能够读取的COCO数据集
# AIPrimeDataset 类继承自 Torch.utils.data.Dataset， 是用来实现Pytorch读取数据的抽象类，我已经实现在ai_prime_dataset.py中
##############
# usage: python raw_Batch2_to_coco.py path_A path_B
# 功能：遍历path_A文件夹子目录内的图片，在path_B寻对应目录下的标签xml,生成图片列表（以及训练列表，测试列表 9：1），并把图片重新编号
# 生成：（运行python的当前路径下生成）
# images：存放重新编号过的图片
# labels：重新编号的xml文件，名字对应图像编号
# all_image.txt 记录了所有图片文件的绝对路径
# train.txt
# test.txt
# By Bryce Chen
##############
import os
import sys
import re
import xml.etree.ElementTree as ET
import pickle
from PIL import Image
import shutil
import time

if len(sys.argv) > 2:
    Pic_Root_dir = sys.argv[1]
    XML_Root_dir = sys.argv[2]

else:
    print('input pic_root dir & xml_root dir !')
    # exit(0)
    #Pic_Root_dir = "Batch1/20180524_images"
    #XML_Root_dir = "Batch1/20180524_labels"

New_Lab_dir = "labels"
New_Pic_dir = "images"

classes = [u"头部", u"人", u"衣服", u"圆桶", u"叉车和桶", u"单独叉车", u"ICB桶", u"小号桶"]
Pic_Root_dirs = ["20180529","20180531","20180601","20180605"]
XML_Root_dirs = ["batch22665","batch26525"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


cwd = os.getcwd()

if os.path.exists(New_Pic_dir):
    shutil.rmtree(New_Pic_dir)
if os.path.exists(New_Lab_dir):
    shutil.rmtree(New_Lab_dir)
os.mkdir(New_Lab_dir)
os.mkdir(New_Pic_dir)


def convert_annotation(jpg_path, new_pic_path, xml_path, new_lab_path, cnt):
    in_file = open(xml_path,'r')
    out_file = open(new_lab_path, 'w')


    temp_file = open("temp.xml", 'w')
    temp_xml = in_file.read().decode("gbk")
    #print temp_xml
    temp_xml= "<root>\n"+temp_xml+"</root>\n" # 原来的xml文件没有root节点，导致解析出错，这里加上root节点再分析
    temp_xml=temp_xml.encode('utf-8')
    temp_file.write(temp_xml)
    temp_file.close()

    temp_file = open('temp.xml','r')
    tree=ET.parse(temp_file)
    root = tree.getroot()
    im = Image.open(jpg_path)
    w, h  = im.size


    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            pass
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

        b = (max(0.0,float(xmlbox.find('xmin').text)), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        b = (min(b[0],b[1]),max(b[0],b[1]),min(b[2],b[3]),max(b[2],b[3]))
        #print((cls_id,b))
        #if min(b) <0 or max(b[0],b[1])>im.size[0] or max(b[2],b[3])>im.size[1]:
        #    print xml_path
        #    print jpg_path
        #    print(b)
        #if b[2]>=b[3] or b[0]>=b[1] or min(b[0],b[1]) <0:
        #    print(xml_path)
        #    print(b)

        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        shutil.copyfile(jpg_path, new_pic_path)

        #print(im.size, cls, cls_id, bb)

    all_image = open("all_image.txt",'a+')
    all_image.write(new_pic_path + "\n") # 向保存所有图片地址的txt内加入一条image
    all_image.close()

    if cnt%10==0:  #每隔10个数据取一个作为测试
        test = open("test.txt", 'a+')
        test.write(new_pic_path + "\n")
        test.close()
    else:
        train = open("train.txt", 'a+')
        train.write(new_pic_path + '\n')
        train.close()



all_image = open("all_image.txt",'w') # 清除
all_image.close()
test = open("test.txt",'w') # 清除
test.close()
train = open("train.txt",'w') # 清除
train.close()
cnt=0
for Pic_Root_dir in Pic_Root_dirs: # 进入每个日期文件夹
    for currentDir, dirs, files in os.walk(Pic_Root_dir):
        for filename in files:
            if ".jpg" in filename:
                jpg_path = cwd + '/' + os.path.join(currentDir, filename)
                for XML_Root_dir in XML_Root_dirs: # 尝试每个标签文件夹
                    xml_path = jpg_path.replace(Pic_Root_dir, XML_Root_dir, 1) # 先替换改变目录
                    #xml_path = xml_path.replace('.jpg', '.xml', 1) #再替换改变文件后缀名
                    xml_path = os.path.join(xml_path, filename.replace('jpg', 'xml'))  # 如果是每张图片单独文件夹的xml
                    if os.path.exists(xml_path): # 假如这张图片在这个标签文件夹下有标注的话
                        new_lab_path = cwd + '/' + New_Lab_dir + '/' + str(cnt).zfill(6) + '.txt'
                        new_pic_path = cwd + '/' + New_Pic_dir + '/' + str(cnt).zfill(6) + '.jpg'
                        # cwd + '/' + os.path.join(currentDir, filename)
                        print(xml_path)
                        print(jpg_path)
                        print(new_lab_path)
                        print(new_pic_path)
                        convert_annotation(jpg_path, new_pic_path, xml_path, new_lab_path, cnt)
                        cnt +=1

print(cnt)
os.remove('temp.xml')