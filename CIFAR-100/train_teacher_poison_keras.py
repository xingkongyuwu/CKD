import torch
import os
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset,TensorDataset
import torch.multiprocessing
from mydataloader import Get_Poison_Dataloader
from tensorflow.python.keras.datasets import cifar100
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from art.defences.transformer.poisoning import NeuralCleanse
import matplotlib.pyplot as plt
import keras
from art.estimators.classification import KerasClassifier
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.models import Sequential
from tensorflow.keras import layers,activations
torch.multiprocessing.set_sharing_strategy('file_system')
# 解决读取数据过多
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
global graph, sess

graph = tf.get_default_graph()
sess = keras.backend.get_session()


def resnet50_model():
	#include_top为是否包括原始Resnet50模型的全连接层，如果不需要自己定义可以设为True
	#不需要预训练模型可以将weights设为None
    resnet50=tf.keras.applications.ResNet50(include_top=False,
                                            weights='imagenet',
                                            input_shape=(32,32,3),
                                            )
	#设置预训练模型冻结的层，可根据自己的需要自行设置
    for layer in resnet50.layers[:15]:
        layer.trainable = False  #

	#选择模型连接到全连接层的位置
    last=resnet50.get_layer(index=30).output
    #建立新的全连接层
    x=tf.keras.layers.Flatten(name='flatten')(last)
    x=tf.keras.layers.Dense(1024,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.5)(x)
    x=tf.keras.layers.Dense(128,activation='relu',name='dense1')(x)
    x=tf.keras.layers.Dropout(0.5,name='dense_dropout')(x)
    x=tf.keras.layers.Dense(10,activation='softmax')(x)

    model = tf.keras.models.Model(inputs=resnet50.input, outputs=x)
    model.summary() #打印模型结构
    return model

def conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same'):
    '''卷积、归一化和relu三合一'''
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inpt)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_bottle(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False):
    '''18中的4个basic_bottle'''
    x = conv2d_bn(inpt, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    x = conv2d_bn(x, filters=filters)
    if if_baisc==True:
        temp = conv2d_bn(inpt, filters=filters, kernel_size=(1,1), strides=2, padding='same')
        outt = layers.add([x, temp])
    else:
        outt = layers.add([x, inpt])
    return outt

def resnet18(class_nums):
    '''主模型'''
    inpt = layers.Input(shape=(32,32,3))
    #layer 1
    x = conv2d_bn(inpt, filters=64, kernel_size=(7,7), strides=2, padding='valid')
    x = layers.MaxPool2D(pool_size=(3,3), strides=2)(x)
    #layer 2
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    #layer 3
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    #GlobalAveragePool
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_nums, activation='softmax')(x)
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model



def get_poison_dataloader(dataloader_x, dataloader_y,poison_label, poison_ratio, add_clean_flag):
    # posion_type: 0 ours , 1 badnets 2 sgn
    total = 50000
    poison_img_num = int(total * poison_ratio / 100)

    num_count = [0 for i in range(100)]
    data_train_y = []
    data_train_x = []
    x = []
    y = []
    # images shape 是四维的
    for i, (images, labels) in enumerate(zip(dataloader_x,dataloader_y)):
        # print(images.shape)
        # print(mask.shape)
        # print(trigger.shape)
        # time.sleep(1321)
        images = images.transpose(0,2)
        images = images.transpose(1, 2)
        # print((np.array(x)).shape)
        # print(labels.item())
        images, labels = images.cuda(0), labels.cuda(0)
        if num_count[labels.item()] < poison_img_num:

            img = (1 - torch.unsqueeze(mask,dim=0)) * images + torch.unsqueeze(mask,dim=0) * trigger
            img = img.tolist()
            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
            # 保存正常数据  数据集中，x , x'都有。而不是只有x'
            x.append(images.tolist())
            y.append(labels.item())
        else:
            img = images.tolist()
            label = labels.item()

        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    # print((np.array(x)).shape)
    # print((np.array(data_train_x)).shape)
    # time.sleep(234)
    if add_clean_flag:
        size = len(x)
        num = int(size * 1)
        data_train_x.extend(x[:num])
        data_train_y.extend(y[:num])

    # print(type(data_train_x))

    return np.array(data_train_x), np.array(data_train_y)


trans_totensor = transforms.Compose([transforms.ToTensor()])
# path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/amax64FalseTrue-1_mask_loss3.936.png"
path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/amin64FalseTrue-1_mask_loss12.202.png"
input_patch = Image.open(path).convert('RGB')
trigger = trans_totensor(input_patch).to(0)
# mask = torch.load("/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/amax64FalseTrue-1_trigger_loss3.936.pth").to(args.gpu)
mask = torch.load("/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/amin64FalseTrue-1_trigger_loss12.202.pth").to(0)











(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)
x_train_poison, y_train_poison = get_poison_dataloader(x,y,0,0.01,True)
x_train_poison = np.transpose(x_train_poison, (0, 2, 3,1 ))

x = torch.from_numpy(x_test)
y = torch.from_numpy(y_test)
x_test_poison, y_test_poison = get_poison_dataloader(x,y,0,1,False)
x_test_poison = np.transpose(x_test_poison, (0, 2, 3,1 ))
y_train = to_categorical(y_train, 100)  #对数据集进行onehot编码
y_test = to_categorical(y_test, 100)
y_train_poison = to_categorical(y_train_poison, 100)  #对数据集进行onehot编码
y_test_poison = to_categorical(y_test_poison, 100)

# model.fit(x_train_poison, y_train_poison, epochs=2, batch_size=64)

# model = ResNet18(
#     weights=None,
#     classes=100,
#     input_shape=(32,32,3),
# )
# model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model = resnet18(100)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


classifier = KerasClassifier(model=model)
classifier.fit(x_train_poison, y_train_poison, nb_epochs=100, batch_size=64)


print("ACC")

clean_preds = np.argmax(classifier.predict(x_test), axis=1)
clean_correct = np.sum(clean_preds == np.argmax(y_test, axis=1))
clean_total = y_test.shape[0]
clean_acc = clean_correct / clean_total
print(clean_acc)
# _, clean_acc = model.evaluate(x_test, y_test, batch_size=32)


print("ASR")

clean_preds = np.argmax(classifier.predict(x_test_poison), axis=1)
clean_correct = np.sum(clean_preds == np.argmax(y_test_poison, axis=1))
clean_total = y_test.shape[0]
poison_acc1 = clean_correct / clean_total
print(poison_acc1)
_, poison_acc = model.evaluate(x_test_poison, y_test_poison, batch_size=32)



# 检测
# init = tf.global_variables_initializer()
# session.run(init)

cleanse = NeuralCleanse(classifier)
defence_cleanse = cleanse(classifier, steps=10, learning_rate=0.1)

# ttt = [0 for i in range(100)]
# ttt[0] = 1
# pattern, mask1 = defence_cleanse.generate_backdoor(x_test, y_test, np.array(ttt))
# plt.imshow(np.squeeze(mask1 * pattern))



# defence_cleanse = cleanse(classifier, steps=10, learning_rate=0.1)
defence_cleanse.mitigate(x_test, y_test, mitigation_types=["unlearning"])

poison_preds = np.argmax(classifier.predict(x_test_poison), axis=1)
poison_correct = np.sum(poison_preds == np.argmax(y_test_poison, axis=1))
poison_total = y_test_poison.shape[0]
new_poison_acc = poison_correct / poison_total
print("\n Effectiveness of poison after unlearning: %.2f%% (previously %.2f%%)" % (new_poison_acc * 100, poison_acc * 100))
clean_preds = np.argmax(classifier.predict(x_test), axis=1)
clean_correct = np.sum(clean_preds == np.argmax(y_test, axis=1))
clean_total = y_test.shape[0]

new_clean_acc = clean_correct / clean_total
print("\n Clean test set accuracy: %.2f%% (previously %.2f%%)" % (new_clean_acc * 100, clean_acc * 100))





















