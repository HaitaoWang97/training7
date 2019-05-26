#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_ zoo
from mxnet.gluon import utils as gutils
import os
import zipfile


# In[ ]:


#获取图片数据集
data_dir = '../data'
base_url = 'https://apach-mxnet.s3-accelerate.amazonaws.com/'
fname = gutils.download(base_url + 'glion/dataset/hotdog.zip',
        path=data_dir, shal_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
with zipfile.ZipFlie(fname, 'r') as z:
    z.extractall(data_dir)


# In[ ]:


train_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog/train'))
test_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog/test'))
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);


# In[ ]:


#图像通道归一化处理
normalize = gdata.vision.transfroms.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
#训练图像增广处理
train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    normalize])
#测试图片不做翻转
test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize])


# In[ ]:


#下载resnet作为初始模型
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)


# In[ ]:


#将新建的模型除输出层外的所有层都设为resnet的参数，再用较小的学习率微调
#新建的模型的输出层用Xavier函数初始化，要用较大的学习率训练
finetune_net = model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
finetune_net.output.collect_params().setatter('lr_mult', 10)


# In[ ]:


#定义微调模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gdata.DataLoader(
        train_imgs.trainform_first(train_augs), batch_size, shuffle = True)
    test_iter = gdata.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    ctx = d2l.try_all_gpus()
    net.collect_params().resnet_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate,
                                                         'wd': 0.001})
    d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


# In[ ]:


train_fine_tuning(finetune_net, 0.01)


# In[ ]:


scratch_net = model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)

