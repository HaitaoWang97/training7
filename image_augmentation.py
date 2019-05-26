#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
import time


# In[3]:


#读取一张图片
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy())


# In[10]:


#定义绘制多个子图函数,展示不同的图片处理效果
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


# In[23]:


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


# In[24]:


apply(img, gdata.vision.transforms.RandomFlipLeftRight())


# In[13]:


apply(img, gdata.vision.transforms.RandomFlipTopBottom())


# In[14]:


#随机裁剪输出固定图片大小
shape_aug = gdata.vision.transforms.RandomResizedCrop(
(200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)


# In[15]:


#变化颜色
apply(img, gdata.vision.transforms.RandomBrightness(0.5))


# In[16]:


#改变色调
apply(img, gdata.vision.transforms.RandomHue(0.5))


# In[17]:


#RandomColorJitter同时处理图片亮度，色调，饱和度
color_aug = gdata.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5,saturation=0.5, hue=0.5)
apply(img, color_aug)


# In[18]:


#将多个图像叠加起来
augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(), color_aug,shape_aug])
apply(img, augs)


# In[25]:


show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=0.8)


# In[26]:


#用ToTensor将图片格式转化成mxnet需要的格式(批量大小， 通道数， 高， 宽)
flip_aug = gdata.vision.transforms.Compose([
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor()])

no_aug = gdata.vision.transforms.Compose([
        gdata.vision.transforms.ToTensor()])


# In[29]:


num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size):
    return gdata.DataLoader(
        gdata.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


# In[30]:


#多GPU计算
def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


# In[31]:


#将小批量数据复制到ctx指定的各个显存
def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype !=features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
           gutils.split_and_load(labels, ctx),festures.shape[0])


# In[32]:


#定义evaluate_accuracy函数评价模型的分类准确率
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


# In[35]:


#定义train函数
def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on',ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(trainer_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum(),asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f,test acc %.3f, time %.1f sec'
             % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))
        
            


# In[ ]:


#用adam优化
def train_with_aug(train_augs, test_augs, lr=0.001):
    batch_size, ctx, net= 256, try_all_gpus(), d2l.resnet18(10)
    net = initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(),'adam,'{'learnind_rate': lr})
    loss =gloss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)


# In[ ]:


trian_with_data_aug(flip_aug, no_aug)

