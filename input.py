#-*- coding:utf-8 -*-
import tensorflow  as tf
import numpy as np
import random
import sys
import os
import glob
from PIL import Image
import random

debug_lv0=True
debug_lv1=True

def get_fundus_train_test_set(src_name , src_paths , src_labels , random_shuffle=True):
    random.seed(123)
    assert len(src_paths) == len(src_labels) and len(src_paths) != 0
    if random_shuffle ==True:
        random.shuffle(src_paths)
        random.shuffle(src_labels)


    # 여기서 테스트 셋과 트레이닝 세트의 비율을 정합니다
    if src_name == 'glaucoma':
        n_test=100
    elif src_name == 'cataract':
        n_test = 50
    elif src_name == 'normal_0':
        n_test = 300
    elif src_name == 'normal_1':
        n_test = 300
    elif src_name == 'retina':
        n_test = 100
    elif src_name == 'cataract_glaucoma':
        n_test = 10
    elif src_name == 'retina_cataract':
        n_test = 10
    elif src_name == 'retina_glaucoma':
        n_test = 10
    src_train_images=src_paths[:n_test]
    src_train_labels = src_labels[:n_test]

    src_test_images = src_paths[:n_test]
    src_test_labels = src_labels[:n_test]

    return src_train_images , src_train_labels , src_test_images , src_test_labels


def reconstruct_tfrecord_rawdata(tfrecord_path ,resize=(299,299)):

    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image = []
    # caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    ret_img_list = []
    ret_lab_list = []
    ret_fnames = []
    for i, str_record in enumerate(record_iter):
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])

        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = example.features.feature['filename'].bytes_list.value[0]
        filename = filename.decode('utf-8')
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_fnames.append(filename)
    ret_imgs = np.asarray(ret_img_list)
    if  np.ndim(ret_imgs) ==3:
        ret_imgs=ret_imgs[:resize[0],:resize[1],:]
    elif np.ndim(ret_imgs) ==4:
        ret_imgs = ret_imgs[:,:resize[0], :resize[1], :]
    ret_labs = np.asarray(ret_lab_list)

    return ret_imgs, ret_labs ,ret_fnames




def make_fundus_tfrecords(root_folder , src_folder_names , src_labels , save_folder , extension='*.png'):
    """
    usage #
    folder - glaucoma
           - normal_0
           ...
           - cataract

    :param folders:
    :return:
    """
    assert len(src_folder_names) == len(src_labels)
    subdir_paths=map(lambda src_folder_name: os.path.join(root_folder , src_folder_name) , src_folder_names)
    print subdir_paths
    for i,subdir_path in enumerate(subdir_paths):

        target_src_paths=glob.glob(os.path.join(subdir_path , extension))
        target_src_labels=np.zeros(len(target_src_paths) , dtype=np.int32)
        target_src_labels.fill(src_labels[i])
        #print 'name : ',src_folder_names[i] , 'label : ',src_labels[i]
        target_saved_folder=os.path.join(save_folder , src_folder_names[i])
        train_img_paths , train_labs , test_img_paths , test_labs=get_fundus_train_test_set(src_folder_names[i] , src_paths= target_src_paths, \
                                                                                  src_labels=target_src_labels  , random_shuffle=True)

        make_tfrecord_rawdata(target_saved_folder + '_train.tfrecord', train_img_paths, train_labs)
        make_tfrecord_rawdata(target_saved_folder + '_test.tfrecord',  test_img_paths, test_labs)

def make_tfrecord_rawdata(tfrecord_path , paths , labels):
    """
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param paths: e.g)[./pic1.png , ./pic2.png]
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    debug_flag_lv0=False
    debug_flag_lv1=False
    if __debug__ == debug_flag_lv0:
        print 'debug start | batch.py | class : tfrecord_batch | make_tfrecord_rawdata'

    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    paths_labels=zip(paths ,labels)
    error_file_paths=[]
    for ind, (path , label) in enumerate(paths_labels):
        try:
            msg = '\r-Progress : {0}'.format(str(ind) +'/'+str(len(paths_labels)))
            sys.stdout.write(msg)
            sys.stdout.flush()

            np_img=np.asarray(Image.open(path)).astype(np.int8)
            height = np_img.shape[0]
            width = np_img.shape[1]
            raw_img = np_img.tostring()
            dirpath , filename=os.path.split(path)
            filename , extension=os.path.splitext(filename)
            if __debug__ == debug_flag_lv1:
                print ''
                print 'image min', np.min(np_img)
                print 'image max', np.max(np_img)
                print 'image shape' , np.shape(np_img)
                print 'heigth , width',height , width
                print 'filename' , filename
                print 'extension ,',extension


            example = tf.train.Example(features = tf.train.Features(feature = {
                        'height': _int64_feature(height),
                        'width' : _int64_feature(width),
                        'raw_image' : _bytes_feature(raw_img),
                        'label' : _int64_feature(label),
                        'filename':_bytes_feature(tf.compat.as_bytes(filename))
                        }))
            writer.write(example.SerializeToString())
        except IndexError as ie :
            print path
            continue
        except IOError as ioe:
            print path
            continue
        except Exception as e:
            print path
            print str(e)
            continue
    writer.close()


def batch_shuffle(images , labels , filenames=None):
    indices=np.random.permutation(len(labels))

    images=images[indices]
    labels=labels[indices]

    if filenames is not  None:
        filenames=filenames[indices]
        return images , labels , filenames

    return images , labels


def cls_to_onehot(cls , depth):
    onehot=np.zeros([len(cls) , depth])

    for i in range(len(cls)):
        onehot[i , cls[i]]=1
    return onehot

def read_one_example( tfrecord_path , batch_size , resize ):
    filename_queue = tf.train.string_input_producer(tfrecord_path , num_epochs=10)
    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    image_shape = tf.pack([height , width , 3 ])
    image=tf.reshape(image ,  image_shape)
    if not resize == None :
        resize_height , resize_width  = resize
        image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=resize_height,
                                               target_width=resize_width)
#    images  = tf.train.shuffle_batch([image ] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
    return image,label


def get_batch( tfrecord_path , batch_size , resize  , mode , num_epochs):
    resize_height , resize_width  = resize
    filename_queue = tf.train.string_input_producer(tfrecord_path , num_epochs=1000)
    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64),
        'filename': tf.FixedLenFeature([] , tf.string)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    filename = tf.cast(features['filename'] , tf.string)

    image_shape = tf.stack([height , width , 3 ])  #image_shape shape is ..
    image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
    image=tf.reshape(image ,  image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=resize_height,
                                           target_width=resize_width)
    if mode == 'train':
        images  , labels  , filenames= tf.train.shuffle_batch([image ,label ,filename] , batch_size =batch_size  , capacity =30000 ,num_threads=16 , min_after_dequeue=10)
        return images , labels , filenames
    if mode == 'test':
        pass

def get_example_queue(mode, batch_size , image_size , depth):
    if mode == 'train':
        example_queue=tf.RandomShuffleQueue(
                capacity=16 * batch_size,
                min_after_dequeue=8*batch_size,
                dtypes = [tf.float32 , tf.int32 , tf.string],
                shapes = [[image_size , image_size , depth] , [] , [] ])
        num_threads=16
    elif mode =='test':
        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32, tf.string ],
            shapes=[[image_size, image_size, depth], [], [] ])
        num_threads = 1
    return example_queue

def enqueue(example_queue,image_tensor , label_tensor , filename_tensor):
    example_enqueue_op = example_queue.enqueue([image_tensor, label_tensor, filename_tensor])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op]))



def dequeue(example_queue,batch_size):
    images, cls = example_queue.dequeue_many(batch_size)
    cls= tf.reshape(cls, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    images, cls = example_queue.dequeue_many(batch_size)
    return images , cls
"""
def cls_to_onehot(indices,cls, n_classes):
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, cls], axis=1),
        [batch_size, n_classes], 1.0, 0.0)
    tf.summary.histogram(''
                         ''
                         'labels', labels)
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, n_classes], 1.0, 0.0)
"""


def get_batch_tensor(mode):
    """

    아래 줄이 반드시 정의 되어 있어야 합니다
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    마지막에는
    coord.request_stop()
    coord.join(threads)
    이 정의되어 있어야 합니다
    """

    fetches = ['normal_0', 'glaucoma', 'retina', 'cataract', 'cataract_glaucoma', 'retina_cataract',
                   'retina_glaucoma']
    epochs=[10,]
    if mode=='train' or mode == 'Train':
        fetches=map(lambda fetch : fetch +'_train' ,fetches)
    elif mode == 'test' or mode == 'Test':
        fetches=map(lambda fetch: fetch +'_test', fetches)
    batches=[30,14,14,6,4,3,3]
    assert len(fetches) == len(batches)

    fbe=zip(fetches , batches , epochs)
    images_list=[]
    labels_list=[]
    filenames_list=[]

    for f, b , e  in fbe:
        print 'name:', f, '\tbatch:', b
        tfrecord_path = tf.gfile.Glob('./dataset' + '/%s.tfrecord' % f)
        #Glob을쓰는이유는 이렇게 해야 tensor가 인식을 한다
        #Glob의 원래 목적은 정규식 패턴에 해당하는 파일을 모두 찾아 반환하는 거지만  여기서는 그냥 하나의 파일을 찾기 위해 사용한다

        if __debug__ == debug_lv0:
            print 'tfrecord path : ',tfrecord_path

        if mode == 'train' or mode == 'Train':
            images, labels, filenames = get_batch(tfrecord_path, batch_size=b, resize=(299, 299), mode=mode , num_epochs=e )
            #images_list, labels_list, filenames_list  is list that was included tensor
        elif mode == 'test' or mode == 'Test':
            tfrecord_path=tfrecord_path[0]
            print '####tfrecord_path',tfrecord_path[0]
            images, labels , filenames=reconstruct_tfrecord_rawdata(tfrecord_path,resize=(299, 299))
            if __debug__ == debug_lv0:
                print 'image shape : ',np.shape(images)
                print 'label shape : ', np.shape(labels)
                print 'fname shape : ', len(filenames)
        images_list.append(images)
        labels_list.append(labels)
        filenames_list.append(filenames)
    print 'Done'
    return images_list, labels_list, filenames_list
"""
for i in xrange(2):
    imgs_labs_fnames=zip(images , labels , filenames)



    for idx , (imgs , labs , fnames) in enumerate(imgs_labs_fnames):
        
        if idx ==0:
            batch_xs=imgs
            batch_ys=labs
            batch_fs=fnames
        else:
            batch_xs.vstack(imgs)
            batch_ys.hstack(labs)
            batch_fs.hstrack(fnames)
            assert len(batch_xs)==len(batch_ys)==len(batch_fs)
        print np.shape(batch_xs)

    print np.shape(imgs)
    print labs
    print fnames_
"""



def get_batches_from_tensor(sess ,images , labels , filenames ):
    imgs, labs, fnames = sess.run([images, labels, filenames])
    imgs_labs_fnames=zip(imgs,labs ,fnames)
    for i,(img,lab,fname) in enumerate(imgs_labs_fnames):
        if i ==0 :
            tmp_imgs = img
            tmp_labs = lab
            tmp_fnames = fname
        else:
            tmp_imgs=np.vstack((tmp_imgs ,img))
            tmp_labs=np.hstack((tmp_labs ,lab))
            tmp_fnames = np.hstack((tmp_fnames, fname))
    imgs=tmp_imgs
    labs=tmp_labs
    fnames=tmp_fnames
    imgs = np.asarray(imgs).reshape([-1, 299, 299, 3])
    labs = np.asarray(labs).reshape([-1])
    fnames = np.asarray(fnames).reshape([-1])

    assert len(imgs)==len(labs) == len(fnames) , '# images : {} , # labels {} , # filenames : {}'.format(len(imgs) , len(labs) , len(fnames
                                                                                                                                     ))
    return imgs, labs, fnames


if __name__ =='__main__':
    images , labels , filenames=get_batch_tensor(mode='train')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord , sess=sess)
    for i in range(2):
        get_batches_from_tensor(sess ,images , labels , filenames)

    coord.request_stop()
    coord.join(threads=threads)
