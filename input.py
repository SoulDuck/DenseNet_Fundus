#-*- coding:utf-8 -*-
import tensorflow  as tf
import numpy as np
import random
import sys
import os
import glob
from PIL import Image
import random
def get_fundus_train_test_set(src_name , src_paths , src_labels , random_shuffle=True):
    assert len(src_paths) == len(src_labels)
    if random_shuffle ==True:
        indices=np.random.permutation(len(src_labels))
        src_paths=src_paths[indices]
        src_labels=src_labels[indices]

    # 여기서 테스트 셋과 트레이닝 세트의 비율을 정합니다
    if src_name == 'glaucoma':
        n_train=100
    elif src_name == 'cataract':
        n_train = 50
    elif src_name == 'normal_0':
        n_train = 300
    elif src_name == 'normal_1':
        n_train = 300
    elif src_name == 'retina':
        n_train = 100
    elif src_name == 'cataract_glaucoma':
        n_train = 10
    elif src_name == 'retina_cataract':
        n_train = 10
    elif src_name == 'retina_glaucoma':
        n_train = 10
    src_train_images=src_paths[:n_train]
    src_train_labels = src_labels[:n_train]

    src_test_images = src_paths[n_train:]
    src_test_labels = src_labels[n_train:]

    return src_train_images , src_train_labels , src_test_images , src_test_labels




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
        target_src_labels=np.zeros(len(target_src_paths))
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
    debug_flag_lv0=True
    debug_flag_lv1=True
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


def get_shuffled_batch( tfrecord_path , batch_size , resize ):
    resize_height , resize_width  = resize
    filename_queue = tf.train.string_input_producer(tfrecord_path , num_epochs=100)
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
    images  , labels  , filename= tf.train.shuffle_batch([image ,label ,filename] , batch_size =batch_size  , capacity =30000 ,num_threads=1 , min_after_dequeue=10)
    return images  ,labels , filename

mode='train'

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


def get_batch_tensor():


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
    batches=[30,14,14,6,4,3,3]
    assert len(fetches) == len(batches)

    fb=zip(fetches , batches)
    images=[]
    labels=[]
    filenames=[]
    for f,b in fb:
        print 'name:',f ,'\tbatch:',b
        tfrecord_path = tf.gfile.Glob('dataset'+'/*%s.tfrecord'%f)
        print tfrecord_path
        imgs , labs , fnames = get_shuffled_batch(tfrecord_path , batch_size=b , resize=(299,299))
        images.append(imgs)
        labels.append(labs)
        filenames.append(fnames)
    print 'Done'
    return images, labels,filenames
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
    imgs = np.asarray(imgs).reshape([-1, 299, 299, 3])
    labs = np.asarray(labs).reshape([-1])

    return imgs, labs, fnames


if __name__ =='__main__':
    images , labels , filenames=get_batch_tensor()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord , sess=sess)
    for i in range(2):
        get_batches_from_tensor(sess ,images , labels , filenames)

    coord.request_stop()
    coord.join(threads=threads)
