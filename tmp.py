
import tensorflow as tf
import numpy as np

def reconstruct_tfrecord_rawdata(tfrecord_path):

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
        print height ,width
        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = (example.features.feature['filename'].bytes_list.value[0])
        filename = filename.decode('utf-8')
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_fnames.append(filename)
    ret_imgs = np.asarray(ret_img_list)
    ret_labs = np.asarray(ret_lab_list)

    return ret_imgs, ret_labs ,ret_fnames

imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/normal_0_train.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/glaucoma_train.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/retina.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/cataract.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/cataract_glaucoma_train.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/retina_cataract_train.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/retina_glaucoma_train.tfrecord')
print np.shape(imgs)
imgs , labs , fnames=reconstruct_tfrecord_rawdata('./dataset/retina_train.tfrecord')
print np.shape(imgs)

a=np.asarray([[1,3],[2,4]])
print np.argmax(a,axis=1)