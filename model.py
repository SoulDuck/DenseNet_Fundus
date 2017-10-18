#-*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import input



TF_VERSION=TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))
print TF_VERSION
class DenseNet:
    def __init__(self, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        print kwargs

        self.logs_path='./logs/'

        """from input"""
        print 'model'
        n_classes=2
        data_shape=(299,299,3)
        self.n_classes = n_classes
        self.data_shape =data_shape


        self.growth_rate = growth_rate
        self.depth = depth
        self.total_blocks = total_blocks
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.first_output_features = growth_rate * 2
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.reduction = reduction
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        if not bc_mode:
            print "Build %s model with %d blocks %d composite layers each" % (
            model_type, self.total_blocks, self.layers_per_block)

        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print 'layer per block : ', self.layers_per_block
            print model_type
            print self.total_blocks
            print "Build %s model with %d blocks ,  %d bottleneck layers and %d composite layers each." \
                  % (model_type, self.total_blocks, self.layers_per_block, self.layers_per_block)
        print "Reduction at transition layers : %.1f" % self.reduction
        ##함수 실행부##
        self._define_inputs()
        self._build_graph()
        self.get_batches_from_tensor = input.get_batches_from_tensor
        self._images_test_list, self._labels_test_list, self._fnames_test_list = input.get_batch_tensor(mode='test')
        self._images_tensor_list , self._labels_tensor_list,self._fnames_tensor_list=input.get_batch_tensor(mode='train')


        """**images_tensor_list=[imgs_1_tensor , imgs_2_tensor , imgs_3_tensor]**"""
        self._initialize_session()
        """
        batch_xs, batch_ys, batch_fs = self.get_batches_from_tensor(sess=self.sess, images=self._images_tensor_list, \
                                                                    labels=self._labels_tensor_list,
                                                                     filenames=self._fnames_tensor_list)
        print np.shape(batch_xs)                                                             
        """

        self._count_trainable_params()
        print 'DenseNet model initialize Done'

    def _initialize_session(self):

        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            init=tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
            self.sess.run(init)
            logswriter = tf.summary.FileWriter
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)


        self.saver = tf.train.Saver()

        self.summary_writer = logswriter(logdir=self.logs_path)
        self.summary_writer.add_graph(tf.get_default_graph())
        print 'initialize...done'

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))
    @property
    def model_identifer(self):
        return "{}_growth_rate{}_depth{}_dataset_{}".format(self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def log_loss_accuracy(self , loss , accuracy  , epoch , prefix , should_print= True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss_%s' %prefix, simple_value=float(loss)),
                                    tf.Summary.Value(tag='accuracy_%s' % prefix, simple_value=float(accuracy))])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        print 'Debug | _define_inputs | '
        print 'shape :',shape
        self.x_ = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')

        self.y_ = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')

        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

        self.f_ = tf.placeholder(tf.string , shape=[None], name='filenames')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self , _input , out_features , kernel_size =3 ):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            print output
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output
    def bottlenect(self , _input , out_features) :
        with tf.variable_scope("bottle_neck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features*4
            output = self.conv2d(output , out_features= inter_features , kernel_size =1  , padding ='VALID')
            print 'debug bottlenect ',output
            output = self.dropout(output)
        return output
    def add_internal_layer(self , _input , growth_rate):
        #기능이 뭐지#
        if not self.bc_mode:
            comp_out= self.composite_function(_input ,out_features=growth_rate  , kernel_size=3)
        elif self.bc_mode:
            bottlenect_out = self.bottlenect(_input , out_features = growth_rate)
            comp_out = self.composite_function(bottlenect_out , out_features=growth_rate ,kernel_size=3)
        if TF_VERSION >= 1.0:
            output= tf.concat(axis=3 , values=(_input , comp_out))
        else:
            output  = tf.concat(3 ,(_input , comp_out))
            print output
        return output
    def add_block(self , _input , growth_rate  , layers_per_block):
        output = _input
        print _input
        for layer in range(layers_per_block):
            with  tf.variable_scope("layer_%d"%layer):
                output=self.add_internal_layer(output , growth_rate)
        return output

    def transition_layer(self , _input):
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(_input , out_features , kernel_size=1)
        output = self.avg_pool(output ,k=2)
        return output

    def transition_layer_to_clssses(self , _input):
        output = self.batch_norm(_input)
        output = tf.nn.relu(output)
        last_pool_kernel = int(output.get_shape()[-2])
        output=self.avg_pool(output , k=last_pool_kernel)


        features_total=int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W=self.weight_variable_xavier([features_total , self.n_classes] , name ='W')
        bias = self.bias_variable([self.n_classes])
        logits=tf.matmul(output, W)+bias

        return logits


    # 여기부터는 convolution layer을 상속해야 한다
    def conv2d(self , _input , out_features , kernel_size , strides=[1,1,1,1] , padding='SAME'):
        in_fearues=int(_input.get_shape()[-1])
        kernel=self.weight_variable_msra([kernel_size,kernel_size,in_fearues , out_features] , name='kernel')
        return tf.nn.conv2d(_input , kernel , strides , padding)

    def avg_pool(self , _input , k ):
        ksize=[1,k,k,1]
        strides=[1,k,k,1]
        padding='VALID'
        output=tf.nn.avg_pool(_input , ksize ,strides,padding)
        return output
    def batch_norm(self , _input):
        output = tf.contrib.layers.batch_norm(_input , scale=True , \
                                              is_training = self.is_training, updates_collections=None)
        return output
    def dropout(self , _input):
        if self.keep_prob <1:
            output = tf.cond(self.is_training , lambda : tf.nn.dropout(_input , self.keep_prob),lambda: _input)
        else:
            output = _input
        return output

    def weight_variable_msra(self , shape , name):
        return tf.get_variable(name=name , shape=shape , initializer=tf.contrib.layers.variance_scaling_initializer())
    def weight_variable_xavier(self , shape , name):
        return tf.get_variable(name=name , shape=shape , initializer=tf.contrib.layers.xavier_initializer())
    def bias_variable(self , shape  , name='bias' ):
        initial=tf.constant(0.0 , shape=shape)
        return tf.get_variable(name,initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block=self.layers_per_block #여기에 왜 있는지 모르겠는뎀 ;;;;

        with tf.variable_scope("Initial_convolution"):
            output=self.conv2d(self.x_, out_features=self.first_output_features , kernel_size=3 , strides=[1,2,2,1])
            print '##########',output
            print '##########',layers_per_block
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d"%block):
                output=self.add_block(output,growth_rate ,layers_per_block)
            if block != self.total_blocks -1 :
                with tf.variable_scope("Transition_after_block_%d"%block):
                    output= self.transition_layer(output)
        #logits 설정
        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_clssses(output)
        self.prediction= tf.nn.softmax(logits , name='softmax')

        #loss 설정

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits , labels=self.y_))
        self.cross_entropy=cross_entropy
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])


        optimizer= tf.train.MomentumOptimizer(self.learning_rate , self.nesterov_momentum , use_nesterov=True)
        self.train_step = optimizer.minimize(cross_entropy+l2_loss*self.weight_decay)
        self.correct_prediction  = tf.equal(
            tf.argmax(self.prediction ,1 ),
            tf.argmax(self.y_ , 1))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction , dtype = tf.float32))


    def training(self  , learning_rate):

        max_iter=100

        for step in range(max_iter):
            batch_xs ,batch_ys , batch_fs = self.get_batches_from_tensor(sess=self.sess  , images=self._images_tensor_list,\
                                                                         labels=self._labels_tensor_list , filenames=self._fnames_tensor_list )
            batch_ys=input.cls_to_onehot(batch_ys , self.n_classes )
            batch_xs , batch_ys=input.batch_shuffle(batch_xs , batch_ys)
            feed_dict = {
                #self._images_tensor_list , self._labels_tensor_list , self._fnames_tensor_list
                self.x_: batch_xs,
                self.y_: batch_ys,
                self.learning_rate: learning_rate,
                self.is_training: True}
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            _ , loss, accuracy=self.sess.run(fetches=fetches, feed_dict=feed_dict)
            self.log_loss_accuracy(loss=loss , accuracy=accuracy ,epoch = step, prefix='per_batch')
        self.coord.request_stop()
        self.coord.join(threads=self.threads)

    def testing(self):
        #cataract , normal 각각의 accuracy을 보여주고 마지막엔 모두를 더한 total accuracy을 보여준다
        acc_global=[]
        pred_list=[]
        imgs_labs_fnames_list=zip(self._images_test_list,self._labels_test_list,self._fnames_test_list)

        #여기에는 cataract , glaucoam , retina test  image ,label , fnames가 들어있다
        for i, (imgs_list , labs_list , fnames_list ) in enumerate(imgs_labs_fnames_list):
            imgs_labs_fnames_list=zip(imgs_list , labs_list , fnames_list)
            print '# : ',len(imgs_labs_fnames_list)
            for img , lab , fname in imgs_labs_fnames_list:
                print fname
                h,w,c=np.shape(img)
                img=img.reshape([1,h,w,c])
                feed_dict = {
                    #self._images_tensor_list , self._labels_tensor_list , self._fnames_tensor_list
                    self.x_: img,
                    self.is_training: False}
                fetches =  self.prediction
                pred = self.sess.run(fetches=fetches, feed_dict=feed_dict)
                pred_list.append(pred)
            pred_list=np.asarray(pred_list)
            pred_list=np.argmax(pred_list, axis=0)
            acc=np.mean(pred_list)
            print 'fname :{} accuracy : {}'.format(fname , acc )
            acc_global.extend([pred_list == lab])
        acc_global=np.mean(acc_global)
        print 'total accuracy : ',acc_global
    #self._images_test_list, self._labels_test_list, self._fnames_test_list






