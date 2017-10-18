#-*- coding:utf-8 -*-
import input as input
import argparse
import model

parser  = argparse.ArgumentParser()
parser.add_argument('--train' , dest='mode',action='store_true' , help='Train the model')
parser.add_argument('--test' , dest='mode',action='store_true')
parser.add_argument('--growth_rate' , '-k' , type=int , choices=[12,24,40] , default=12 , help='growth rate for every layer')
parser.add_argument('--depth' , '-d' , type=int , choices=[40 , 100 , 190 , 250] , default=40 , help ='Depth of every network')
parser.add_argument('--total_blocks' , '-tb' , type=int , default=3 , metavar='')
parser.add_argument('--keep_prob' , '-kp' , type=float , metavar='')
parser.add_argument('--weight_decay' , '-wd' , type=float , default=1e-4 , metavar='')
parser.add_argument('--nesterov_momentum' , '-nm' , type=float , default=0.9 , metavar='' )
parser.add_argument('--model_type' , '-m' , type=str , choices=['DenseNet' , 'DenseNet-BC'] , default='DenseNet')
parser.add_argument('--dataset', '-ds' , type=str , choices=['C10', 'C10+', 'C100' , 'C100+' , 'SVHN' , 'Fundus' ] , default='C10')

parser.add_argument('--reduction' , '-red' , type=float , default=0.5  , metavar='')

parser.add_argument('--logs', dest='should_save_logs' ,action='store_true')
parser.add_argument('--no_logs', dest='should_save_logs', action='store_false')
parser.set_defaults(should_save_logs=True)

parser.add_argument('--saves' , dest='should_save_model' , action = 'store_true')
parser.add_argument('--no-saves' , dest='should_save_model', action ='store_false')
parser.set_defaults(should_save_model=True)

parser.add_argument('--renew_logs', dest='renew_logs' , action='store_true')
parser.add_argument('--not_renew_logs' , dest='renew_logs' , action='store_false')
parser.set_defaults(renew_logs=False)
args=parser.parse_args()


### 임시적으로 사용함 ###
# args.test=True
args.dataset = 'Fundus'
args.total_blocks=7


if not args.keep_prob:
    if args.dataset in ['C10' , 'C100'  , 'SVHN' , 'Fundus']:
        args.keep_prob=0.8
    else:
        args.keep_prob=1.0
if args.model_type == 'Densenet':
    args.bc_mode= False
    args.reduction = 1.0
elif args.model_type=='DenseNet-BC':
    args.bc_mode=True
src_folder_names=['normal_0' , 'normal_1','glaucoma', 'cataract', 'retina','retina_glaucoma','retina_cataract','cataract_glaucoma']
src_labels=[1,1,0,0,0,0,0,0]
input.make_fundus_tfrecords(root_folder='../fundus_data/cropped_original_fundus_300x300' , src_folder_names=src_folder_names , src_labels=src_labels , save_folder='./dataset')


model_params = vars(args)
densenet=model.DenseNet(**model_params)
densenet.testing()
densenet.training(learning_rate=0.1) #100에 한번씩 test을 한다


