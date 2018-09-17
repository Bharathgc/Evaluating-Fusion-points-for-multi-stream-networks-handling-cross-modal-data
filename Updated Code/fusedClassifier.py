
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import batch_norm
import time
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib.layers import flatten, fully_connected
import os 
import argparse
import cnn_class as fc


def load_data(source_path,std_size): 
	'''
	This method does the loading of data and shit. Source path contains images and data_file. 
	data_file contains image names and classes. Don't judge me. I know the code could be better. 
	data is resized to fit the model requirements. Labels are converted to numeric values.
	Also, it's possible Bharath could be gay. Just putting it out there.
	'''
	global class_dict
	print("Data is being Loaded. Kindly wait . . . .  :) :( ;) B) x) .. ran out of emojis")
	file_h=open(source_path+data_file)
	rgb_images,depth_images,labels=[],[],[]
	for i in file_h:
		if not i == "":
			data=i.split(',')
			rgb_images.append(cv2.resize(cv2.imread(source_path+data[0]),std_size))
			depth_images.append(cv2.resize(cv2.imread(source_path+data[1]),std_size))
			labels.append(class_dict[data[2][:-1]])
	return(rgb_images,depth_images,labels)

def normalizeData(rgb_images,viz=False):   
	'''
	This doesn't do any Normalization actually. I thought it would be moderately funny to add it here. 
	Go ahead and change the method to do a mean center normalization. Won't make a big difference. Like I care
	'''
	rgb_images=(np.float32(rgb_images))/255.0
	for k in range(3):
		if(viz):
			plt.figure(2)
			plt.title(str(class_labels[labels[rand_index]]))
			plt.imshow(rgb_images[rand_index],cmap='gray')
			plt.show()
	return (rgb_images)
		
def training(train_output,train_input):  
	'''
	method for training. Forward pass and backprop included.  No biggie :)
	Accuracy calculation tensor also included.
	I wish human training was as simple. I would train pavithra to use her brain occasionally.
	'''

	dataLength = len(train_input)
	update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer=tf.train.AdamOptimizer(learning_rate=lr)
		Trainables=optimizer.minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess=tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	t_accuracy=0
	loss_list,accuracy_list,time_list=list(),list(),list()
	print ("Training the model......\n")
	for i in range(EPOCHS): 
		train_input,train_output=shuffle(train_input,train_output)
		count=0
		for offset in range(0,dataLength,BATCH_SIZE):
			count+=1
			t5=time.time()
			print("Performing iteration {} of {}".format(count,int(np.ceil(dataLength/BATCH_SIZE))))
			end=offset + BATCH_SIZE
			batch_x1,batch_x2,batch_y=train_input[offset:end,0],train_input[offset:end,1],train_output[offset:end]
			if (len(batch_x1) == 0 or len(batch_x2)==0 or len (batch_y) == 0):
				continue
			batch_x1,batch_x2=np.asarray(batch_x1),np.asarray(batch_x2)
			
			batch_x1,batch_x2=batch_x1.reshape(-1,std_size[1],std_size[0],3),batch_x2.reshape(-1,std_size[1],std_size[0],3)
			probe_time=time.time()
			_,l,acc=sess.run([Trainables,loss,accuracy_operation],feed_dict={x_1:batch_x1,x_2:batch_x2,y:batch_y,drop_prob:0.7})
			time_info=time.time()-probe_time
			loss_list.append(l)
			print ("Epoch: {} Accuracy : {:.2f}% Loss : {} Time taken for current batch: {} s".format(i+1,acc*100,l,time.time()-t5))
			accuracy_list.append(acc)
			time_list.append(time_info)

			t_accuracy+=acc*len(batch_x1)
		t_accuracy=t_accuracy/dataLength
	return (sess,loss_list,accuracy_list,time_list)		

def validate(test_output,test_input,sess):  
	'''
	# Validation method. I use this for both validating and testing methods. 
	Data and graph passed are different for the above mentioned phases.. I know, naming is confusing.. But hey, Fuck you! :p
	'''
	print("\nValidating\n")
	dataLength=len(test_input)
	prediction = tf.argmax(logits,1)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	count=0
	total_accuracy=0	
	loss_list,accuracy_list,time_list=list(),list(),list()
	
	for offset in range(0,dataLength,BATCH_SIZE):
			count+=1
			t5=time.time()
			print("\nPerforming iteration {} of {}".format(count,int(np.ceil(dataLength/BATCH_SIZE))))
			end=offset + BATCH_SIZE
			batch_x1,batch_x2,batch_y=test_input[offset:end,0],test_input[offset:end,1],test_output[offset:end]
			batch_x1,batch_x2=np.asarray(batch_x1),np.asarray(batch_x2)
			batch_x1,batch_x2=batch_x1.reshape(-1,std_size[1],std_size[0],3),batch_x2.reshape(-1,std_size[1],std_size[0],3)
			tp=time.time()
			accuracy,l=sess.run([accuracy_operation,loss],feed_dict={x_1:batch_x1,x_2:batch_x2,y:batch_y,drop_prob:1})
			time_info=time.time()-tp
			loss_list.append(l)
			accuracy_list.append(accuracy)
			time_list.append(time_info)
			total_accuracy+=accuracy*len(batch_x1)
	total_accuracy=total_accuracy/dataLength	
	print ("Accuracy : {:.2f} % Loss : {} Time taken for current batch: {} s".format(total_accuracy*100,l,time.time()-t5))
	return (loss_list,accuracy_list,time_list)

if __name__ =='__main__':
	'''
	Setting up the save environment for the log files
	'''
	parser=argparse.ArgumentParser(description =  "Training two stream Classifiers")
	parser.add_argument('model_name',help='Enter the model name. Logs files will be found under logs/<this name >')
	model_name=parser.parse_args().model_name
	source_path="NYUD_final/"
	data_file='final_data_file.txt'
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	log_path="log/"+model_name +'/'
	truncate_data=False
	print ("Your logging will be in " + log_path)
	model_savepath=log_path +model_name+'.ckpt'
	
	train_loss_file_path=log_path  + 'train_data.txt'
	valid_loss_file_path=log_path  + 'valid_data.txt'
	test_loss_file_path =log_path  + 'test_data.txt'
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	'''
	Setting up the hyper parameters
	'''
	class_dict={'bedroom':0,'kitchen':1,'living_room':2,'bathroom':3,'dining_room':4,'unknown':5} # OUR classes for classification
	train_test_split=0.80 #80 percentage training test data split.
	data_len=0
	BATCH_SIZE=6
	EPOCHS=3
	lr=0.0001 
	NUM_CLASSES = 6  # Determined from the nyud script.
	std_size=(224,224) # specifications from the Alexnet paper. Also, fits in most GPUs
	k_fold=2
	vis=False # if you want to visualize the loaded data .
	

	#Data Loading  and visualization
	rand_index=np.random.randint(1)
	rgb_images,depth_images,labels=load_data(source_path,std_size)
	rgb_images,depth_images,labels=shuffle(rgb_images,depth_images,labels)
	rgb_images=normalizeData(rgb_images)
	depth_images=normalizeData(depth_images)
	if vis:
		# Rupa + push_up = Pushpa. I need a Life. *sigh*
		cv2.imshow('rgb',rgb_images[rand_index])
		cv2.imshow('depth',depth_images[rand_index])
		print (labels[rand_index])
		cv2.waitKey(0)
	print("Input images1 length : {}\t Input images2 length :{}\t  Labels length : {}\n".format(len(rgb_images),len(depth_images),len(labels)))


	#splitting into train and test data
	inputs=[[rgb_images[i],depth_images[i]] for i in range(len(rgb_images))]
	inputs=np.asarray(inputs)
	data_len=int(np.floor(len(rgb_images)*train_test_split))
	train_input,train_output=inputs[:data_len],labels[:data_len]
	test_input,test_output=inputs[data_len:],labels[data_len:]
	train_input,train_output = shuffle(train_input,train_output)
	test_input,test_output = shuffle(test_input,test_output)
	print("Train input shape : {} \nTraing output shape : {} \nTest input shape : {}\nTest output shape : {} ".format(train_input.shape,len(train_output),test_input.shape,len(test_output)))
	if vis:
		# Showing the images of the train input. If you need. Shimmy shimmy A shimiy A shimmiy ya... Swalla lallaaa
		cv2.imshow('train_0',train_input[rand_index,0])
		cv2.imshow('train_1',train_input[rand_index,1])
		print (train_output[rand_index])
		cv2.waitKey(0)

	# preparing the tensors
	x_1=tf.placeholder(tf.float32,(None,std_size[1],std_size[0],3),name='x1') #data input 1
	x_2=tf.placeholder(tf.float32,(None,std_size[1],std_size[0],3),name='x2') #data input 2
	y=tf.placeholder(tf.int32,(None),name='y')								  #data output
	loss_values=tf.placeholder(tf.float32,(None),name='loss_list')			  # cross-entropy loss
	accuracy_values=tf.placeholder(tf.float32,(None),name='accuracy_list')	  # accuracy loss
	one_hot_y=tf.one_hot(y,NUM_CLASSES)										  # one hot representation
	drop_prob=tf.placeholder(tf.float32,(None),name='dropout_probability')

	# Our model graph and loss function
	model=fc.CNN(NUM_CLASSES,drop_prob)
	logits=model.vggnet_fused3(x_1,x_2)
	logits=tf.identity(logits,name='forward_pass')
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_y),name='loss')
	t3=time.time()
	saver=tf.train.Saver()
	
	#Training with K-fold cross validation
	train_input_copy=np.copy(train_input)
	train_output_copy=train_output[:]
	start=0 
	offset=int(len(train_output)/k_fold)
	end=int(offset)
	train_ls_,train_acc_,train_t_,valid_ls_,valid_acc_,valid_t= list(),list(),list(),list(),list(),list()
	best_graph=None
	max_acc=0
	model_iter=0
	r_sess=None
	for i in range(k_fold):

		valid_input,valid_output=train_input[start:end],train_output[start:end]
		del (train_output[start:end])
		train_input=np.delete(train_input,np.s_[start:end],0)
		sess,ls_list,acc_list,t_list=training(train_output,train_input)
		if max_acc==0 :
			train_ls_=[0]*len(ls_list)
			train_acc_=[0]*len(acc_list)
			train_t_ = [0]*len(t_list)
		train_ls_=np.add(train_ls_,ls_list)
		train_acc_=np.add(train_acc_,acc_list)
		train_t_ =np.add(train_t_,t_list)
		ls_list,acc_list,t_list=validate(valid_output,valid_input,sess)
		if max_acc==0 :
			valid_ls_=[0]*len(ls_list)
			valid_acc_=[0]*len(acc_list)
			valid_t_ = [0]* len(t_list)
		valid_ls_=np.add(valid_ls_,ls_list)
		valid_acc_=np.add(valid_acc_,acc_list)
		valid_t_ =np.add(valid_t_,t_list)
		
		if(np.mean(acc_list) > max_acc):
			max_acc=np.mean(acc_list)
			model_iter=i	
			save_path=saver.save(sess,model_savepath)
			
			print("Saving model in  " + model_savepath)
			print("Saved iteration number is {}".format(model_iter))
		start=end
		end=start+offset
		train_input=np.copy(train_input_copy)
		train_output=train_output_copy[:]
		sess.close()

		
	
	# Recording the loss values accuracy values and timing data for training and validation sets
	train_loss_file=open(train_loss_file_path,'w')
	[train_loss_file.write(str(i/k_fold)+','+str(j/k_fold)+','+str(k/k_fold) + '\n') for i,j,k in zip(train_ls_.tolist(),train_acc_.tolist(),train_t_.tolist())]

	valid_loss_file=open(valid_loss_file_path,'w')
	[valid_loss_file.write(str(i/k_fold)+','+str(j/k_fold)+','+str(k/k_fold) + '\n') for i,j,k in zip(valid_ls_.tolist(),valid_acc_.tolist(),valid_t_.tolist())]

	train_loss_file.close()
	valid_loss_file.close()
	print("Training Operation Completed!!!!!")
	t4=time.time()
	if(t4-t3<3600):
		print("Training time {:.2f} minutes".format((t4-t3)/60))
	else:
		print("Training time {:.2f} hours ".format((t4-t3)/3600))
	
	
	restored_sess=tf.Session()

	print("Running on test data")
	saver.restore(restored_sess, model_savepath)
	restored_sess.run(tf.global_variables_initializer())
	print("Model restored...")
	ls_,acc_,t_=validate(test_output,test_input,restored_sess)
	test_loss_file=open(test_loss_file_path,'w')
	[test_loss_file.write(str(i/k_fold)+','+str(j/k_fold)+','+str(k/k_fold) + '\n') for i,j,k in zip(ls_,acc_,t_)]
	restored_sess.close()
	test_loss_file.close()
	
