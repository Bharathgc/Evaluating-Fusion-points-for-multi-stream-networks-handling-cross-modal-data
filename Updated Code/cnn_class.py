import tensorflow as tf
import numpy as np

class CNN(object):
	def __init__(self, num_classes, keep_prob  ):
		super(CNN, self).__init__()
		#self.NUM_SAMPLES = num_samples
		#self.WIDTH = width
		#self.HEIGHT = height
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob

	def conv_layer_relu(self,x, weights, biases, stride, name, relu = 'TRUE', padding = 'SAME'):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable("weights", weights, initializer = tf.truncated_normal_initializer())
			biases = tf.get_variable("biases", biases, initializer = tf.truncated_normal_initializer())

			conv = tf.nn.conv2d(x, weights, strides= stride, padding = padding, name = scope.name)

			if relu == 'TRUE':
				conv = tf.nn.relu(tf.add(conv, biases), name = scope.name + "_relu")

			return conv

	def maxpool(self,x, filter_size, stride,name):
		return tf.nn.max_pool(x, ksize = filter_size, strides = stride, padding = 'VALID', name = name)

	def fc_relu(self,x, weights, biases, name, relu = 'TRUE'):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable("weights", weights, initializer = tf.truncated_normal_initializer())
			biases = tf.get_variable("biases", biases, initializer = tf.truncated_normal_initializer())

			fc = tf.add(tf.matmul(x, weights ), biases, name = scope.name)

			if relu == 'TRUE':
				fc = tf.nn.relu(fc, name = scope.name + "_relu")
			return fc

	def dropout(self,x, name):
		return tf.nn.dropout(x, self.KEEP_PROB, name = name)

	def alex_net(self, x):
		
		#reshaping into 4d tensor		
		x = tf.reshape(x , [-1, 224,224,3])

		#conv1 layer with relu
		conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
		#maxpool_1
		pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
		#normalization layer after conv1
		norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

		#conv2 layer with relu
		conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
		#maxpool_2
		pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
		#normalization after conv2
		norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

		#conv3 layer with relu
		conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
		#conv4 layer with relu
		conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
		#conv5 layer with relu
		conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
		#maxpool_2 after conv5
		pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")

		#stretching data into array for fc layers
		x2 = tf.reshape(pool3,[-1, 6*6*256])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out
		
	def alexnet_stream_2(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1 layer with relu
			conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
			#maxpool_1
			pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
			#normalization layer after conv1
			norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

			#conv2 layer with relu
			conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
			#maxpool_2
			pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
			#normalization after conv2
			norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

		return norm2

	def alexnet_bottom_2(self,norm2):
	
		#conv3 layer with relu
		conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
		#conv4 layer with relu
		conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
		#conv5 layer with relu
		conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
		#maxpool_2 after conv5
		pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")

		#stretching data into array for fc layers
		x2 = tf.reshape(pool3,[-1, 6*6*256])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out
		
	def alexnet_fused2(self,x_1,x_2):
		stream1=self.alexnet_stream_2(x_1,'stream_1') # top stream
		stream2=self.alexnet_stream_2(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.alexnet_bottom_2(fuse_point) # bottom stream
		return fused_output

	def alexnet_stream_3(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1 layer with relu
			conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
			#maxpool_1
			pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
			#normalization layer after conv1
			norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

			#conv2 layer with relu
			conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
			#maxpool_2
			pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
			#normalization after conv2
			norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

			#conv3 layer with relu
			conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
		return conv3

	def alexnet_bottom_3(self,conv3):
		#conv4 layer with relu
		conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
		#conv5 layer with relu
		conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
		#maxpool_2 after conv5
		pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")

		#stretching data into array for fc layers
		x2 = tf.reshape(pool3,[-1, 6*6*256])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out
		
	def alexnet_fused3(self,x_1,x_2):
		stream1=self.alexnet_stream_3(x_1,'stream_1') # top stream
		stream2=self.alexnet_stream_3(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.alexnet_bottom_3(fuse_point) # bottom stream
		return fused_output

	def alexnet_stream_4(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1 layer with relu
			conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
			#maxpool_1
			pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
			#normalization layer after conv1
			norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

			#conv2 layer with relu
			conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
			#maxpool_2
			pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
			#normalization after conv2
			norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

			#conv3 layer with relu
			conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
			#conv4 layer with relu
			conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
		
		return conv4

	def alexnet_bottom_4(self,conv4):
		
		#conv5 layer with relu
		conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
		#maxpool_2 after conv5
		pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")

		#stretching data into array for fc layers
		x2 = tf.reshape(pool3,[-1, 6*6*256])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out
		
	def alexnet_fused4(self,x_1,x_2):
		stream1=self.alexnet_stream_4(x_1,'stream_1') # top stream
		stream2=self.alexnet_stream_4(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.alexnet_bottom_4(fuse_point) # bottom stream
		return fused_output

	def alexnet_stream_5(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1 layer with relu
			conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
			#maxpool_1
			pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
			#normalization layer after conv1
			norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

			#conv2 layer with relu
			conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
			#maxpool_2
			pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
			#normalization after conv2
			norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

			#conv3 layer with relu
			conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
			#conv4 layer with relu
			conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
			#conv5 layer with relu
			conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
			#maxpool_2 after conv5
			pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")
		
		return pool3

	def alexnet_bottom_5(self,pool3):
		
		#stretching data into array for fc layers
		x2 = tf.reshape(pool3,[-1, 6*6*256])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out
		
	def alexnet_fused5(self,x_1,x_2):
		stream1=self.alexnet_stream_5(x_1,'stream_1') # top stream
		stream2=self.alexnet_stream_5(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.alexnet_bottom_5(fuse_point) # bottom stream
		return fused_output

	def alexnet_stream_6(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1 layer with relu
			conv1 = self.conv_layer_relu(x, [11,11,3,96], [96], [1,4,4,1], "alex_conv1")
			#maxpool_1
			pool1 = self.maxpool(conv1,[1,3,3,1], [1,2,2,1], "alex_pool1")
			#normalization layer after conv1
			norm1 = tf.nn.local_response_normalization(pool1, name = "alex_norm1")

			#conv2 layer with relu
			conv2 = self.conv_layer_relu(norm1, [5,5,96,256], [256], [1,1,1,1], "alex_conv2")
			#maxpool_2
			pool2 = self.maxpool(conv2,[1,3,3,1], [1,2,2,1], "alex_pool2")
			#normalization after conv2
			norm2 = tf.nn.local_response_normalization(pool2, name = "alex_norm2")

			#conv3 layer with relu
			conv3 = self.conv_layer_relu(norm2, [3,3,256,384], [384], [1,1,1,1], "alex_conv3")
			#conv4 layer with relu
			conv4 = self.conv_layer_relu(conv3, [3,3,384,384], [384], [1,1,1,1], "alex_conv4")
			#conv5 layer with relu
			conv5 = self.conv_layer_relu(conv4, [3,3,384,256], [256], [1,1,1,1], "alex_conv5")
			#maxpool_2 after conv5
			pool3 = self.maxpool(conv5,[1,3,3,1], [1,2,2,1], "alex_pool3")

			#stretching data into array for fc layers
			x2 = tf.reshape(pool3,[-1, 6*6*256],name='alex_linear')

			#fc6 with relu
			fc6 = self.fc_relu(x2, [6*6*256, 4096], [4096], "alex_fc6")
			#dropout for fc6
			dropout_fc6 = self.dropout(fc6, "alex_drop_fc6")

		return dropout_fc6
	
	def alexnet_bottom_6(self,dropout_fc6):
		
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "alex_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "alex_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES], [self.NUM_CLASSES], "alex_out", relu = 'FALSE')

		return out

	def alexnet_fused6(self,x_1,x_2):
		stream1=self.alexnet_stream_6(x_1,'stream_1') # top stream
		stream2=self.alexnet_stream_6(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.alexnet_bottom_6(fuse_point) # bottom stream
		return fused_output
		
	#Implementation of VGG net
	def vgg_net(self,x):

		#reshaping into 4d tensor		
		x = tf.reshape(x , [-1, 224,224,3])

		#conv1_1 layer with relu
		conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
		#conv1_2 layer with relu
		conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
		#maxpool 1 
		pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
		#norm layer after pool1
		norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

		#conv2_1 layer with relu
		conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
		#conv2_2 layer with relu
		conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
		#maxpool 2 
		pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
		#norm layer after pool2
		norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")

		#conv3_1 layer with relu
		conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
		#conv3_2 layer with relu
		conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
		#conv3_3 layer with relu
		conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
		#maxpool 3 
		pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
		#norm layer after pool3
		norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")

		#conv4_1 layer with relu
		conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
		#conv4_2 layer with relu
		conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
		#conv4_3 layer with relu
		conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
		#maxpool 4 
		pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
		#norm layer after pool4
		norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")

		#conv5_1 layer with relu
		conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
		#conv5_2 layer with relu
		conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
		#conv5_3 layer with relu
		conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
		#maxpool 5
		pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
		
		#reshaping for fc layers
		x2 = tf.reshape(pool5, [-1, 7*7*512])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out
		
	def vggnet_stream_2(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1_1 layer with relu
			conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
			#conv1_2 layer with relu
			conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
			#maxpool 1 
			pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
			#norm layer after pool1
			norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

			#conv2_1 layer with relu
			conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
			#conv2_2 layer with relu
			conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
			#maxpool 2 
			pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
			#norm layer after pool2
			norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")
			
		return norm2
		
	def vggnet_bottom_2(self,norm2):
		
		#conv3_1 layer with relu
		conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
		#conv3_2 layer with relu
		conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
		#conv3_3 layer with relu
		conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
		#maxpool 3 
		pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
		#norm layer after pool3
		norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")
					
		#conv4_1 layer with relu
		conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
		#conv4_2 layer with relu
		conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
		#conv4_3 layer with relu
		conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
		#maxpool 4 
		pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
		#norm layer after pool4
		norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")

		#conv5_1 layer with relu
		conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
		#conv5_2 layer with relu
		conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
		#conv5_3 layer with relu
		conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
		#maxpool 5
		pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
		
		#reshaping for fc layers
		x2 = tf.reshape(pool5, [-1, 7*7*512])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out
		
	def vggnet_fused2(self,x_1,x_2):
		stream1=self.vggnet_stream_2(x_1,'stream_1') # top stream
		stream2=self.vggnet_stream_2(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.vggnet_bottom_2(fuse_point) # bottom stream
		return fused_output
		
	def vggnet_stream_3(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1_1 layer with relu
			conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
			#conv1_2 layer with relu
			conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
			#maxpool 1 
			pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
			#norm layer after pool1
			norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

			#conv2_1 layer with relu
			conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
			#conv2_2 layer with relu
			conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
			#maxpool 2 
			pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
			#norm layer after pool2
			norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")

			#conv3_1 layer with relu
			conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
			#conv3_2 layer with relu
			conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
			#conv3_3 layer with relu
			conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
			#maxpool 3 
			pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
			#norm layer after pool3
			norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")
			
		return norm3
		
	def vggnet_bottom_3(self,norm3):
			
		#conv4_1 layer with relu
		conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
		#conv4_2 layer with relu
		conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
		#conv4_3 layer with relu
		conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
		#maxpool 4 
		pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
		#norm layer after pool4
		norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")

		#conv5_1 layer with relu
		conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
		#conv5_2 layer with relu
		conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
		#conv5_3 layer with relu
		conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
		#maxpool 5
		pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
		
		#reshaping for fc layers
		x2 = tf.reshape(pool5, [-1, 7*7*512])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out
		
	def vggnet_fused3(self,x_1,x_2):
		stream1=self.vggnet_stream_3(x_1,'stream_1') # top stream
		stream2=self.vggnet_stream_3(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.vggnet_bottom_3(fuse_point) # bottom stream
		return fused_output
		
	def vggnet_stream_4(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1_1 layer with relu
			conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
			#conv1_2 layer with relu
			conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
			#maxpool 1 
			pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
			#norm layer after pool1
			norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

			#conv2_1 layer with relu
			conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
			#conv2_2 layer with relu
			conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
			#maxpool 2 
			pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
			#norm layer after pool2
			norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")

			#conv3_1 layer with relu
			conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
			#conv3_2 layer with relu
			conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
			#conv3_3 layer with relu
			conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
			#maxpool 3 
			pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
			#norm layer after pool3
			norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")
			
			#conv4_1 layer with relu
			conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
			#conv4_2 layer with relu
			conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
			#conv4_3 layer with relu
			conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
			#maxpool 4 
			pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
			#norm layer after pool4
			norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")
			
		return norm4
		
	def vggnet_bottom_4(self,norm4):

		#conv5_1 layer with relu
		conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
		#conv5_2 layer with relu
		conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
		#conv5_3 layer with relu
		conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
		#maxpool 5
		pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
		
		#reshaping for fc layers
		x2 = tf.reshape(pool5, [-1, 7*7*512])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out
	
	def vggnet_fused4(self,x_1,x_2):
		stream1=self.vggnet_stream_4(x_1,'stream_1') # top stream
		stream2=self.vggnet_stream_4(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.vggnet_bottom_4(fuse_point) # bottom stream
		return fused_output
	
	def vggnet_stream_5(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1_1 layer with relu
			conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
			#conv1_2 layer with relu
			conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
			#maxpool 1 
			pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
			#norm layer after pool1
			norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

			#conv2_1 layer with relu
			conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
			#conv2_2 layer with relu
			conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
			#maxpool 2 
			pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
			#norm layer after pool2
			norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")

			#conv3_1 layer with relu
			conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
			#conv3_2 layer with relu
			conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
			#conv3_3 layer with relu
			conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
			#maxpool 3 
			pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
			#norm layer after pool3
			norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")
			
			#conv4_1 layer with relu
			conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
			#conv4_2 layer with relu
			conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
			#conv4_3 layer with relu
			conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
			#maxpool 4 
			pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
			#norm layer after pool4
			norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")
			
			#conv5_1 layer with relu
			conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
			#conv5_2 layer with relu
			conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
			#conv5_3 layer with relu
			conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
			#maxpool 5
			pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
			
		return pool5
		
	def vggnet_bottom_5(self,pool5):

		#reshaping for fc layers
		x2 = tf.reshape(pool5, [-1, 7*7*512])

		#fc6 with relu
		fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
		#dropout for fc6
		dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out
		
	def vggnet_fused5(self,x_1,x_2):
		stream1=self.vggnet_stream_5(x_1,'stream_1') # top stream
		stream2=self.vggnet_stream_5(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.vggnet_bottom_5(fuse_point) # bottom stream
		return fused_output
		
	def vggnet_stream_6(self,x,scope_name):
		with tf.variable_scope(scope_name):
			#reshaping into 4d tensor		
			x = tf.reshape(x , [-1, 224,224,3])

			#conv1_1 layer with relu
			conv1_1 = self.conv_layer_relu(x, [3,3,3,64], [64], [1,1,1,1], "vgg_conv1_1")
			#conv1_2 layer with relu
			conv1_2 = self.conv_layer_relu(conv1_1, [3,3,64,64], [64], [1,1,1,1], "vgg_conv1_2")
			#maxpool 1 
			pool1 = self.maxpool(conv1_2,[1,2,2,1], [1,2,2,1], "vgg_pool1")
			#norm layer after pool1
			norm1 = tf.nn.local_response_normalization(pool1, name = "vgg_norm1")

			#conv2_1 layer with relu
			conv2_1 = self.conv_layer_relu(norm1, [3,3,64,128], [128], [1,1,1,1], "vgg_conv2_1")
			#conv2_2 layer with relu
			conv2_2 = self.conv_layer_relu(conv2_1, [3,3,128,128], [128], [1,1,1,1], "vgg_conv2_2")
			#maxpool 2 
			pool2 = self.maxpool(conv2_2,[1,2,2,1], [1,2,2,1], "vgg_pool2")
			#norm layer after pool2
			norm2 = tf.nn.local_response_normalization(pool2, name = "vgg_norm2")

			#conv3_1 layer with relu
			conv3_1 = self.conv_layer_relu(norm2, [3,3,128,256], [256], [1,1,1,1], "vgg_conv3_1")
			#conv3_2 layer with relu
			conv3_2 = self.conv_layer_relu(conv3_1, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_2")
			#conv3_3 layer with relu
			conv3_3 = self.conv_layer_relu(conv3_2, [3,3,256,256], [256], [1,1,1,1], "vgg_conv3_3")
			#maxpool 3 
			pool3 = self.maxpool(conv3_3,[1,2,2,1], [1,2,2,1], "vgg_pool3")
			#norm layer after pool3
			norm3 = tf.nn.local_response_normalization(pool3, name = "vgg_norm3")
			
			#conv4_1 layer with relu
			conv4_1 = self.conv_layer_relu(norm3, [3,3,256,512], [512], [1,1,1,1], "vgg_conv4_1")
			#conv4_2 layer with relu
			conv4_2 = self.conv_layer_relu(conv4_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_2")
			#conv4_3 layer with relu
			conv4_3 = self.conv_layer_relu(conv4_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv4_3")
			#maxpool 4 
			pool4 = self.maxpool(conv4_3,[1,2,2,1], [1,2,2,1], "vgg_pool4")
			#norm layer after pool4
			norm4 = tf.nn.local_response_normalization(pool4, name = "vgg_norm4")
			
			#conv5_1 layer with relu
			conv5_1 = self.conv_layer_relu(norm4, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_1")
			#conv5_2 layer with relu
			conv5_2 = self.conv_layer_relu(conv5_1, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_2")
			#conv5_3 layer with relu
			conv5_3 = self.conv_layer_relu(conv5_2, [3,3,512,512], [512], [1,1,1,1], "vgg_conv5_3")
			#maxpool 5
			pool5 = self.maxpool(conv5_3,[1,2,2,1], [1,2,2,1], "vgg_pool5")
			
			#reshaping for fc layers
			x2 = tf.reshape(pool5, [-1, 7*7*512])

			#fc6 with relu
			fc6 = self.fc_relu(x2, [7*7*512, 4096], [4096], "vgg_fc6")
			#dropout for fc6
			dropout_fc6 = self.dropout(fc6, "vgg_drop_fc6")
			
		return dropout_fc6
		
	def vggnet_bottom_6(self,dropout_fc6):
		#fc7 with relu
		fc7 = self.fc_relu(dropout_fc6, [4096,4096], [4096], "vgg_fc7")
		#dropout for fc7
		dropout_fc7 = self.dropout(fc7, "vgg_drop_fc7")

		#fc8 or output WITHOUT acivation and dropout
		out = self.fc_relu(dropout_fc7, [4096, self.NUM_CLASSES],[self.NUM_CLASSES], "vgg_out", relu = 'FALSE')

		return out
		
	def vggnet_fused6(self,x_1,x_2):
		stream1=self.vggnet_stream_6(x_1,'stream_1') # top stream
		stream2=self.vggnet_stream_6(x_2,'stream_2') # top stream
		#fusion by averaging
		fuse_point=tf.add(stream1,stream2)
		fuse_point=tf.scalar_mul(1.0/2.0,fuse_point)
		# joining the network
		fused_output=self.vggnet_bottom_6(fuse_point) # bottom stream
		return fused_output