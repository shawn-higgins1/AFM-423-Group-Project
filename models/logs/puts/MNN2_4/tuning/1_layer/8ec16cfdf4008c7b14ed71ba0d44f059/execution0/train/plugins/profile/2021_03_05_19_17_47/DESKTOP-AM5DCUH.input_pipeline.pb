	0���P@0���P@!0���P@	y�KKV�?y�KKV�?!y�KKV�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60���P@ �ҥ�N@1�������?AHP�s�?Ia�$�s @Yz�m�(�?*	      J@2U
Iterator::Model::ParallelMapV2a��+e�?!��؉��7@)a��+e�?1��؉��7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%u��?!O��N�D<@)a��+e�?1��؉��7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF%u��?!'vb'vb9@)Έ����?1'vb'v�1@:Preprocessing2F
Iterator::Model�N@aÓ?!��N�ĎB@)lxz�,C|?1�؉�؉*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!      @)�q����o?1      @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�(���?!;�;qO@)���_vOn?1b'vb'v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!;�;�@)HP�s�b?1;�;�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���QI�?!    �;@)/n��R?1�N��N� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9y�KKV�?IW��?�X@Qe�8�N��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 �ҥ�N@ �ҥ�N@! �ҥ�N@      ��!       "	�������?�������?!�������?*      ��!       2	HP�s�?HP�s�?!HP�s�?:	a�$�s @a�$�s @!a�$�s @B      ��!       J	z�m�(�?z�m�(�?!z�m�(�?R      ��!       Z	z�m�(�?z�m�(�?!z�m�(�?b      ��!       JGPUYy�KKV�?b qW��?�X@ye�8�N��?