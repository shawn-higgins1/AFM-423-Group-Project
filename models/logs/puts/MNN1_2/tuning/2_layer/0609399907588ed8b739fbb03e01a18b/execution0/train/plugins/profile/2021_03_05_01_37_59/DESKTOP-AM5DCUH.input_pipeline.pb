	�g��b7@�g��b7@!�g��b7@	q̄��7�?q̄��7�?!q̄��7�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�g��b7@�vi��4@1}w+Kt�?A�+e�X�?I��9̗@Y�]��y�t?*	gfffffM@2U
Iterator::Model::ParallelMapV2r�����?!���X>@)r�����?1���X>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!�?!w�"��9@)Zd;�O��?1�!͎3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?!3q�{�6@)�+e�X�?1Y�Cc3@:Preprocessing2F
Iterator::Modelc�ZB>�?!�2q�{�E@)ŏ1w-!?1w�"��)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!K���+@)���_vOn?1K���+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���x�&�?!͎Z�|L@)_�Q�k?1�_��!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!ΎZ��5@)����Mb`?1ΎZ��5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o_��?!�Q���f<@)�~j�t�X?1�Cc}h@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9q̄��7�?I��=�X@Q��&QG�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�vi��4@�vi��4@!�vi��4@      ��!       "	}w+Kt�?}w+Kt�?!}w+Kt�?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	��9̗@��9̗@!��9̗@B      ��!       J	�]��y�t?�]��y�t?!�]��y�t?R      ��!       Z	�]��y�t?�]��y�t?!�]��y�t?b      ��!       JGPUYq̄��7�?b q��=�X@y��&QG�?