	=b��Bo5@=b��Bo5@!=b��Bo5@	�*�����?�*�����?!�*�����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6=b��Bo5@0H���2@1�Wya�?AM�St$�?I"�T3k	@Yq�GR��p?*	�����YI@2U
Iterator::Model::ParallelMapV2%u��?!�T�6|�<@)%u��?1�T�6|�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!ST���8@)/�$��?1�<pƵ4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-C��6�?!|1z�?9@)�J�4�?1z0��k�0@:Preprocessing2F
Iterator::ModelQ�|a2�?!θ	jD@)�~j�t�x?1d���+�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n��r?!4{d[!@)/n��r?14{d[!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���Q��?!�1G���M@)F%u�k?1θ	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!۽=�@)����Mb`?1۽=�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ZӼ��?!原<@)Ǻ���V?1L�*:@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�*�����?I:�0���X@Q��2���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0H���2@0H���2@!0H���2@      ��!       "	�Wya�?�Wya�?!�Wya�?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	"�T3k	@"�T3k	@!"�T3k	@B      ��!       J	q�GR��p?q�GR��p?!q�GR��p?R      ��!       Z	q�GR��p?q�GR��p?!q�GR��p?b      ��!       JGPUY�*�����?b q:�0���X@y��2���?