	_%��7@_%��7@!_%��7@	�Ӥ�f�?�Ӥ�f�?!�Ӥ�f�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6_%��7@�ާ�А4@1�f*�#��?A��?�߮?I+�ƈ@Y��f�W�?*	ffffffI@2U
Iterator::Model::ParallelMapV2�?�߾�?!�~����:@)�?�߾�?1�~����:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!      9@)/�$��?1k�Z�V�4@:Preprocessing2F
Iterator::Model䃞ͪϕ?!�n����D@)ŏ1w-!?1|�^���-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�{�Pk�?!�L&��d9@)���Q�~?1�p8�-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��_vOv?!Q(
�B%@)��_vOv?1Q(
�B%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph��|?5�?!"�H$	M@)��_�Le?1����x@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!U*�J�R@)/n��b?1U*�J�R@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ZӼ��?!}>����;@)��_�LU?1����x@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Ӥ�f�?I_���X@QS�	*j�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ާ�А4@�ާ�А4@!�ާ�А4@      ��!       "	�f*�#��?�f*�#��?!�f*�#��?*      ��!       2	��?�߮?��?�߮?!��?�߮?:	+�ƈ@+�ƈ@!+�ƈ@B      ��!       J	��f�W�?��f�W�?!��f�W�?R      ��!       Z	��f�W�?��f�W�?!��f�W�?b      ��!       JGPUY�Ӥ�f�?b q_���X@yS�	*j�?