	Ym�_u�D@Ym�_u�D@!Ym�_u�D@	�֘^\`�?�֘^\`�?!�֘^\`�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Ym�_u�D@�lscz�B@1�"�k$�?Aq���h�?I����W
@Y�����?*	effff�K@2U
Iterator::Model::ParallelMapV2����Mb�?!�9�!��<@)����Mb�?1�9�!��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ǘ���?!�*uK=@)�Pk�w�?1EZ�>�"9@:Preprocessing2F
Iterator::ModeltF��_�?!
�Z܄E@)�q����?1lXY'�5,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��_�L�?!��ow�2@)F%u�{?1��H��'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceŏ1w-!o?!�v�,�|@)ŏ1w-!o?1�v�,�|@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!�D�#{L@)��H�}m?1�8/
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!FA;��@)HP�s�b?1FA;��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!Q�]F(5@)��_�LU?1��ow�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�֘^\`�?I�L��~�X@Qr�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�lscz�B@�lscz�B@!�lscz�B@      ��!       "	�"�k$�?�"�k$�?!�"�k$�?*      ��!       2	q���h�?q���h�?!q���h�?:	����W
@����W
@!����W
@B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?b      ��!       JGPUY�֘^\`�?b q�L��~�X@yr�����?