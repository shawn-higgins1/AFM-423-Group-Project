	��bC8@��bC8@!��bC8@	_������?_������?!_������?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��bC8@0�1"Q�5@1uU����?A��3���?I�[!�Ʋ@Y��'*֬?*	333333J@2U
Iterator::Model::ParallelMapV2%u��?!}@u�<@)%u��?1}@u�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF%u��?!T���09@)Ǻ����?1k�4w�_5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���S㥋?! P{k�9@)HP�sׂ?1�2���1@:Preprocessing2F
Iterator::Model��A�f�?!1�]�W�C@)a��+ey?1ɳ���'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!�:�ֆi @)"��u��q?1�:�ֆi @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!�2��N@)F%u�k?1T���0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!ㄔ<ˈ@)����Mb`?1ㄔ<ˈ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Q��?!�<ˈ>�<@)�~j�t�X?1�com��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9_������?Ii
���X@QHl]d'�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0�1"Q�5@0�1"Q�5@!0�1"Q�5@      ��!       "	uU����?uU����?!uU����?*      ��!       2	��3���?��3���?!��3���?:	�[!�Ʋ@�[!�Ʋ@!�[!�Ʋ@B      ��!       J	��'*֬?��'*֬?!��'*֬?R      ��!       Z	��'*֬?��'*֬?!��'*֬?b      ��!       JGPUY_������?b qi
���X@yHl]d'�?