	�- ��6@�- ��6@!�- ��6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�- ��6@Ov3�W4@1_�sa��?A�{��?I֎�ut@*	hffff�F@2U
Iterator::Model::ParallelMapV2��0�*�?!�1��n:@)��0�*�?1�1��n:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*�?!�1��n:@)��ׁsF�?1�jg���5@:Preprocessing2F
Iterator::Model�ݓ��Z�?!�q˸e�D@)�ZӼ�}?1�b\�X/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_�L�?!��?C�6@)9��v��z?1Q����,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!���o�7!@)�q����o?1���o�7!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF%u��?!i�4G�#M@)a��+ei?1~�i�_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!��8��@)ŏ1w-!_?1��8��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!��up�b9@)/n��R?1F�̈́m@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�ǅ-��X@Q����Q�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ov3�W4@Ov3�W4@!Ov3�W4@      ��!       "	_�sa��?_�sa��?!_�sa��?*      ��!       2	�{��?�{��?!�{��?:	֎�ut@֎�ut@!֎�ut@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�ǅ-��X@y����Q�?