	ra�v3@ra�v3@!ra�v3@	��~��H�?��~��H�?!��~��H�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ra�v3@�_���1@1p$�`S��?A�C�����?I��%VF#�?YPj��?*	�����G@2U
Iterator::Model::ParallelMapV2�������?!�9T,h;@)�������?1�9T,h;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!5�wL�;@)��_vO�?1���cj`7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!>���>8@)���Q�~?1`��;0@:Preprocessing2F
Iterator::Model�ݓ��Z�?!u�E]tD@)-C��6z?1��n��+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!���n @)���_vOn?1���n @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!�.�袋M@)-C��6j?1��n��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!/�袋.@)�J�4a?1/�袋.@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��~��H�?Iͣ3n+�X@Q�X����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�_���1@�_���1@!�_���1@      ��!       "	p$�`S��?p$�`S��?!p$�`S��?*      ��!       2	�C�����?�C�����?!�C�����?:	��%VF#�?��%VF#�?!��%VF#�?B      ��!       J	Pj��?Pj��?!Pj��?R      ��!       Z	Pj��?Pj��?!Pj��?b      ��!       JGPUY��~��H�?b qͣ3n+�X@y�X����?