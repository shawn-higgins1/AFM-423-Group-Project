	�d73��5@�d73��5@!�d73��5@	�	�S,�?�	�S,�?!�	�S,�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�d73��5@�R	[3@1;���R��?A�Ӻj��?Ip��m @Y�Z^��6s?*	�����LK@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate$����ۗ?!dEVdEVE@)��_vO�?1|��{��C@:Preprocessing2U
Iterator::Model::ParallelMapV2V-��?!鈎舎:@)V-��?1鈎舎:@:Preprocessing2F
Iterator::Model/�$��?!�;�;C@)9��v��z?1�<��<�'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!wqwq'@)HP�s�r?1��٘�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�4�?!O��N��N@)Ǻ���f?14H�4H�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��_��_
@)��H�}]?1��_��_
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!�%X�%XF@)/n��R?1�� @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!�� @)/n��R?1�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!R�Q��?)a2U0*�C?1R�Q��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�	�S,�?I��5�B�X@Q��E��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�R	[3@�R	[3@!�R	[3@      ��!       "	;���R��?;���R��?!;���R��?*      ��!       2	�Ӻj��?�Ӻj��?!�Ӻj��?:	p��m @p��m @!p��m @B      ��!       J	�Z^��6s?�Z^��6s?!�Z^��6s?R      ��!       Z	�Z^��6s?�Z^��6s?!�Z^��6s?b      ��!       JGPUY�	�S,�?b q��5�B�X@y��E��?