	2�	��4@2�	��4@!2�	��4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2�	��4@����52@1�k�,	P�?A�ZӼ��?I�����?*	23333�J@2U
Iterator::Model::ParallelMapV2���Q��?!��s�M�;@)���Q��?1��s�M�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!�	�^;9@)g��j+��?1�y��5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q����?!�{��F�<@)������?1��qCv�5@:Preprocessing2F
Iterator::Model'�����?!�-� �C@)-C��6z?1|��h�'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!��B�@)����Mbp?1��B�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�p=
ף�?! ���%N@)F%u�k?1�Cc}@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!R��K3@)ŏ1w-!_?1R��K3@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�8@���X@Q����B �?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����52@����52@!����52@      ��!       "	�k�,	P�?�k�,	P�?!�k�,	P�?*      ��!       2	�ZӼ��?�ZӼ��?!�ZӼ��?:	�����?�����?!�����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�8@���X@y����B �?