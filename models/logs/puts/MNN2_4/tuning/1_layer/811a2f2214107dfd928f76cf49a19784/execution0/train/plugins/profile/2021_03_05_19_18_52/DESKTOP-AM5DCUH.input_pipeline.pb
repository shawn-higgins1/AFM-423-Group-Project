	l��7�5@l��7�5@!l��7�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-l��7�5@�쟧3@1	Q����?A���Mb�?I�>�G�@*	33333�F@2U
Iterator::Model::ParallelMapV2tF��_�?!B7%�!6:@)tF��_�?1B7%�!6:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!���"k�8@)HP�sׂ?1�nJ�C4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��0�*�?!s���9@)�q����?1���<.1@:Preprocessing2F
Iterator::Modelr�����?!F�V�N~C@)�����w?1����)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!TRb�!@)����Mbp?1TRb�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!�)���N@)F%u�k?1<.}�+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!TRb�@)����Mb`?1TRb�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u��?!<.}�+=@)Ǻ���V?1���"k�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI
A��X@Q����C�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�쟧3@�쟧3@!�쟧3@      ��!       "		Q����?	Q����?!	Q����?*      ��!       2	���Mb�?���Mb�?!���Mb�?:	�>�G�@�>�G�@!�>�G�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q
A��X@y����C�?