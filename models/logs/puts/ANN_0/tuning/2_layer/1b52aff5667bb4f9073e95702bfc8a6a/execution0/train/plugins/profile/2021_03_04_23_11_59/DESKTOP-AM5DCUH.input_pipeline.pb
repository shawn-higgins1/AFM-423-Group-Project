	 p��s�3@ p��s�3@! p��s�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails- p��s�3@�_?�o1@1*�D/�X�?A�3���?I,�z��M @*�����YH@)       =2U
Iterator::Model::ParallelMapV2���Q��?!y�ڠ�>@)���Q��?1y�ڠ�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�X�?!p���h7@)U���N@�?1O��EM3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!ƊH�8@)vq�-�?1�����80@:Preprocessing2F
Iterator::Model�e��a��?!�2�͟�F@)�ZӼ�}?1qo�y(-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!�+K�x� @)	�^)�p?1�+K�x� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�]K�=�?!�v2`OK@)-C��6j?1;ǳƊH@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�\0�Vm@)����Mb`?1�\0�Vm@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�	g\D�X@QA�=���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�_?�o1@�_?�o1@!�_?�o1@      ��!       "	*�D/�X�?*�D/�X�?!*�D/�X�?*      ��!       2	�3���?�3���?!�3���?:	,�z��M @,�z��M @!,�z��M @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�	g\D�X@yA�=���?