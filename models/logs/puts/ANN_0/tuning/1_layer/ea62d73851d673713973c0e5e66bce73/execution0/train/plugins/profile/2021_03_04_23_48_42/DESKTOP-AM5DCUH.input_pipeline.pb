	���{x2@���{x2@!���{x2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���{x2@,D���(1@1xak���?AA��ǘ��?IȘ�����?*	33333�J@2U
Iterator::Model::ParallelMapV2���_vO�?!<�܎!�;@)���_vO�?1<�܎!�;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���{�?!��r;�>@)a��+e�?1�F⼑87@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!��^���9@)�~j�t��?1g<��x6@:Preprocessing2F
Iterator::Model��ZӼ�?!�g<�C@)Ǻ���v?1P'��I�$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!S��Ԧ6@)�q����o?1S��Ԧ6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL7�A`�?!s���N@)�����g?1�1�v�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�{��^�
@)��H�}]?1�{��^�
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIx		��X@Q�C{�3�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	,D���(1@,D���(1@!,D���(1@      ��!       "	xak���?xak���?!xak���?*      ��!       2	A��ǘ��?A��ǘ��?!A��ǘ��?:	Ș�����?Ș�����?!Ș�����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qx		��X@y�C{�3�?