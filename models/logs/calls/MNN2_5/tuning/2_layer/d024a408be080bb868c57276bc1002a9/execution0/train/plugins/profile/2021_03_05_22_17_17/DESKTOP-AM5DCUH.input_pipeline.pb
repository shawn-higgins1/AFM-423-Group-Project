	sf�B�5@sf�B�5@!sf�B�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-sf�B�5@#�� 3@1�P��C��?Ak�w��#�?I	O��'�@*	������G@2U
Iterator::Model::ParallelMapV2�]K�=�?!���/�-<@)�]K�=�?1���/�-<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!h8����9@)��ׁsF�?14��}�4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!d+����7@)��H�}}?1���c+�.@:Preprocessing2F
Iterator::Model��ZӼ�?!`[4�E@)�ZӼ�}?1pR��.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!"h8��� @)����Mbp?1"h8��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!����cL@)_�Q�k?1�}ylE�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?![4��}@)HP�s�b?1[4��}@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI ���X@Q w�r��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#�� 3@#�� 3@!#�� 3@      ��!       "	�P��C��?�P��C��?!�P��C��?*      ��!       2	k�w��#�?k�w��#�?!k�w��#�?:		O��'�@	O��'�@!	O��'�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q ���X@y w�r��?