	���$l3@���$l3@!���$l3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���$l3@�/J�_�1@1	�/�
�?A-����?IøDk��?*	    �I@2U
Iterator::Model::ParallelMapV2� �	��?!-�H%�=@)� �	��?1-�H%�=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u��?!�@*9/�<@)�g��s��?1���O �4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!�[mM7@)��ZӼ�?1Z�'�J�3@:Preprocessing2F
Iterator::Model�e��a��?!a����tE@)S�!�uq{?1+9/��*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!�6��;�@)	�^)�p?1�6��;�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!�@*9/�L@)��_vOf?1A*9/��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�����@)��H�}]?1�����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����X@Q��U{���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�/J�_�1@�/J�_�1@!�/J�_�1@      ��!       "		�/�
�?	�/�
�?!	�/�
�?*      ��!       2	-����?-����?!-����?:	øDk��?øDk��?!øDk��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����X@y��U{���?