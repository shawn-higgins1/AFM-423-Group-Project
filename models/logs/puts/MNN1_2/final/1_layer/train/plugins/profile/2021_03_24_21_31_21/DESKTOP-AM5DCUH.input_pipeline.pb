	���6@���6@!���6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���6@�>U��3@1HP�s��?A�D���J�?I3nj���@*	433333K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenater�����?!�����D@@)�!��u��?1LKKKK�9@:Preprocessing2U
Iterator::Model::ParallelMapV2_�Qڋ?!      9@)_�Qڋ?1      9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!������7@)46<�R�?1jiiii	4@:Preprocessing2F
Iterator::Model���<,�?!KKKKKB@)�HP�x?1-----m&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!yxxxxx@)��H�}m?1yxxxxx@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��W�2ġ?!������O@)�����g?1�����R@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!jiiiii@)����Mb`?1jiiiii@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ݓ��Z�?!_A@)a2U0*�S?1������@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�
5(�X@Q�z�ku�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�>U��3@�>U��3@!�>U��3@      ��!       "	HP�s��?HP�s��?!HP�s��?*      ��!       2	�D���J�?�D���J�?!�D���J�?:	3nj���@3nj���@!3nj���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�
5(�X@y�z�ku�?