	�<��7@�<��7@!�<��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�<��7@{��B�4@1:������?AbX9�Ȧ?I���@*	43333sH@2U
Iterator::Model::ParallelMapV2�!��u��?!�o�W��<@)�!��u��?1�o�W��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�<����9@)��_�L�?1�έ�D5@:Preprocessing2F
Iterator::Model�g��s��?!m���H�E@)�ZӼ�}?1��wc�	-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�$��?!A=S��x5@)y�&1�|?1V�nL>�,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!�O(;�RL@)��H�}m?1ɀz�r@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!V�nL>�@)y�&1�l?1V�nL>�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��uǋ-@)�J�4a?1��uǋ-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!*08͸7@)/n��R?1L�����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�0w�ѵX@Q�3"���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{��B�4@{��B�4@!{��B�4@      ��!       "	:������?:������?!:������?*      ��!       2	bX9�Ȧ?bX9�Ȧ?!bX9�Ȧ?:	���@���@!���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�0w�ѵX@y�3"���?