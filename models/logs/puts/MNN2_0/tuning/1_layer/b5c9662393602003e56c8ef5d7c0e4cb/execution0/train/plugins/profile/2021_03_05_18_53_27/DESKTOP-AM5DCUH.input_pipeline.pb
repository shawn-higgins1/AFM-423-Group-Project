	�S�Ƶ5@�S�Ƶ5@!�S�Ƶ5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�S�Ƶ5@DԷ�%3@1 c�ZB>�?AA��ǘ��?I��);��@*	������K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateHP�s�?!h��%�D@)^K�=��?1�s��C@:Preprocessing2U
Iterator::Model::ParallelMapV2�Pk�w�?!ہ�v`.9@)�Pk�w�?1ہ�v`.9@:Preprocessing2F
Iterator::Model8��d�`�?!��%~B@)�~j�t�x?1����7�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipP�s��?!k�ځ�O@){�G�zt?1��v`�"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t�x?!����7�%@)ŏ1w-!o?1�%~F�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!}F��Q�@)/n��b?1}F��Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��+e�?!d!Y�F@)Ǻ���V?1��(�3J@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!}F��Q��?)/n��R?1}F��Q��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��(�3J�?)Ǻ���F?1��(�3J�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�W�P��X@QC8ԤW8�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	DԷ�%3@DԷ�%3@!DԷ�%3@      ��!       "	 c�ZB>�? c�ZB>�?! c�ZB>�?*      ��!       2	A��ǘ��?A��ǘ��?!A��ǘ��?:	��);��@��);��@!��);��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�W�P��X@yC8ԤW8�?