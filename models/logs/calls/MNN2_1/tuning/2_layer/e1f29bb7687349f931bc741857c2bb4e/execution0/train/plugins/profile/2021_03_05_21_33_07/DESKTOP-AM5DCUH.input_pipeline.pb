	��&��7@��&��7@!��&��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��&��7@���S�q5@1֐��҇�?A�ׁsF��?I�����@*	�����LM@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�D���J�?!����8E@)��&��?1�j�ٻ�C@:Preprocessing2U
Iterator::Model::ParallelMapV2-C��6�?!�Y��5@)-C��6�?1�Y��5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��_�L�?!�h�>X�1@)�J�4�?1#��?�,@:Preprocessing2F
Iterator::ModelΈ����?!p��昽?@)�����w?1�à���#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{�G�z�?!�UƙQ@)��H�}m?1gB��Ȓ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!:f���M@)����Mb`?1:f���M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9��v���?!��N./F@)��_�LU?1�h�>X�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!:f���M�?)����MbP?1:f���M�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!���M�a�?)a2U0*�C?1���M�a�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIN��L��X@Qd���R�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���S�q5@���S�q5@!���S�q5@      ��!       "	֐��҇�?֐��҇�?!֐��҇�?*      ��!       2	�ׁsF��?�ׁsF��?!�ׁsF��?:	�����@�����@!�����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qN��L��X@yd���R�?