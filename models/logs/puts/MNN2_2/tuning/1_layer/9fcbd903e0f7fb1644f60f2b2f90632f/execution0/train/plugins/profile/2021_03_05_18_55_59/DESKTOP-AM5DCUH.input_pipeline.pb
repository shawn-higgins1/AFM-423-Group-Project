	��mC5@��mC5@!��mC5@	=�~���?=�~���?!=�~���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��mC5@��8G3@1����?A,e�X�?Iq9^���?Y�lXSY�?*	�����G@2U
Iterator::Model::ParallelMapV2a��+e�?!�[K���:@)a��+e�?1�[K���:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�9T,h;@)�g��s��?1_ƫӗ�6@:Preprocessing2F
Iterator::Model�j+��ݓ?!��[K��D@)y�&1�|?1N6�d�M.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��ׁsF�?!۶m۶m5@)Ǻ���v?1>���>(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice"��u��q?!yu�2^�"@)"��u��q?1yu�2^�"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!o��M@)F%u�k?1$I�$I�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!͡bAs@)ŏ1w-!_?1͡bAs@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I+��?!�<��<�7@)/n��R?1�0�0@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9<�~���?I @+?ҨX@Q�7T+��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��8G3@��8G3@!��8G3@      ��!       "	����?����?!����?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	q9^���?q9^���?!q9^���?B      ��!       J	�lXSY�?�lXSY�?!�lXSY�?R      ��!       Z	�lXSY�?�lXSY�?!�lXSY�?b      ��!       JGPUY<�~���?b q @+?ҨX@y�7T+��?