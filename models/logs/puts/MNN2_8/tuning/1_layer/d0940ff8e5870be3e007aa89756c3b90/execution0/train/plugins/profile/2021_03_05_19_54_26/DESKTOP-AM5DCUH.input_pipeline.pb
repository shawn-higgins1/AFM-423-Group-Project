	���N�z5@���N�z5@!���N�z5@	�~�OJ�?�~�OJ�?!�~�OJ�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���N�z5@���%�3@1�����?AǺ���?IdyW=`�?YT�qs*�?*	����̌M@2U
Iterator::Model::ParallelMapV2HP�sג?!\óO8"?@)HP�sג?1\óO8"?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX�5�;N�?!�i4Iy�<@)��<,Ԋ?1���8s*6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�@%6�&5@)M�O��?1G�+�1@:Preprocessing2F
Iterator::Modelݵ�|г�?!v�\�5<E@)S�!�uq{?1$C:f�&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!VH�A�@)ŏ1w-!o?1VH�A�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�� �rh�?!�M�I��L@)F%u�k?1t{-9�U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!q�)`>@)a2U0*�c?1q�)`>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�&S��?!���N��>@)��_�LU?1Џ-�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�~�OJ�?I!WS�zX@Q�}�X,9 @Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���%�3@���%�3@!���%�3@      ��!       "	�����?�����?!�����?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	dyW=`�?dyW=`�?!dyW=`�?B      ��!       J	T�qs*�?T�qs*�?!T�qs*�?R      ��!       Z	T�qs*�?T�qs*�?!T�qs*�?b      ��!       JGPUY�~�OJ�?b q!WS�zX@y�}�X,9 @