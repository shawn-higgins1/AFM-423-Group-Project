	�΅�^�3@�΅�^�3@!�΅�^�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�΅�^�3@��-2@1nk�K�?A�,C��?IZ�!���?*	fffff�I@2U
Iterator::Model::ParallelMapV2������?!������@@)������?1������@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!u�UTW9@)'�����?1���6��4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!����7@)� �	�?1O`?��.@:Preprocessing2F
Iterator::ModelM�St$�?!o ���F@)�I+�v?1�Q\Gq%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice;�O��nr?!���b:�!@);�O��nr?1���b:�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO��e�c�?!��@~�K@)y�&1�l?1����ZJ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!��{�@)HP�s�b?1��{�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIE����X@Q�.^�@��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��-2@��-2@!��-2@      ��!       "	nk�K�?nk�K�?!nk�K�?*      ��!       2	�,C��?�,C��?!�,C��?:	Z�!���?Z�!���?!Z�!���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qE����X@y�.^�@��?