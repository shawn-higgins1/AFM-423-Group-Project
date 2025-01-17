	�d��7�5@�d��7�5@!�d��7�5@	��w;��?��w;��?!��w;��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�d��7�5@["��{3@1	^��?A��p��?I�%�"�� @Y��Q�d�?*	������H@2U
Iterator::Model::ParallelMapV2�Pk�w�?!�#k0��;@)�Pk�w�?1�#k0��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!���S-(:@)�I+��?1eF��!6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�St$�?!�2m셼6@)�q����?1I��ic/@:Preprocessing2F
Iterator::Model�g��s��?!텼��SE@)�<,Ԛ�}?1��jR`-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!i�́D+@)y�&1�l?1i�́D+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�v��/�?!zCM
�L@)�~j�t�h?1�&%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!`��n�@)����Mb`?1`��n�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!f�	��&9@)a2U0*�S?1@���P@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��w;��?I����
�X@Q��;N?�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	["��{3@["��{3@!["��{3@      ��!       "		^��?	^��?!	^��?*      ��!       2	��p��?��p��?!��p��?:	�%�"�� @�%�"�� @!�%�"�� @B      ��!       J	��Q�d�?��Q�d�?!��Q�d�?R      ��!       Z	��Q�d�?��Q�d�?!��Q�d�?b      ��!       JGPUY��w;��?b q����
�X@y��;N?�?