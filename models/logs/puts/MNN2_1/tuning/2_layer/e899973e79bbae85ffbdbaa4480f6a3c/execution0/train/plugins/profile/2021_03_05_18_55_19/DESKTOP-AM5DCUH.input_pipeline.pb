	�'�$�7@�'�$�7@!�'�$�7@	��.e��?��.e��?!��.e��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�'�$�7@F{���4@1��qS�?A�E���Ԩ?I�ӀA�g @YT�^P�?*	hffff�I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��ׁsF�?!���C@)�:pΈ�?1�'H=�xA@:Preprocessing2U
Iterator::Model::ParallelMapV2���S㥋?!� U��:@)���S㥋?1� U��:@:Preprocessing2F
Iterator::ModelDio��ɔ?!�)�Y7�C@)_�Q�{?1�K<A*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!^��d޵(@)U���N@s?1Y�	R�%"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!}���gN@)a��+ei?1��q/�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I+��?!��K<E@)/n��b?1�P��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!�K<A
@)_�Q�[?1�K<A
@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!5��U��?)����MbP?15��U��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�Y7�"��?)Ǻ���F?1�Y7�"��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��.e��?Id5(���X@QZU.�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	F{���4@F{���4@!F{���4@      ��!       "	��qS�?��qS�?!��qS�?*      ��!       2	�E���Ԩ?�E���Ԩ?!�E���Ԩ?:	�ӀA�g @�ӀA�g @!�ӀA�g @B      ��!       J	T�^P�?T�^P�?!T�^P�?R      ��!       Z	T�^P�?T�^P�?!T�^P�?b      ��!       JGPUY��.e��?b qd5(���X@yZU.�?