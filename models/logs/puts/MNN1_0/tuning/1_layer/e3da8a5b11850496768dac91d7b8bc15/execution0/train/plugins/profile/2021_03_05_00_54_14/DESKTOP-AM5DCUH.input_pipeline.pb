	ݴ�!.6@ݴ�!.6@!ݴ�!.6@	g�7&#�?g�7&#�?!g�7&#�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ݴ�!.6@�_"�:�3@1�_���?A�b�=y�?I>�٬��@Y��IӠhn?*	     �K@2U
Iterator::Model::ParallelMapV2?W[���?!t�E]t;@)?W[���?1t�E]t;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�q����?!]t�E]<@)tF��_�?1��.��5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?�߾�?!颋.��8@)g��j+��?1F]t�E5@:Preprocessing2F
Iterator::Model'�����?!t�E]tC@) �o_�y?1颋.��&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!颋.��@)���_vOn?1颋.��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�4�?!�.�袋N@)�~j�t�h?1�E]t�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!]t�E@)����Mb`?1]t�E@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX�5�;N�?!�袋.�>@)��_�LU?1颋.��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9g�7&#�?I&����X@Q�_�w|�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�_"�:�3@�_"�:�3@!�_"�:�3@      ��!       "	�_���?�_���?!�_���?*      ��!       2	�b�=y�?�b�=y�?!�b�=y�?:	>�٬��@>�٬��@!>�٬��@B      ��!       J	��IӠhn?��IӠhn?!��IӠhn?R      ��!       Z	��IӠhn?��IӠhn?!��IӠhn?b      ��!       JGPUYg�7&#�?b q&����X@y�_�w|�?