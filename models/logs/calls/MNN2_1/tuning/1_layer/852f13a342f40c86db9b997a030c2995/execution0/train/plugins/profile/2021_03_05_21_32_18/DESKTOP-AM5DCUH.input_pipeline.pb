	ϟ6�ӱO@ϟ6�ӱO@!ϟ6�ӱO@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ϟ6�ӱO@��6ʆN@1�=@����?A.����?I�R$_	� @*	������J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate=�U����?!(�$��pF@)Ǻ���?1��n�J�D@:Preprocessing2U
Iterator::Model::ParallelMapV2�(��0�?!���6@)�(��0�?1���6@:Preprocessing2F
Iterator::ModeljM�?!�rg?��A@)_�Q�{?1f�a+mS)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����w?!��;��%@)����Mbp?1K�T~��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe�X��?!�FL`�P@)a��+ei?1�)�\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!]Ų���
@)��H�}]?1]Ų���
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapp_�Q�?!�uq��G@)-C��6Z?1o�e�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��H�}M?!]Ų����?)��H�}M?1]Ų����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!o�e��?)-C��6J?1o�e��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIe��G��X@Q�{�#�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��6ʆN@��6ʆN@!��6ʆN@      ��!       "	�=@����?�=@����?!�=@����?*      ��!       2	.����?.����?!.����?:	�R$_	� @�R$_	� @!�R$_	� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qe��G��X@y�{�#�?