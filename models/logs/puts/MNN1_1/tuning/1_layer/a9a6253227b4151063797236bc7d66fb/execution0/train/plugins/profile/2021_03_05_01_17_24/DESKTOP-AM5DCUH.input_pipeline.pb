	�w�W�5@�w�W�5@!�w�W�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�w�W�5@����O3@1 ���-��?AHP�s�?I4���2@*�����K@)       =2U
Iterator::Model::ParallelMapV2���H�?!��Z@�c=@)���H�?1��Z@�c=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���Q��?!�.�{��;@)46<�R�?1 �/�%4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!�+�,U�6@)��ׁsF�?1{aZu�L2@:Preprocessing2F
Iterator::Model������?!D�"qE@)�<,Ԛ�}?1�7���*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!a�E�qP@)	�^)�p?1a�E�qP@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��?��?!����L@)��_�Le?1=E�A9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!9(g޲�@)a2U0*�c?19(g޲�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����Mb�?!_�V*�=@)����MbP?1_�V*��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIG9��X@Q�\���|�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����O3@����O3@!����O3@      ��!       "	 ���-��? ���-��?! ���-��?*      ��!       2	HP�s�?HP�s�?!HP�s�?:	4���2@4���2@!4���2@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qG9��X@y�\���|�?