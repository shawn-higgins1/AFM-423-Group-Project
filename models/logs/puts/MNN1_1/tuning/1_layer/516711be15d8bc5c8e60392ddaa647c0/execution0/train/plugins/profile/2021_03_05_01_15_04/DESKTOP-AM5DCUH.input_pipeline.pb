	�`U��~5@�`U��~5@!�`U��~5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�`U��~5@����3@1ep��:��?A�� ��z�?I���o@*	33333sI@2U
Iterator::Model::ParallelMapV2	�^)ː?!�_?*@@)	�^)ː?1�_?*@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?�߾�?!�W�_�:@)g��j+��?1-�z�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!�rL�5@)y�&1�|?1^>��A�+@:Preprocessing2F
Iterator::ModelZd;�O��?!����F@)F%u�{?1nI�Y��)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!M3"l�@)���_vOn?1M3"l�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!_?*hK@){�G�zd?1�uN4x�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�"�&o@)����Mb`?1�"�&o@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!-�z�6@)����MbP?1�"�&o�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�ա�h�X@Q���W���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����3@����3@!����3@      ��!       "	ep��:��?ep��:��?!ep��:��?*      ��!       2	�� ��z�?�� ��z�?!�� ��z�?:	���o@���o@!���o@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�ա�h�X@y���W���?