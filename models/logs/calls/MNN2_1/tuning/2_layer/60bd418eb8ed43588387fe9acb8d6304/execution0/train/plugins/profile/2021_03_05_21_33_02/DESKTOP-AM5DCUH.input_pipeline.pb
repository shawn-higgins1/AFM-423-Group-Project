	y!�h:@y!�h:@!y!�h:@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-y!�h:@U���y7@1Q��r���?AǺ���?I��b�$@*�����YH@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�N@aÓ?!3�͟��C@)/n���?1���h�B@:Preprocessing2U
Iterator::Model::ParallelMapV2 �o_Ή?!���h�9@) �o_Ή?1���h�9@:Preprocessing2F
Iterator::Model�ݓ��Z�?!��t�gC@) �o_�y?1���h�)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_�y?!���h�)@)"��u��q?1�ɀ=��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2�%䃞?!�EM�q�N@)�����g?1��Ľ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�\0�Vm@)����Mb`?1�\0�Vm@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0�*�?!Q��_&E@)��_�LU?1�r��Z@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��H�}M?!"@����?)��H�}M?1"@����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!;ǳƊH�?)-C��6J?1;ǳƊH�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�G���X@Q�u\�%�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	U���y7@U���y7@!U���y7@      ��!       "	Q��r���?Q��r���?!Q��r���?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	��b�$@��b�$@!��b�$@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�G���X@y�u\�%�?