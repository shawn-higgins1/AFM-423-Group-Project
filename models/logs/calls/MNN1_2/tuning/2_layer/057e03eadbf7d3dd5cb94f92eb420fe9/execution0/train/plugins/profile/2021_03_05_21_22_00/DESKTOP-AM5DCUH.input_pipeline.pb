	[��Ye9@[��Ye9@![��Ye9@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-[��Ye9@��S��5@1�q��rg�?A��0�*�?I��b�@*	�����H@2U
Iterator::Model::ParallelMapV2�!��u��?!��@��@=@)�!��u��?1��@��@=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!w��v��:@)Ǻ����?17�<7�<7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!��2��26@)y�&1�|?1F�E�-@:Preprocessing2F
Iterator::ModelˡE����?!��C��CE@)-C��6z?1W�W�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!������@)���_vOn?1������@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!?�?�L@)�����g?1����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�!��!�@)��H�}]?1�!��!�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!L8{L8{8@)/n��R?1��A��A@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI֘�{��X@Qo��!W�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��S��5@��S��5@!��S��5@      ��!       "	�q��rg�?�q��rg�?!�q��rg�?*      ��!       2	��0�*�?��0�*�?!��0�*�?:	��b�@��b�@!��b�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q֘�{��X@yo��!W�?