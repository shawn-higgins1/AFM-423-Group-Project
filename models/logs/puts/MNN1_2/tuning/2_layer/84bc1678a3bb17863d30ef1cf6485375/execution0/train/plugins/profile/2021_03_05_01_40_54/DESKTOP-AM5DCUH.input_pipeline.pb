	R&5�@7@R&5�@7@!R&5�@7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-R&5�@7@���4@1;�э���?A��J�H��?I�A��u@*	�����I@2U
Iterator::Model::ParallelMapV22U0*��?!}\?�ZV?@)2U0*��?1}\?�ZV?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*�?!�ޜy��7@)a2U0*��?1�!�c)3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�+e�X�?!H�R&�6@)���_vO~?1S�ޜy�-@:Preprocessing2F
Iterator::Model�z6�>�?!4��͙�F@)y�&1�|?1ٛ3O��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��_��@)����Mbp?1��_��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!�j2fXK@)a��+ei?1�@ݳ �@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!@�ZV��@)/n��b?1@�ZV��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!����9�8@)/n��R?1@�ZV��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����X@QH\��{�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���4@���4@!���4@      ��!       "	;�э���?;�э���?!;�э���?*      ��!       2	��J�H��?��J�H��?!��J�H��?:	�A��u@�A��u@!�A��u@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����X@yH\��{�?