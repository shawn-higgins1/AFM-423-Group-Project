	~(F�`5@~(F�`5@!~(F�`5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-~(F�`5@؞Y�3@1+�����?A��+e�?I,~SX�` @*	43333�H@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ǘ���?!�6W҂;@@)�ZӼ��?1���M�t<@:Preprocessing2U
Iterator::Model::ParallelMapV2������?!aq*?7@)������?1aq*?7@:Preprocessing2F
Iterator::Model�l����?!�u��}�B@)lxz�,C|?1:��o§+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_vO�?!���m��5@)F%u�{?1���s*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!�%�8k� @)�J�4q?1�%�8k� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2U0*��?! �O�vO@)-C��6j?1�-}Ļ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�<�Z5@)����Mb`?1�<�Z5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!�Z5P8@)a2U0*�S?1?�]�=@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIc���X@Q�΀"?�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	؞Y�3@؞Y�3@!؞Y�3@      ��!       "	+�����?+�����?!+�����?*      ��!       2	��+e�?��+e�?!��+e�?:	,~SX�` @,~SX�` @!,~SX�` @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qc���X@y�΀"?�?