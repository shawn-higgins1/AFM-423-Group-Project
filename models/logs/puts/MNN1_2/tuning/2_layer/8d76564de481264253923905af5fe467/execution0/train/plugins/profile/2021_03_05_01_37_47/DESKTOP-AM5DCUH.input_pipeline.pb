	kծ	57@kծ	57@!kծ	57@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-kծ	57@;���4@1��0���?A����z�?I���#W @*	33333�I@2U
Iterator::Model::ParallelMapV2���H�?!\
�IƢ>@)���H�?1\
�IƢ>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!�����8@)��_vO�?1���*�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea��+e�?!�N��`�7@)"��u���?1��� �0@:Preprocessing2F
Iterator::Model�z6�>�?!�eu�E@)_�Q�{?1F~ I4*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!��P�~I@)ŏ1w-!o?1��P�~I@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<,Ԛ�?!P�ᚊ!L@)�~j�t�h?17��<@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!H4:��@)����Mb`?1H4:��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS�!�uq�?!1�H��9@)����MbP?1H4:���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��-�_�X@QȄ4
��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	;���4@;���4@!;���4@      ��!       "	��0���?��0���?!��0���?*      ��!       2	����z�?����z�?!����z�?:	���#W @���#W @!���#W @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��-�_�X@yȄ4
��?