	��`�$Q3@��`�$Q3@!��`�$Q3@	�dȼ��u?�dȼ��u?!�dȼ��u?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��`�$Q3@D�.l��1@1�����?A$����ۧ?I��oD���?Y������P?*	������K@2U
Iterator::Model::ParallelMapV2�N@aÓ?!�\*[A@)�N@aÓ?1�\*[A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!b�����7@)Ǻ����?17���$4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!��/r�7@)�5�;Nс?1�9ÂK/@:Preprocessing2F
Iterator::ModelF%u��?!�a����G@)�ZӼ�}?1P�&!�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!?ZMB@)	�^)�p?1?ZMB@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<,Ԛ�?!c�>ZMBJ@)a��+ei?1tc�>ZM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!S��u@7@)�J�4a?1S��u@7@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�dȼ��u?I'��E�X@Q�������?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	D�.l��1@D�.l��1@!D�.l��1@      ��!       "	�����?�����?!�����?*      ��!       2	$����ۧ?$����ۧ?!$����ۧ?:	��oD���?��oD���?!��oD���?B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUY�dȼ��u?b q'��E�X@y�������?