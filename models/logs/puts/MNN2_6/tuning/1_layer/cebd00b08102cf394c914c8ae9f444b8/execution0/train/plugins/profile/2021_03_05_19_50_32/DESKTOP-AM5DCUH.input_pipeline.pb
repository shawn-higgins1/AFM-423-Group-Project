	 Uܸ=5@ Uܸ=5@! Uܸ=5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails- Uܸ=5@��ǘ��2@1$��ŋ��?A�����?I�Ss����?*	�����G@2U
Iterator::Model::ParallelMapV2�HP��?!��9T,h:@)�HP��?1��9T,h:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!��n��;@)��_vO�?1���cj`7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�*�?!p-��[K6@)9��v��z?1ۍ��v#,@:Preprocessing2F
Iterator::ModelΈ����?!��8+?!D@)-C��6z?1��n��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!͡bAs @)ŏ1w-!o?1͡bAs @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!{����M@)_�Q�k?1���D�o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�Cł�P@)����Mb`?1�Cł�P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!����8@)/n��R?1�0�0@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIV��H�X@Ql*�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ǘ��2@��ǘ��2@!��ǘ��2@      ��!       "	$��ŋ��?$��ŋ��?!$��ŋ��?*      ��!       2	�����?�����?!�����?:	�Ss����?�Ss����?!�Ss����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qV��H�X@yl*�����?