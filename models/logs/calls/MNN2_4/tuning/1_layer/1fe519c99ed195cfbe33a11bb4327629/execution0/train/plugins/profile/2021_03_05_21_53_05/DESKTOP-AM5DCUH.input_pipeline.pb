	U��785@U��785@!U��785@	K��+��?K��+��?!K��+��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6U��785@��F!�2@1���&���?A�lV}��?Iɏ�k @Yݴ�!��?*	������H@2U
Iterator::Model::ParallelMapV2���QI�?!��u��<@)���QI�?1��u��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�!��u��?!G�/��^<@)��@��ǈ?1�A�w�X8@:Preprocessing2F
Iterator::Model'�����?!��x�E@)�ZӼ�}?1$I�$I�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�*�?!R`��n�4@)F%u�{?1yv��1�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!Y��W�@)���_vOn?1Y��W�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!5���xL@)Ǻ���f?1!�
��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!`��n�@)����Mb`?1`��n�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!ݔ�=�6@)/n��R?1P���˴@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9K��+��?I�D�78�X@Qgl��T�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��F!�2@��F!�2@!��F!�2@      ��!       "	���&���?���&���?!���&���?*      ��!       2	�lV}��?�lV}��?!�lV}��?:	ɏ�k @ɏ�k @!ɏ�k @B      ��!       J	ݴ�!��?ݴ�!��?!ݴ�!��?R      ��!       Z	ݴ�!��?ݴ�!��?!ݴ�!��?b      ��!       JGPUYK��+��?b q�D�78�X@ygl��T�?