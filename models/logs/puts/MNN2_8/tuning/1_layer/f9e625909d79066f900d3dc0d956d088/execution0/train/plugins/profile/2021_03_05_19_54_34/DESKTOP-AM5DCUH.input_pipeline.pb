	W#��2�7@W#��2�7@!W#��2�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-W#��2�7@>�#dx5@1�J��?A5�8EGr�?I-�i��� @*	������G@2U
Iterator::Model::ParallelMapV2���QI�?!��
��
>@)���QI�?1��
��
>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!"�k"�k9@)M�O��?1��7��75@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!2�z1�z6@)_�Q�{?1$I�$I�,@:Preprocessing2F
Iterator::Model��_�L�?!�F��F�E@)9��v��z?1��O��O+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!>�b>�b @)�q����o?1>�b>�b @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!�&�&L@)a��+ei?1W�V�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!a��`��@)����Mb`?1a��`��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!̧^̧^8@)��H�}M?1��@��@�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���X@QH���r�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>�#dx5@>�#dx5@!>�#dx5@      ��!       "	�J��?�J��?!�J��?*      ��!       2	5�8EGr�?5�8EGr�?!5�8EGr�?:	-�i��� @-�i��� @!-�i��� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���X@yH���r�?