	0עhW5@0עhW5@!0עhW5@	>nlm�S�?>nlm�S�?!>nlm�S�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60עhW5@)?���2@1�7�n��?A'�����?I�d73�� @Yyxρ�i?*	     K@2U
Iterator::Model::ParallelMapV2?W[���?!h/����;@)?W[���?1h/����;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�St$���?!�Kh/��>@)9��v���?1/����8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM�St$�?!�^B{	�4@)Έ����?1��8��81@:Preprocessing2F
Iterator::Model��JY�8�?!{	�%�D@)F%u�{?1�q�q(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!������@)��H�}m?1������@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!���K�M@)��_vOf?1      @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�Kh/�@)����Mb`?1�Kh/�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<��?!�q��@@)_�Q�[?1��Kh/	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9@nlm�S�?I��m���X@Q}���n��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	)?���2@)?���2@!)?���2@      ��!       "	�7�n��?�7�n��?!�7�n��?*      ��!       2	'�����?'�����?!'�����?:	�d73�� @�d73�� @!�d73�� @B      ��!       J	yxρ�i?yxρ�i?!yxρ�i?R      ��!       Z	yxρ�i?yxρ�i?!yxρ�i?b      ��!       JGPUY@nlm�S�?b q��m���X@y}���n��?