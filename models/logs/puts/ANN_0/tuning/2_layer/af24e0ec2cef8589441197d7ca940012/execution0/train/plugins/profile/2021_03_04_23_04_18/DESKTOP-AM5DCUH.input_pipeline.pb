	?�ܹ3@?�ܹ3@!?�ܹ3@	�l�s?�l�s?!�l�s?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?�ܹ3@2���1@16sHj�d�?AX˝�`8�?I3�68��?YiUMuO?*	������F@2U
Iterator::Model::ParallelMapV2 �o_Ή?!�k(��;@) �o_Ή?1�k(��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!��t��;@)M�O��?1c:��,&6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�g��s��?!�#���>7@)��H�}}?1Cy�5�/@:Preprocessing2F
Iterator::ModeljM�?!�YLg�D@)9��v��z?1$���>�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice_�Q�k?!1��t�@)_�Q�k?11��t�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�A`��"�?!�,���M@)-C��6j?1�}�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!�,����@)��_�Le?1�,����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�l�s?I4uN+�X@Q`FWZPa�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2���1@2���1@!2���1@      ��!       "	6sHj�d�?6sHj�d�?!6sHj�d�?*      ��!       2	X˝�`8�?X˝�`8�?!X˝�`8�?:	3�68��?3�68��?!3�68��?B      ��!       J	iUMuO?iUMuO?!iUMuO?R      ��!       Z	iUMuO?iUMuO?!iUMuO?b      ��!       JGPUY�l�s?b q4uN+�X@y`FWZPa�?