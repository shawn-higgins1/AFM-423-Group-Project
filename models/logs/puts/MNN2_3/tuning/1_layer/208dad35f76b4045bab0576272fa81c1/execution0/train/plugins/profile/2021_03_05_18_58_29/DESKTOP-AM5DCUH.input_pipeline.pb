	�)���4@�)���4@!�)���4@	���=&�?���=&�?!���=&�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�)���4@8���2@1�A'���?A��J?��?I6rݔ��?YOϻ��0�?*	������G@2U
Iterator::Model::ParallelMapV2��@��ǈ?!5L��]9@)��@��ǈ?15L��]9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!ܴ���<@)46<�R�?1|�e*��6@:Preprocessing2F
Iterator::Model+�����?!A(�S�pD@)���_vO~?1���/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�j+��݃?!ä���U4@)a��+ey?1#aG7��)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!��7x�Y@)y�&1�l?1��7x�Y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�!��u��?!��F�K�M@)_�Q�k?1ܴ���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!��V�9�@)��_vOf?1��V�9�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!k���z7@)�~j�t�X?1<Eg@(	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���=&�?Iŷ��X@Qd��0���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	8���2@8���2@!8���2@      ��!       "	�A'���?�A'���?!�A'���?*      ��!       2	��J?��?��J?��?!��J?��?:	6rݔ��?6rݔ��?!6rݔ��?B      ��!       J	Oϻ��0�?Oϻ��0�?!Oϻ��0�?R      ��!       Z	Oϻ��0�?Oϻ��0�?!Oϻ��0�?b      ��!       JGPUY���=&�?b qŷ��X@yd��0���?