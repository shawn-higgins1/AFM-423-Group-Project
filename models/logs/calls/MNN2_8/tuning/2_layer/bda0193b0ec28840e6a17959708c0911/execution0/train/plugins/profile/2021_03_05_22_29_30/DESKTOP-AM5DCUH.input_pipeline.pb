	W���xU7@W���xU7@!W���xU7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-W���xU7@��F��r4@1�~�T���?A,e�X�?IG�˵h@*	�����M@2U
Iterator::Model::ParallelMapV2Έ����?!�>�1�@@)Έ����?1�>�1�@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�� �rh�?!񹒗�B=@)���S㥋?1��Ni]<7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!]V��F3@)�&S��?1W��FS/@:Preprocessing2F
Iterator::Model-C��6�?!�`�F@)y�&1�|?1�k��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�k��@)y�&1�l?1�k��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�p=
ף�?!r���K@)-C��6j?1�`�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!���h�@)�J�4a?1���h�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�:pΈ�?!#��7'?@)/n��R?1$c��J�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIn�%`��X@Qj���gP�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��F��r4@��F��r4@!��F��r4@      ��!       "	�~�T���?�~�T���?!�~�T���?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	G�˵h@G�˵h@!G�˵h@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qn�%`��X@yj���gP�?