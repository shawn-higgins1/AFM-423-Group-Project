	]��J�7@]��J�7@!]��J�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-]��J�7@v���/~4@1x�-;�?�?A|�Pk��?I�0{�vZ@*	33333�F@2U
Iterator::Model::ParallelMapV2Zd;�O��?!Y&
ݔT9@)Zd;�O��?1Y&
ݔT9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!��hAj<@)'�����?1���z�7@:Preprocessing2F
Iterator::Model�:pΈ�?!�^dm�C@)F%u�{?1<.}�+-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ׁsF�?!���`�5@) �o_�y?1ݔT���+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!�`��ҷ@)��H�}m?1�`��ҷ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!E����N@)-C��6j?1Sb�1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!)tSRb@)/n��b?1)tSRb@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ����?!���"k�8@)��_�LU?1ӷ�2Q�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIR�[^�X@Qh+�x��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	v���/~4@v���/~4@!v���/~4@      ��!       "	x�-;�?�?x�-;�?�?!x�-;�?�?*      ��!       2	|�Pk��?|�Pk��?!|�Pk��?:	�0{�vZ@�0{�vZ@!�0{�vZ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qR�[^�X@yh+�x��?