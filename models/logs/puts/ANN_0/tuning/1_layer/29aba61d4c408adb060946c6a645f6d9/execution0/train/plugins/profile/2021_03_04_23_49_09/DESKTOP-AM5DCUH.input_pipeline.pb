	�V����2@�V����2@!�V����2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�V����2@m���1@1�-�v���?A�&OYMף?I.�R��?*	33333�J@2U
Iterator::Model::ParallelMapV2��ǘ���?!mϤ?>@)��ǘ���?1mϤ?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!���s�M8@)A��ǘ��?1��K3��4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�<,Ԛ�?!�}U�R;@)46<�R�?1�Q<��84@:Preprocessing2F
Iterator::Model������?!��qCv�E@)y�&1�|?1f�'�Y�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!��dDPu@)���_vOn?1��dDPu@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����o�?!q|���zL@)�~j�t�h?14鏃qC@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��B�@)����Mb`?1��B�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�Ԗ�X@Q\�J���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	m���1@m���1@!m���1@      ��!       "	�-�v���?�-�v���?!�-�v���?*      ��!       2	�&OYMף?�&OYMף?!�&OYMף?:	.�R��?.�R��?!.�R��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�Ԗ�X@y\�J���?