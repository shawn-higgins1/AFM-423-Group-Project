	�"����7@�"����7@!�"����7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�"����7@�j,am�4@1��)1	�?A��ڊ�e�?I;]��k@*	     �G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��y�):�?!�Q�٨�B@)	�^)ː?1��
brA@:Preprocessing2U
Iterator::Model::ParallelMapV2��@��ǈ?!�F}g��9@)��@��ǈ?1�F}g��9@:Preprocessing2F
Iterator::Model�ݓ��Z�?!�w6�;D@)_�Q�{?1�Q�٨�,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��0�*x?!�w6�;)@)	�^)�p?1��
br!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?Ɯ?!,����M@)�~j�t�h?1&W�+�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��F}g�@)��H�}]?1��F}g�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*��?!Q�٨�lD@)Ǻ���V?1�٨�l�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!m�w6�;�?)-C��6J?1m�w6�;�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!Q�٨�l�?)a2U0*�C?1Q�٨�l�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI=��]�X@Q�������?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�j,am�4@�j,am�4@!�j,am�4@      ��!       "	��)1	�?��)1	�?!��)1	�?*      ��!       2	��ڊ�e�?��ڊ�e�?!��ڊ�e�?:	;]��k@;]��k@!;]��k@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q=��]�X@y�������?