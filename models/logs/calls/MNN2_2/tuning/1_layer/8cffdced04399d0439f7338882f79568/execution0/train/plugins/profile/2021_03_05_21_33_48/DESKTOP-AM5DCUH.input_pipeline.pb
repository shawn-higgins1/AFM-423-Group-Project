	�<�r6@�<�r6@!�<�r6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�<�r6@bMeQء3@1o�j{�?A,e�X�?Ikׄ��`@*	�����LF@2U
Iterator::Model::ParallelMapV2�HP��?!�zʰZ;@)�HP��?1�zʰZ;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!����9@)HP�sׂ?1eX���4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateM�O��?!*��5O�6@)-C��6z?1�����,@:Preprocessing2F
Iterator::Model�l����?!�-(�j�D@) �o_�y?1xI@,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice���_vOn?!�$ ��� @)���_vOn?1�$ ��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}гY���?!��<�BM@)��H�}m?1&�D�$ @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�ݡ��@)����Mb`?1�ݡ��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�St$�?!�ZoAV9@)a2U0*�S?13��[P�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�S�ɹ�X@Q�V#�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	bMeQء3@bMeQء3@!bMeQء3@      ��!       "	o�j{�?o�j{�?!o�j{�?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	kׄ��`@kׄ��`@!kׄ��`@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�S�ɹ�X@y�V#�?