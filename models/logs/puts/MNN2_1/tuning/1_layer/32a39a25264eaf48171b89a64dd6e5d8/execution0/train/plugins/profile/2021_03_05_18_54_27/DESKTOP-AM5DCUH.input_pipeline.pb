	v�+.��5@v�+.��5@!v�+.��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-v�+.��5@$0��K3@1䠄���?A�3��7�?I����=@*23333�H@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate;�O��n�?!c`��7B@)�St$���?1�la�6�@@:Preprocessing2U
Iterator::Model::ParallelMapV2���_vO�?!Z_|���=@)���_vO�?1Z_|���=@:Preprocessing2F
Iterator::Modelj�t��?!.�ݦ��E@)S�!�uq{?1r~*! +@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����w?!��7V{'@)ŏ1w-!o?1O��N��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!�3"Y&=L@)��_vOf?1�ݦ���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���1@)����Mb`?1���1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�j+��ݓ?!+Z_|��C@)Ǻ���V?1�j�o�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!���2��?)-C��6J?1���2��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�6��n�?)a2U0*�C?1�6��n�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�kȧ�X@Q�_:�V�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$0��K3@$0��K3@!$0��K3@      ��!       "	䠄���?䠄���?!䠄���?*      ��!       2	�3��7�?�3��7�?!�3��7�?:	����=@����=@!����=@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�kȧ�X@y�_:�V�?