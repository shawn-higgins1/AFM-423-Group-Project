	�2��m3@�2��m3@!�2��m3@	1U�#M�?1U�#M�?!1U�#M�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�2��m3@��f�+1@1I��_���?A��_vO�?I�~NA~V @YM.��:�o?*	����̌I@2U
Iterator::Model::ParallelMapV2HP�sג?! @� B@)HP�sג?1 @� B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!ݺu�֭7@)��ZӼ�?1_�~���3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!J�*U�T5@)�ZӼ�}?1�<y���+@:Preprocessing2F
Iterator::Model�(��0�?! A�	H@)a��+ey?1B�"D(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!�۷o߾@)ŏ1w-!o?1�۷o߾@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�A`��"�?!�}���I@)y�&1�l?1[�lٲe@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�۷o߾@)ŏ1w-!_?1�۷o߾@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no91U�#M�?I��Pc�X@Q �n���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��f�+1@��f�+1@!��f�+1@      ��!       "	I��_���?I��_���?!I��_���?*      ��!       2	��_vO�?��_vO�?!��_vO�?:	�~NA~V @�~NA~V @!�~NA~V @B      ��!       J	M.��:�o?M.��:�o?!M.��:�o?R      ��!       Z	M.��:�o?M.��:�o?!M.��:�o?b      ��!       JGPUY1U�#M�?b q��Pc�X@y �n���?