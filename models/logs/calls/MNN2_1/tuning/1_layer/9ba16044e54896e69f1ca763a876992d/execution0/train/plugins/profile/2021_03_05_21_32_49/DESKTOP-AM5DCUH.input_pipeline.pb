	Afg�;M6@Afg�;M6@!Afg�;M6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Afg�;M6@����3@1S=��M�?A���JY��?Ien���@*	gffff&H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�:pΈ�?!	����B@)�o_��?1,��A�IA@:Preprocessing2U
Iterator::Model::ParallelMapV2�]K�=�?!�%~F�;@)�]K�=�?1�%~F�;@:Preprocessing2F
Iterator::Model��A�f�?!�&c��E@)ŏ1w-!?1�f*�Px/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ���v?!s��\;0'@)���_vOn?1}�lqN�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!O��H]L@)��_vOf?1.>9\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�f*�Px@)ŏ1w-!_?1�f*�Px@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapjM�?!.8��C@)����MbP?1���f*� @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!���C��?)-C��6J?1���C��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!���2��?)a2U0*�C?1���2��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��Ǘ�X@Q�=�|�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����3@����3@!����3@      ��!       "	S=��M�?S=��M�?!S=��M�?*      ��!       2	���JY��?���JY��?!���JY��?:	en���@en���@!en���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��Ǘ�X@y�=�|�?