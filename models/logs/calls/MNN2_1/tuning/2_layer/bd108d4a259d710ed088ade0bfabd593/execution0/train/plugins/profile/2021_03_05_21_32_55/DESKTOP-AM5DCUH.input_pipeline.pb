	�>�D�7@�>�D�7@!�>�D�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�>�D�7@��}q��4@1Ul��C�?A�~j�t��?Iޒ���@*	333333I@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate{�G�z�?!v]�u]�C@)Έ����?1=��<�sB@:Preprocessing2U
Iterator::Model::ParallelMapV2g��j+��?!n۶m�67@)g��j+��?1n۶m�67@:Preprocessing2F
Iterator::Model�l����?!�i��iZB@)_�Q�{?1������*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C|?!�a�a+@)a2U0*�s?1�0�0#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�|a2U�?!Y�eY��O@)��H�}m?1$I�$I�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!������@)�J�4a?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!�뺮�:E@)Ǻ���V?1��8��8@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!Y�eY�e�?)-C��6J?1Y�eY�e�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�0�0�?)a2U0*�C?1�0�0�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��83�nX@QF@�Y'@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��}q��4@��}q��4@!��}q��4@      ��!       "	Ul��C�?Ul��C�?!Ul��C�?*      ��!       2	�~j�t��?�~j�t��?!�~j�t��?:	ޒ���@ޒ���@!ޒ���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��83�nX@yF@�Y'@