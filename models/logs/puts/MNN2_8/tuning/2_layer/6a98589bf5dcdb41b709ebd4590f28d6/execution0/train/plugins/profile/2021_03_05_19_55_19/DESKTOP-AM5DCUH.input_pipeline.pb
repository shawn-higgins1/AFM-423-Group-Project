	ʇ�j�7@ʇ�j�7@!ʇ�j�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ʇ�j�7@Y�� `4@1߿yq��?A�b�=y�?I�	��@*	33333�M@2U
Iterator::Model::ParallelMapV2{�G�z�?!�Ω��@@){�G�z�?1�Ω��@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate� �	��?!�^B{	�9@)������?1�r�S�3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!�|ז�4@)��ׁsF�?1������0@:Preprocessing2F
Iterator::Model�?�߾�?!ؖ�`G@)���_vO~?1yTn�s�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!���|ז@)ŏ1w-!o?1���|ז@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����Mb�?!(iv��J@)-C��6j?1S����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!��fa��@)HP�s�b?1��fa��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�St$���?!�s�p5�;@)a2U0*�S?1�@�_) @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�pQ`z�X@Qbݣ�g��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Y�� `4@Y�� `4@!Y�� `4@      ��!       "	߿yq��?߿yq��?!߿yq��?*      ��!       2	�b�=y�?�b�=y�?!�b�=y�?:	�	��@�	��@!�	��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�pQ`z�X@ybݣ�g��?